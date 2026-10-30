import os
import json
import random
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models.concatenation.concatenation_method import ConcatenationMethod
from models.combiner.combiner_method import CombinerMethod
from models.limn.limn_method import LIMNMethod
from models.tg_cir.tg_cir_method import TG_CIRMethod

from training_helpers.dataloader import TripletDataset, GeometricBalancedSampler
from training_helpers.arg_parser import parse_args
from utils.backbones import BACKBONE_DIMS

from dotenv import load_dotenv

load_dotenv()

TRIPLETS_FOLDER = os.getenv("TRIPLETS_FOLDER")
RUNS_FOLDER = os.getenv("RUNS_FOLDER")

#--------------------Defining Training and Testing Loops--------------------#
def train(method, dataloader, optimizer):
    method.model.train()
    epoch_loss = torch.zeros((), dtype=torch.float64, device=method.device)   # running sum, stays on the GPU
    num_batches = 0

    for batch_index, features in enumerate(dataloader):
        print(f"Training - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        loss = method.training_step(*features)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().double()     
        num_batches += 1

    epoch_loss = epoch_loss.item() / num_batches
    print(f"Epoch Training Loss: {epoch_loss:.4f}")

    return epoch_loss

@torch.no_grad()
def test(method, dataloader):
    method.eval()
    epoch_loss = torch.zeros((), dtype=torch.float64, device=method.device)   # running sum, stays on the GPU
    num_batches = 0

    for batch_index, features in enumerate(dataloader):
        print(f"Validating - Batch {batch_index+1}/{len(dataloader)}", end='\r')
        loss = method.training_step(*features)

        epoch_loss += loss.detach().double()
        num_batches += 1

    epoch_loss = epoch_loss.item() / num_batches
    print(f"Epoch Validating Loss: {epoch_loss:.4f}")
    return epoch_loss

#--------------------Main Function--------------------#
def main():
    #--------------------------------------------------Parsing and Logging--------------------------------------------------#
    args = parse_args()

    #resuming: the hyperparameters of the original run (settings.json) replace the command line ones
    if args.pretrained != '':
        with open(os.path.join(RUNS_FOLDER, args.pretrained, "settings.json")) as f:
            saved_settings = json.load(f)
        for key in ("model", "dataset", "backbone", "seed", "lr", "tau", "batch_size", "balance"):
            setattr(args, key, saved_settings[key])

    balance_str = "balanced" if args.balance else "unbalanced"

    tau_str = f"tau{args.tau}"
    lr_str = f"lr{args.lr}"

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    experiment_name = f"{args.dataset}_{args.backbone}_{args.model}_{args.seed}_{tau_str}_{lr_str}_{balance_str}_{timestamp}"
    
    if args.pretrained == '':
        save_dir = os.path.join(RUNS_FOLDER, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(RUNS_FOLDER, args.pretrained)
        print(f"Loading pretrained model from {save_dir}")

    log_path = os.path.join(save_dir, "log.txt")
    settings_path = os.path.join(save_dir, "settings.json")

    #-------------------------------------------Loading Data, Model and Settings---------------------------------------#
    train_path = f"{TRIPLETS_FOLDER}/{args.dataset.upper()}/train_triplets.csv"
    val_path = f"{TRIPLETS_FOLDER}/{args.dataset.upper()}/val_triplets.csv"

    #Setting random seeds for reproducibility    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    methods = {
        'concatenation': ConcatenationMethod,
        'combiner': CombinerMethod,
        'limn': LIMNMethod,
        'tg_cir': TG_CIRMethod
    }

    dims = BACKBONE_DIMS[args.backbone]
    method = methods[args.model](
        loss_tau=args.tau,
        global_dim=dims['global_tokens'],
        audio_local_dim=dims['local_audio_tokens'],
        text_local_dim=dims['local_text_tokens']
    )
    
    #Instantiate datasets
    train_dataset = TripletDataset(train_path, method.needs_local_tokens, args.backbone, split = 'train', dataset = args.dataset)
    val_dataset = TripletDataset(val_path, method.needs_local_tokens, args.backbone, split = 'val', dataset = args.dataset)

    #use or don't use the balanced sampler for training
    if args.balance:
        train_sampler = GeometricBalancedSampler(train_dataset.triplets_df, batch_size=args.batch_size, seed=seed)
        sampling_args = dict(batch_sampler=train_sampler)
    else:
        sampling_args = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(
        train_dataset,
        collate_fn=method.custom_collate_fn,
        num_workers=8,
        pin_memory=True,
        **sampling_args
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=True, #avoid batches with different sizes 
        collate_fn=method.custom_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    #Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method.to(device)

    #Load Model Weights
    if args.pretrained != '':
        method.load_checkpoint(save_dir)

        print(f"Loaded pretrained model weights from {save_dir}")
        print(f"Resuming training from epoch {method.start_epoch} with best loss {method.best_loss:.4f}")

    #Log Settings (read by evaluate.py and by the grid search, so it must stay machine-readable)
    settings = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model_architecture": str(method.model),
        "model": args.model,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "lr": args.lr,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "balance": args.balance,
        "epochs": args.epochs,

        "train_dataset": train_path,
        "validation_dataset": val_path,
        "n_training_triplets": len(train_loader.dataset),
        "n_training_songs": int(train_loader.dataset.get_num_songs()),
        "n_validation_triplets": len(val_loader.dataset),
        "n_validation_songs": int(val_loader.dataset.get_num_songs()),
        "audio_length_seconds": round(train_loader.dataset.get_audio_len(), 2),
        "n_trainable_parameters": sum(p.numel() for p in method.model.parameters() if p.requires_grad),
    }

    #a resumed run keeps the settings.json of the original run
    if args.pretrained == '':
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4)


    #--------------------------------------------------Training Loop--------------------------------------------------#
    #Define Optimizer
    optimizer = torch.optim.Adam(method.model.parameters(), lr=args.lr, weight_decay=1e-4)

    early_stop_patience = 5           
    early_stop_counter = 0             

    for epoch in range(method.start_epoch, method.start_epoch+args.epochs):

        with open(log_path, "a") as f:
            f.write(f"\n-----------Epoch {epoch+1}-----------\n")
        print(f"\n-----------Epoch {epoch+1}-----------\n")

        #train
        train_loss = train(method, train_loader, optimizer)
        with open(log_path, "a") as f:
            f.write('Train Loss: {:.4f}\n'.format(train_loss))

        #validate
        val_loss = test(method, val_loader)
        with open(log_path, "a") as f:
            f.write('Val Loss: {:.4f}\n'.format(val_loss))

        # the method keeps the losses and the best loss, and saves its own checkpoint
        method.update_epoch(epoch, train_loss, val_loss)
        method.save_checkpoint(save_dir)
        
        #---------- Early Stopping Logic ----------#
        if method.is_best:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        #---------- Early stopping condition ----------#
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs with no improvement.")
            break
        
if __name__ == "__main__":
    main()
