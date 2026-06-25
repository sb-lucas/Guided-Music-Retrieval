import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.limn import LIMN
from models.one_layer import MLP
from models.baldrati import Baldrati
from models.tg_cir import TG_CIR

from training_helpers.dataloader import TripletDataset, GeometricBalancedSampler
from training_helpers.limn_custom_collate import limn_collate_fn
from training_helpers.tg_cir_custom_collate import tg_cir_collate_fn
from training_helpers.arg_parser import parse_args
from training_helpers.checkpoint import save_checkpoint
from training_helpers.plot import plot_losses

from losses.bbcl import BatchBasedClassificationLoss
from losses.limn_losses import LIMN_Loss
from losses.tg_cir_losses import TG_CIR_Loss

from model_evaluation import ModelEvaluationMTIQ, ModelEvaluationMTSQ

import os
import pandas as pd
from datetime import datetime
import time
import inspect


#----------------------Defining Inference Functions for Each Model----------------------#

def inference_tgcir(model, loss_func, device, local_ref, global_ref, local_text, global_text, local_tar, global_tar):
    #sends everything to the device
    local_ref, global_ref, local_text = local_ref.to(device), global_ref.to(device), local_text.to(device)
    global_text, local_tar, global_tar = global_text.to(device), local_tar.to(device), global_tar.to(device)

    #forward pass
    E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde = model(global_ref, global_text, global_tar, local_ref, local_text, local_tar)

    #loss
    loss = loss_func(E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde)

    return loss

# reference_song_local_tokens, reference_song_global_tokens, modification_text_local_tokens, 
# modification_text_global_tokens, target_song_local_tokens, target_song_global_tokens, target_id, reference_id
def inference_limn(model, loss_func, device, er, fr, em, fm, et, ft, er_len, em_len, et_len):
    # performs a forward pass and computes the loss
    er, fr, em, fm, et, ft = er.to(device), fr.to(device), em.to(device), fm.to(device), et.to(device), ft.to(device)
    er_len, em_len, et_len = er_len.to(device), em_len.to(device), et_len.to(device)

    eq_len = er_len + em_len
    x_hat_q, a_hat_q, x_hat_t, a_hat_t = model(fr, fm, ft, er, em, et, eq_len, et_len)

    loss = loss_func(a_hat_q, a_hat_t, model.condition_masks_q, model.condition_masks_t, x_hat_q, x_hat_t)

    return loss

def inference_mlp(model, loss_func, device, reference_song_embedding, modification_text_embedding, target_song_embedding):
    ref_song_embs, text_embs, target_embs = reference_song_embedding.to(device), modification_text_embedding.to(device), target_song_embedding.to(device)

    composed = model(ref_song_embs, text_embs)
    loss = loss_func(composed, target_embs)

    return loss

#--------------------Defining Training and Testing Loops--------------------#

def train(model, dataloader, optimizer, device, inference, loss_func):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for batch_index, features in enumerate(dataloader):
        print(f"Training - Batch {batch_index+1}/{len(dataloader)}", end='\r')

        loss = inference(model, loss_func, device, *features)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
    print(f"Epoch Training Loss: {epoch_loss:.4f}")

    return epoch_loss

@torch.no_grad()
def test(model, dataloader, device, inference, loss_func):
    model.eval()
    epoch_loss = 0
    num_batches = 0

    for batch_index, features in enumerate(dataloader):
        print(f"Testing - Batch {batch_index+1}/{len(dataloader)}", end='\r')
        loss = inference(model, loss_func, device, *features)

        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
    print(f"Epoch Testing Loss: {epoch_loss:.4f}")
    return epoch_loss

#--------------------Main Function--------------------#
def main():
    #--------------------------------------------------Parsing and Logging--------------------------------------------------#
    args = parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    experiment_name = f"{args.dataset}_{timestamp}_{args.model}"
    
    if args.pretrained == '':
        save_dir = os.path.join("runs", experiment_name)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join("runs", args.pretrained)
        print(f"Loading pretrained model from {save_dir}")

    log_path = os.path.join(save_dir, "log.txt")
    settings_path = os.path.join(save_dir, "settings.txt")

    #-------------------------------------------Loading Data, Model and Settings---------------------------------------#
    if args.dataset == 'mtiq':
        train_path = "path/to/triplets_mtiq_train.csv"
        val_path = "path/to/triplets_mtiq_val.csv"
    elif args.dataset == 'mtsq':
        train_path = "path/to/triplets_mtsq_train.csv"
        val_path = "path/to/triplets_mtsq_val.csv"

    #Select and Setup the chosen Model
    if args.return_local_tokens:
        if args.model == 'limn':
            model = LIMN()
            inference = inference_limn
            ccfn=limn_collate_fn
            loss_func = LIMN_Loss()
        elif args.model == 'tg_cir':
            model = TG_CIR()
            inference = inference_tgcir
            ccfn=tg_cir_collate_fn
            loss_func = TG_CIR_Loss()
    else:
        inference = inference_mlp
        ccfn = None
        loss_func = BatchBasedClassificationLoss()
        if args.model == 'one_layer':
            model = MLP(input_dim=512*2, output_dim=512)
        elif args.model == 'baldrati':
            model = Baldrati(clap_feature_dim=512, projection_dim=256, hidden_dim=128)

    #Setting random seeds for reproducibility    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #Load DataLoaders
    #1 - Instantiate datasets
    train_dataset = TripletDataset(train_path, args.return_local_tokens, split = 'train', dataset = args.dataset)
    val_dataset = TripletDataset(val_path, args.return_local_tokens, split = 'val', dataset = args.dataset)

    #2 - Instantiate samplers
    train_sampler = GeometricBalancedSampler(train_dataset.triplets_df, batch_size=args.batch_size, seed=seed)

    #3 - Load DataLoaders using batch_sampler
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,
        collate_fn=ccfn,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=True, #avoid batches with different sizes 
        collate_fn=ccfn,
        num_workers=8,
        pin_memory=True
    )

    #Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Load Model Weights
    if args.pretrained != '':
        model_weights_path = os.path.join(save_dir, "checkpoint.pth.tar")
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1

        best_loss = checkpoint.get('best_loss', float('inf'))

        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])

        print(f"Loaded pretrained model weights from {model_weights_path}")
        print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")
    else:
        is_best = True
        best_loss = float('inf')
        start_epoch = 0
        train_losses = []
        val_losses = []

        state = { 
            'epoch': start_epoch, 
            'state_dict': model.state_dict(), 
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        save_checkpoint(state, is_best, save_dir)

    #Log Settings
    with open(settings_path, "w") as f:
        f.write("Experiment Settings\n")

        f.write(f"Train Dataset: {train_path}\n")
        f.write(f"Validation Dataset: {val_path}\n")
        f.write(f"Number of training triplets: {len(train_loader.dataset)}\n")
        f.write(f"Number of training songs: {train_loader.dataset.get_num_songs()}\n")
        f.write(f"Number of validation triplets: {len(val_loader.dataset)}\n")
        f.write(f"Number of validation songs: {val_loader.dataset.get_num_songs()}\n")
        f.write(f"Audio length (seconds): {train_loader.dataset.get_audio_len():.2f}\n")

        f.write("\n")
        
        f.write(f"Model: {args.model}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Model Architecture: \n{str(model)}\n")

    #--------------------------------------------------Training Loop--------------------------------------------------#
    #Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    early_stop_patience = 5           
    early_stop_counter = 0             

    for epoch in range(start_epoch, start_epoch+args.epochs):

        with open(log_path, "a") as f:
            f.write(f"\n-----------Epoch {epoch+1}-----------\n")
        print(f"\n-----------Epoch {epoch+1}-----------\n")

        #train
        train_loss = train(model, train_loader, optimizer, device, inference, loss_func)
        with open(log_path, "a") as f:
            f.write('Train Loss: {:.4f}\n'.format(train_loss))
        train_losses.append(train_loss)

        #validate
        val_loss = test(model, val_loader, device, inference, loss_func)
        with open(log_path, "a") as f:
            f.write('Val Loss: {:.4f}\n'.format(val_loss))
        val_losses.append(val_loss)

        # remember best loss and save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        state = { 
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        save_checkpoint(state, is_best, save_dir)
        
        #---------- Early Stopping Logic ----------#
        if is_best:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        #---------- Early stopping condition ----------#
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs with no improvement.")
            break
        
    #-------------- Plotting Losses --------------#
    plot_losses(train_losses, val_losses, save_dir, label='Loss')

    #-------------- Evaluation --------------#
    weights_path = save_dir.split('/')[-1]
    if args.dataset == 'mtiq':
        evaluator = ModelEvaluationMTIQ(model_weights_path=weights_path, model_architecture=args.model, save_path=save_dir)
        evaluator.evaluate()

    elif args.dataset == 'mtsq':
        evaluator = ModelEvaluationMTSQ(model_weights_path=weights_path, model_architecture=args.model, save_path=save_dir)
        evaluator.evaluate()

if __name__ == "__main__":
    main()






















