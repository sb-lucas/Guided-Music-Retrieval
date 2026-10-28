import os
import shutil
import torch
from abc import ABC, abstractmethod

from training_helpers.bbcl import BatchBasedClassificationLoss

class BaseMethod(ABC):
    def __init__(self, loss_tau, global_dim, audio_local_dim=None, text_local_dim=None):
        #sizes of the embeddings of the backbone in use, provided by train.py. Set before build_model(), which needs them
        self.global_dim = global_dim
        self.audio_local_dim = audio_local_dim
        self.text_local_dim = text_local_dim

        self.model = self.build_model()
        self.device = torch.device("cpu") 
        self.needs_local_tokens = False
        self.loss_func = BatchBasedClassificationLoss(loss_tau)
        self.custom_collate_fn = None

        self.is_best = False        #whether the last finished epoch had the best validation loss
        self.best_loss = float('inf')
        self.epoch = -1             #last finished epoch
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []

    @abstractmethod
    def build_model(self):
        """returns the nn.Module of this method"""

    def to(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def to_device(self, *tensors):
        #the method moves its inputs to its device, so the caller does not have to
        return tuple(tensor.to(self.device) for tensor in tensors)

    def training_step(self, ref_global, mod_text_global, target_global, query_id, target_id):
        ref_global, mod_text_global, target_global, query_id, target_id = self.to_device(ref_global, mod_text_global, target_global, query_id, target_id)

        composed = self.model(ref_global, mod_text_global)
        loss = self.loss_func(composed, target_global, query_id, target_id)

        return loss

    def load_checkpoint(self, path, checkpoint_name='checkpoint'):   # overwrites the weights and the training state of this method
        '''
        checkpoint_name: 'checkpoint' (last epoch) or 'model_best'
        '''

        model_weights_path = os.path.join(path, f"{checkpoint_name}.pth.tar")
        checkpoint = torch.load(model_weights_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.start_epoch = self.epoch + 1
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

    @torch.no_grad()
    def compose(self, queries):
        #queries: dict from EvaluationSet.get_queries
        self.eval()     #dropout must be off to compose queries
        ref_global, mod_text_global = self.to_device(queries['ref_global'], queries['mod_text_global'])

        composed_embedding = self.model(ref_global, mod_text_global)

        return composed_embedding

    def encode_pool(self, audio_cache, song_ids):
        #cases when the model does not encode the target, and uses CLAP directly
        #returns the embeddings of the tracks of the pool, shape [len(song_ids), D], in the order of song_ids
        return torch.stack([audio_cache[song_id][0] for song_id in song_ids])

    def update_epoch(self, epoch, train_loss, val_loss):
        """Keeps the training state up to date after each epoch"""
        self.epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.is_best = val_loss < self.best_loss
        self.best_loss = min(val_loss, self.best_loss)

    def save_checkpoint(self, save_dir, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        os.makedirs(save_dir, exist_ok=True)

        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(state, checkpoint_path)
        if self.is_best:
            best_path = os.path.join(save_dir, 'model_best.pth.tar')
            shutil.copyfile(checkpoint_path, best_path)