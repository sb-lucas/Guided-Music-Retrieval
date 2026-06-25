import os
import sys
import torch
import shutil
import pickle

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
        
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)