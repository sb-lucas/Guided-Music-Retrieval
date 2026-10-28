import os
import sys
import torch
import shutil
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--dataset', type=str, default='mtsq', help='Dataset Name')

    parser.add_argument('--backbone', type=str, default='CLAP', help='Backbone that extracted the embeddings (its sizes are in utils/backbones.py)')

    parser.add_argument('--model', type=str, default='concatenation', help='Fusion Module Architecture Name')
    
    parser.add_argument('--seed', type=int, default=42, help='Training Seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Loss Hyperparameter')
    parser.add_argument('--balance', action='store_true', help='Whether to use a balanced sampling strategy')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of Training Epochs')
    
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model directory inside runs/')

    args = parser.parse_args()

    return args