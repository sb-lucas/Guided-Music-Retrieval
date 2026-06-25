import os
import sys
import torch
import shutil
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--model', type=str, default='one_layer', help='Fusion Module Architecture Name')
    parser.add_argument('--dataset', type=str, default='mtiq', help='Dataset Name')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of Training Epochs')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model directory inside runs/')
    parser.add_argument('--return_local_tokens', action='store_true', help='Whether to return local tokens from the dataset')

    args = parser.parse_args()

    return args