import os
import sys
import torch
import shutil
import matplotlib.pyplot as plt
import pickle

def plot_losses(train_losses, val_losses, directory, label):
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Train {label}', marker='o')
    plt.plot(val_losses, label=f'Val {label}', marker='s')

    # Labels and legend
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(f'Training and Validation {label}')
    plt.legend()

    # Save the plot
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, f"{label}_plot.png"))

    # Close the plot to free memory
    plt.close()