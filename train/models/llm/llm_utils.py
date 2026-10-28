import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import csv
import ast
import torch.nn.functional as F

def LLM_fusion(self, song_name, modification_text):
        target_desc = self.model.get_target_description(song_name, modification_text)

        g, l = self.clap_model.get_text_tokens(target_desc)
        g = g.detach().cpu()
        g = F.normalize(g.float().squeeze(), dim=-1)

        return g.numpy()

model_name = "Qwen/Qwen3-8B"
self.model = LLM_Based(model_name)