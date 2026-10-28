import torch
import torch.nn.functional as F
from torch import nn


class Baseline(nn.Module):
    def __init__(self, alpha, beta):
        super(Baseline, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, ref_global, mod_text_global):
        sum_embedding = self.alpha * ref_global + self.beta * mod_text_global
        
        return F.normalize(sum_embedding, dim=-1)