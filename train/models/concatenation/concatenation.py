import torch
import torch.nn as nn
import torch.nn.functional as F

class Concatenation(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Concatenation, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, ref_song_embs, text_embs):
        x = torch.cat((ref_song_embs, text_embs), dim=-1)
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)

        return x