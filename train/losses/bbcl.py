import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchBasedClassificationLoss(nn.Module):
    def __init__(self, tau=1):
        super().__init__()

        self.tau = tau

    def forward(self, ref_features, tar_features):
        """
        ref_features: (B, D) tensor of fused (ref + text) embeddings
        tar_features: (B, D) tensor of target embeddings
        """

        batch_size = ref_features.size(0)
        device = ref_features.device

        # Compute similarity matrix (dot products between ref and tar)
        pred = ref_features @ tar_features.T  # shape: (B, B)

        #scale by temperature tau
        pred = pred / self.tau

        # Labels are always 0...B-1 (we want pred[i][i] to be the highest)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss expects logits and correct class indices
        loss = F.cross_entropy(pred, labels)

        return loss
