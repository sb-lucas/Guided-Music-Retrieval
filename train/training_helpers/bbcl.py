import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchBasedClassificationLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.tau = tau

    def forward(self, ref_features, tar_features, query_ids, target_ids):
        """
        ref_features: (B, D) tensor of fused (ref + text) embeddings
        tar_features: (B, D) tensor of target embeddings
        query_ids: (B,) tensor of query ids
        target_ids: (B,) tensor of target ids
        """

        batch_size = ref_features.size(0)
        device = ref_features.device

        # Compute similarity matrix (dot products between ref and tar)
        pred = ref_features @ tar_features.T  # shape: (B, B)

        #scale by temperature tau
        pred = pred / self.tau

        same_query  = query_ids[:, None]  == query_ids[None, :] # [B, B] True where triplets i and j have the same (reference, modification)
        same_target = target_ids[:, None] == target_ids[None, :] # [B, B] True where they have the same target track
        
        same_query_or_target = (same_query | same_target) #triplet has the same query or has the same target track
        false_neg = same_query_or_target & ~torch.eye(batch_size, dtype=torch.bool, device=pred.device) #remove self-comparisons

        pred = pred.masked_fill(false_neg, float('-inf'))

        # Labels are always 0...B-1 (we want pred[i][i] to be the highest)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss expects logits and correct class indices
        loss = F.cross_entropy(pred, labels)

        return loss