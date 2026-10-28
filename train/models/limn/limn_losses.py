import torch
import torch.nn as nn
import torch.nn.functional as F
from training_helpers.bbcl import BatchBasedClassificationLoss

def latent_factor_ranking_loss(Xq_hat, Xt_hat, tau=10):
    """
    Args:
        Xq_hat: tensor of shape [B, U, D] - query latent factor tokens
        Xt_hat: tensor of shape [B, U, D] - target latent factor tokens
        tau: temperature scalar
    
    Returns:
        loss: scalar tensor
    """
    B, U, D = Xq_hat.shape
    
    # Normalize tokens along the feature dim D to get cosine similarity by dot product
    Xq_hat_norm = F.normalize(Xq_hat, p=2, dim=2)  # [B, U, D]
    Xt_hat_norm = F.normalize(Xt_hat, p=2, dim=2)  # [B, U, D]

    # Compute pairwise cosine similarity sums over U latent factors:
    # We want s(Xq_hat[i,u], Xt_hat[j,u]) summed over u for each pair (i,j)

    # Step 1: For each pair (i,j), compute similarity per factor u by dot product on D:
    # Expand dims for broadcasting:
    # Xq_hat_norm: [B, U, D] -> [B, 1, U, D]
    # Xt_hat_norm: [B, U, D] -> [1, B, U, D]
    Xq_exp = Xq_hat_norm.unsqueeze(1)  # [B, 1, U, D]
    Xt_exp = Xt_hat_norm.unsqueeze(0)  # [1, B, U, D]

    # Dot product along D gives similarity per factor:
    sim_per_factor = (Xq_exp * Xt_exp).sum(dim=3)  # [B, B, U]

    # Sum similarities over U factors:
    sim_sum = sim_per_factor.sum(dim=2)  # [B, B]

    # Divide by temperature tau
    sim_sum = sim_sum / tau  # [B, B]

    # For each sample i in batch, compute numerator and denominator of the softmax
    # Numerator: sim_sum[i, i]
    numerator = torch.exp(torch.diagonal(sim_sum))  # [B]

    # Denominator: sum over j of exp(sim_sum[i, j])
    denominator = torch.exp(sim_sum).sum(dim=1)  # [B]

    # Compute loss per sample
    loss_per_sample = -torch.log(numerator / denominator)  # [B]

    # Return mean loss
    return loss_per_sample.mean()


def l1_condition_mask_loss(condition_masks_q, condition_masks_t):
    # condition_masks_q, condition_masks_t: tensors of shape [U, D]
    
    U = condition_masks_q.size(0)
    
    # Compute L1 norm per mask, then mean over U
    loss_q = torch.sum(torch.abs(condition_masks_q), dim=1).mean()  # scalar
    loss_t = torch.sum(torch.abs(condition_masks_t), dim=1).mean()  # scalar
    
    loss = loss_q + loss_t  # sum as per formula
    
    return loss

class LIMN_Loss(nn.Module):
    def __init__(self, loss_tau):
        super().__init__()
        self.bbcl_loss = BatchBasedClassificationLoss(tau=loss_tau)

    def forward(self, a_hat_q, a_hat_t, condition_masks_q, condition_masks_t, x_hat_q, x_hat_t, query_ids, target_ids):
        # Compute the three components of the loss
        bbcl = self.bbcl_loss(a_hat_q, a_hat_t, query_ids, target_ids)
        l1 = l1_condition_mask_loss(condition_masks_q, condition_masks_t)
        lfrl = latent_factor_ranking_loss(x_hat_q, x_hat_t)

        total_loss = bbcl + l1 + lfrl

        return total_loss