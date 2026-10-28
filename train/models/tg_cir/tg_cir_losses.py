import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalRegularizationLoss(nn.Module):
    def __init__(self, K=12):
        super().__init__()

    def compute_ortho(self, E):
        batch_size, K, _ = E.shape
        # Create identity matrix and expand to batch size
        I = torch.eye(K, device=E.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # L2 Normalize along the feature dimension D to avoid exploding gradients
        E_norm = F.normalize(E, p=2, dim=-1, eps=1e-7)

        # E @ E^T -> shape: (batch_size, K, K)
        E_E_T = torch.bmm(E_norm, E_norm.transpose(1, 2))
        
        # Calculate the norm across the matrix dimensions (dim 1 and 2), then square it
        diff = E_E_T - I
        norm_sq = (diff ** 2).sum(dim=(1, 2))
        
        # Average over the batch
        return norm_sq.mean()

    def forward(self, E_r, E_m, E_t):
        """
        Computes Eq 5: L_ortho = ||E_r E_r^T - I||_F^2 + ||E_m E_m^T - I||_F^2 + ||E_t E_t^T - I||_F^2
        
        Args:
            E_r, E_m, E_t: Extracted attribute tensors of shape (batch_size, K, D)
        Returns:
            Scalar loss tensor.
        """
        loss = self.compute_ortho(E_r) + self.compute_ortho(E_m) + self.compute_ortho(E_t)
        return loss


class TeacherMaskRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m_k_tilde, m_r_tilde):
        """
        Computes Eq 9: L_mask_tea = ||m_r_tilde, 1 - m_k_tilde||^2
        Note: The notation ||x, y||^2 in this context means MSE loss.
        """
        target = 1.0 - m_k_tilde
        loss = F.mse_loss(m_r_tilde, target)
        return loss

class CompositionKnowledgeDistillation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m_k_tilde, m_r_tilde, m_k_hat, m_r_hat):
        """
        Computes Eq 10: L_ckd = ||m_k_tilde - m_k_hat||^2 + ||m_r_tilde - m_r_hat||^2
        """
        loss_k = F.mse_loss(m_k_hat, m_k_tilde)
        loss_r = F.mse_loss(m_r_hat, m_r_tilde)
        return loss_k + loss_r
    

class StudentBBCL(nn.Module):
    def __init__(self, tau = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, ref_features, tar_features):
        """
        Early Fusion Classification Loss.
        
        ref_features: (B, K, D) tensor of composed query embeddings (E_c_hat)
        tar_features: (B, K, D) tensor of target embeddings (E_t)
        """
        batch_size = ref_features.size(0)
        device = ref_features.device

        # Early Fusion: Mean pool along the K dimension
        # (B, K, D) -> (B, D)
        psi_c_hat = ref_features.mean(dim=1)
        psi_t = tar_features.mean(dim=1)

        # L2 Normalize to ensure dot product computes cosine similarity (s)
        ref_norm = F.normalize(psi_c_hat, p=2, dim=-1, eps=1e-7)
        tar_norm = F.normalize(psi_t, p=2, dim=-1, eps=1e-7)

        # Compute similarity matrix
        sim_matrix = ref_norm @ tar_norm.T  # shape: (B, B)

        # Scale by temperature tau
        logits = sim_matrix / self.tau

        # Labels are the diagonal (0...B-1)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss handles the log-softmax and batch mean
        loss = F.cross_entropy(logits, labels)

        return loss

class TeacherBBCL(nn.Module):
    def __init__(self, tau = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, ref_features, tar_features):
        """
        Late Fusion Classification Loss.
        
        ref_features: (B, K, D) tensor of composed query embeddings (E_c_tilde)
        tar_features: (B, K, D) tensor of target embeddings (E_t)
        """
        batch_size = ref_features.size(0)
        device = ref_features.device

        # L2 Normalize along the feature dimension D
        ref_norm = F.normalize(ref_features, p=2, dim=-1, eps=1e-7)
        tar_norm = F.normalize(tar_features, p=2, dim=-1, eps=1e-7)

        # Late Fusion: Compute cosine sim per attribute k, then sum them up
        # 'ikd' (ref) and 'jkd' (tar) -> 'ij' (B x B matrix)
        # einsum cleanly handles the dot product over 'd' and sum over 'k'
        sim_matrix = torch.einsum('ikd, jkd -> ij', ref_norm, tar_norm)

        # Scale by temperature tau
        logits = sim_matrix / self.tau

        # Labels are the diagonal (0...B-1)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss handles the log-softmax and batch mean
        loss = F.cross_entropy(logits, labels)

        return loss

class TargetSimilarityGuidedLoss(nn.Module):
    def __init__(self, tau = 0.1):
        """
        Computes the KL Divergence between the Target Visual Similarity Distribution 
        and the Query Matching Degree Distribution (Eq 12, 13, 14).
        """
        super().__init__()
        self.tau = tau

    def forward(self, E_c_hat, E_t):
        """
        Args:
            E_c_hat: Student composed features, Shape: (B, K, D)
            E_t: Target features, Shape: (B, K, D)
        """
        # Target Visual Similarity Dist (p^t) - Eq 12
        # Uses Late Fusion (Attribute-wise sum)
        E_t_norm = F.normalize(E_t, p=2, dim=-1, eps=1e-7)
        
        # 'ikd' (Target i) and 'jkd' (Target j) -> 'ij'
        sim_t = torch.einsum('ikd, jkd -> ij', E_t_norm, E_t_norm)
        
        # Regular probabilities for the target (P)
        p_t = F.softmax(sim_t / self.tau, dim=1)
        p_t = p_t.clamp(min=1e-8)
        
        # Query Matching Dist (p^c) - Eq 13
        # Uses Early Fusion (Mean-pooled)
        psi_c_hat = E_c_hat.mean(dim=1)  # (B, D)
        psi_t = E_t.mean(dim=1)          # (B, D)
        
        psi_c_norm = F.normalize(psi_c_hat, p=2, dim=-1, eps=1e-7)
        psi_t_norm = F.normalize(psi_t, p=2, dim=-1, eps=1e-7)
        
        sim_c = psi_c_norm @ psi_t_norm.T
        
        # LOG-probabilities for the input (Q) to ensure numerical stability
        log_p_c = F.log_softmax(sim_c / self.tau, dim=1)
        
        # 3. KL Divergence (L_kl) - Eq 14
        # PyTorch kl_div computes KL(target || input) which matches our D_KL(p^t || p^c)
        # reduction='batchmean' divides the total loss by the batch size B, matching Eq 14 exactly.
        loss_kl = F.kl_div(log_p_c, p_t, reduction='batchmean')
        
        return loss_kl

class TG_CIR_Loss(nn.Module):
    def __init__(self, tau = 0.5, l = 1.0, n = 500.0, u = 0.01, v = 500, k = 2):
        """
        Hyper-parameters correspond to Eq 15:
        l: (Teacher Rank Loss weight)
        n: (Teacher Mask Reg weight)
        u: (Orthogonal Reg weight)
        v: (Knowledge Distillation weight)
        k: (KL Divergence weight)
        """
        super().__init__()
        
        # Save trade-off weights
        self.lambda_w = l
        self.eta_w = n
        self.mu_w = u
        self.nu_w = v
        self.kappa_w = k

        # Initialize loss modules (Make sure to bind them to self!)
        self.student_bbcl = StudentBBCL(1)
        self.teacher_bbcl = TeacherBBCL(1)
        self.mask_reg_loss = TeacherMaskRegularization()
        self.ortho_loss = OrthogonalRegularizationLoss()
        self.ckd_loss = CompositionKnowledgeDistillation()
        self.kl_loss = TargetSimilarityGuidedLoss(1)

    def forward(self, E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde):
        """
        Computes the final optimization objective (Equation 15).
        
        Args:
            E_r, E_m, E_t: Extracted attributes (B, K, D)
            E_c_tilde: Teacher composed features (B, K, D)
            E_c_hat: Student composed features (B, K, D)
            m_k_tilde, m_r_tilde: Teacher masks (B, K, 1)
            m_k_hat, m_r_hat: Student masks (B, K, 1)
        """
        
        # 1. Student Rank Loss
        l_stu_rank = self.student_bbcl(E_c_hat, E_t)
        
        # 2. Teacher Rank Loss
        l_tea_rank = self.teacher_bbcl(E_c_tilde, E_t)
        
        # 3. Teacher Mask Regularization
        l_tea_mask = self.mask_reg_loss(m_k_tilde, m_r_tilde)
        
        # 4. Orthogonal Regularization
        l_ortho = self.ortho_loss(E_r, E_m, E_t)
        
        # 5. Composition Knowledge Distillation
        l_ckd = self.ckd_loss(m_k_tilde, m_r_tilde, m_k_hat, m_r_hat)
        
        # 6. KL Divergence
        l_kl = self.kl_loss(E_c_hat, E_t)
        
        # Equation 15: Total Weighted Loss
        total_loss = (
            l_stu_rank + 
            (self.lambda_w * l_tea_rank) + 
            (self.eta_w * l_tea_mask) + 
            (self.mu_w * l_ortho) + 
            (self.nu_w * l_ckd) + 
            (self.kappa_w * l_kl)
        )

        return total_loss



