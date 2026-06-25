import torch
import torch.nn as nn


#-------------------------First Part-----------------------------#
class GlobalAttributeExtractor(nn.Module):
    def __init__(self, embed_dim: int = 512, num_attributes: int = 4):
        """
        Extracts global attribute features using learnable condition masks.
        
        Args:
            embed_dim (int): The dimensionality of the CLIP global feature (D). 
                             Standard CLIP ViT-B/32 uses 512.
            num_attributes (int): The number of global attributes (P).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attributes = num_attributes
        
        # P learnable condition masks C_i, each of dimension D
        # Shape: (P, D)
        self.condition_masks = nn.Parameter(torch.Tensor(num_attributes, embed_dim))
        
        # Initialize the masks with a normal distribution
        nn.init.normal_(self.condition_masks)

    def forward(self, f):
        """
        Args:
            f (torch.Tensor): Global features from CLIP (f_r, f_t, or f_m). 
                              Shape: (batch_size, embed_dim)
                              
        Returns:
            torch.Tensor: Global attribute features (u_r^i). 
                          Shape: (batch_size, num_attributes, embed_dim)
        """
        # Expand f to shape (batch_size, 1, embed_dim) to allow broadcasting across P attributes
        f_expanded = f.unsqueeze(1)
        
        # Expand condition_masks to (1, num_attributes, embed_dim) for batch broadcasting
        masks_expanded = self.condition_masks.unsqueeze(0)
        
        # Element-wise multiplication: f_r \odot C_i
        # Output shape will be (batch_size, num_attributes, embed_dim)
        u = f_expanded * masks_expanded
        
        return u

class LocalAttributeExtractor(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 512, num_attributes: int = 4):
        """
        Extracts local attribute features using 1x1 convolutions for aggregation.
        
        Args:
            in_dim (int): The dimensionality of the CLIP local token features (D').
            embed_dim (int): The target dimensionality (D) to match global features.
            num_attributes (int): The number of local attributes (Q).
        """
        super().__init__()
        self.num_attributes = num_attributes
        
        # Project local features to match global feature dimension
        self.fc_i = nn.Linear(in_dim, embed_dim)
        
        # 1x1 Conv to get weights. 
        # Input channels = embed_dim, Output channels = Q (doing all Q aggregations at once)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=num_attributes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Local features from CLIP. Shape: (batch_size, L, in_dim)
                              where L is the number of tokens.
        Returns:
            torch.Tensor: Local attribute features (v_r). Shape: (batch_size, Q, embed_dim)
        """
        # Project features: (batch_size, L, embed_dim)
        x_proj = self.fc_i(x)
        
        # Get aggregation weights
        # Conv1d expects shape (batch_size, channels, sequence_length)
        x_transposed = x_proj.transpose(1, 2)  # (batch_size, embed_dim, L)
        
        # s shape: (batch_size, Q, L)
        s = self.sigmoid(self.conv(x_transposed)) 
        
        # Weighted pooling via matrix product
        # (batch_size, Q, L) @ (batch_size, L, embed_dim) -> (batch_size, Q, embed_dim)
        v = torch.bmm(s, x_proj)
        
        return v

class TG_CIR_AttributeExtractor(nn.Module):
    def __init__(self, global_dim=512, local_in_dim=768, P=4, Q=4):
        """
        Combines Global and Local Attribute Extractors.
        """
        super().__init__()
        Q_local_attributes = 8
        P_global_attributes = 4
        self.global_extractor = GlobalAttributeExtractor(embed_dim=global_dim, num_attributes=P)
        self.local_extractor = LocalAttributeExtractor(in_dim=local_in_dim, embed_dim=global_dim, num_attributes=Q)
        
    def forward(self, global_feat, local_feat):
        """
        Args:
            global_feat: (batch_size, D)
            local_feat: (batch_size, L, D')
        Returns:
            E: Final attribute features. Shape: (batch_size, K, D) where K = P + Q
        """
        # u shape: (batch_size, P, D)
        u = self.global_extractor(global_feat)
        
        # v shape: (batch_size, Q, D)
        v = self.local_extractor(local_feat)
        
        # Cascade/concatenate along the attribute dimension
        # E shape: (batch_size, P+Q, D)
        E = torch.cat([u, v], dim=1)
        
        return E

#--------------------------------Second Part-----------------------------#
class StudentCompositionBranch(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        """
        Target-free multimodal query composition branch (Student).
        
        Args:
            embed_dim (int): Dimensionality of the attribute features (D).
            hidden_dim (int): Hidden layer size for the MLP.
        """
        super().__init__()
        
        # MLP_s takes concatenated [E_r, E_m] of size 2 * embed_dim
        # and outputs a single probability scalar per attribute.
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, E_r: torch.Tensor, E_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            E_r: Reference image attributes, Shape: (batch_size, K, D)
            E_m: Modification text attributes, Shape: (batch_size, K, D)
            
        Returns:
            E_c_hat: Composed features, Shape: (batch_size, K, D)
            m_k_hat: Keep mask, Shape: (batch_size, K, 1)
            m_r_hat: Replace mask, Shape: (batch_size, K, 1)
        """
        # 1. Concatenate along the feature dimension
        # Shape: (batch_size, K, 2 * D)
        combined_features = torch.cat([E_r, E_m], dim=-1)
        
        # 2. Learn keep mask m_k_hat (Eq 6)
        # Passing through Linear layer applies to the last dimension (2*D -> 1)
        # Output shape: (batch_size, K, 1)
        m_k_hat = torch.sigmoid(self.mlp_s(combined_features))
        
        # 3. Derive replace mask m_r_hat (Eq 6)
        # Shape: (batch_size, K, 1)
        m_r_hat = 1.0 - m_k_hat
        
        # 4. Compose features (Eq 7)
        # Broadcasting the (batch_size, K, 1) masks over the (batch_size, K, D) features
        E_c_hat = (m_k_hat * E_r) + (m_r_hat * E_m)
        
        return E_c_hat, m_k_hat, m_r_hat

class TeacherCompositionBranch(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        """
        Target-based multimodal query composition branch (Teacher).
        """
        super().__init__()
        
        # MLP_t1 for the keep mask (takes [E_t, E_r])
        self.mlp_t1 = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # MLP_t2 for the replace mask (takes [E_t, E_m])
        self.mlp_t2 = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, E_r, E_m, E_t):
        """
        Args:
            E_r: Reference image attributes
            E_m: Modification text attributes
            E_t: Target image attributes
        """
        # 1. Learn keep mask m_k_tilde (Eq 8)
        # Concat target and reference: [E_t, E_r] -> shape (batch_size, K, 2*D)
        t_r_combined = torch.cat([E_t, E_r], dim=-1)
        m_k_tilde = torch.sigmoid(self.mlp_t1(t_r_combined))
        
        # 2. Learn replace mask m_r_tilde (Eq 8)
        # Concat target and text: [E_t, E_m] -> shape (batch_size, K, 2*D)
        t_m_combined = torch.cat([E_t, E_m], dim=-1)
        m_r_tilde = torch.sigmoid(self.mlp_t2(t_m_combined))
        
        # 3. Compose features (Eq 8)
        E_c_tilde = (m_k_tilde * E_r) + (m_r_tilde * E_m)
        
        return E_c_tilde, m_k_tilde, m_r_tilde
    
#------------------------Final Main Model--------------------------------#
class TG_CIR(nn.Module):
    def __init__(self, global_dim=512, local_in_dim=768, P=4, Q=8, hidden_dim=256):
        super().__init__()
        self.attribute_extractor = TG_CIR_AttributeExtractor(global_dim, local_in_dim, P, Q)
        self.student_branch = StudentCompositionBranch(embed_dim=global_dim, hidden_dim=hidden_dim)
        self.teacher_branch = TeacherCompositionBranch(embed_dim=global_dim, hidden_dim=hidden_dim)

    def forward(self, global_ref, global_text, global_tar, local_ref, local_text, local_tar):
        """
        global_ref: [B, D] - reference song features
        global_text: [B, D] - modification text features
        global_tar: [B, D] - target song features
        local_ref: [B, L, 768] - local audio tokens for reference song
        local_text: [B, L', 768] - local text tokens for modification text
        local_tar: [B, L'', 768] - local audio tokens for target song
        """
        # Extract attributes for reference and modification
        if global_tar is not None and local_tar is not None:
            E_t = self.attribute_extractor(global_tar, local_tar)  # [B, K, D]

        #if ref is missing, we just want to encode the target for evaluation
        # so we skip the rest of the forward pass and return E_t
        if global_ref is None:
            return E_t
    
        # Extract attributes for reference and modification
        E_r = self.attribute_extractor(global_ref, local_ref)  # [B, K, D]
        E_m = self.attribute_extractor(global_text, local_text)  # [B, K, D]
    
        # Student branch composition 
        E_c_hat, m_k_hat, m_r_hat = self.student_branch(E_r, E_m)

        #if target is missing, then we're dealing with inference, so we skip the teacher branch and return None for those tensors
        if global_tar is None and local_tar is None:
            return E_c_hat
        
        # Teacher branch composition 
        E_c_tilde, m_k_tilde, m_r_tilde = self.teacher_branch(E_r, E_m, E_t)
            
        # Return ALL tensors needed for the TG_CIR_Loss
        return E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde
            