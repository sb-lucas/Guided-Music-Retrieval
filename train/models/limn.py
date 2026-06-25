import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def safe_normalize(tensor, dim):
    return F.normalize(tensor, dim=dim, eps=1e-6)

class WeightLearning(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        #this is a simple MLP. i will return the sigmoid outputs

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, fr, fm):        
        """
        fr: [B, D] - reference song features
        fm: [B, D] - modification text features
        """
        # Concatenate the two vectors
        combined = torch.cat([fr, fm], dim=1)

        # Pass through the MLP
        weights = self.mlp(combined)  # [B, D]

        return weights

class AudioTokenProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim//2, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, audio_tokens):
        """
        audio_tokens: [B, L, 768] - local audio tokens
        returns: [B, L, 512]
        """
        return F.normalize(self.proj(audio_tokens), dim=-1)  

class TextTokenProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        #dropout should be used 
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),  
            nn.Linear(in_dim//2, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),  
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, text_tokens):
        """
        text_tokens: [B, L', 768] - local text tokens
        returns: [B, L', 512]
        """
        return F.normalize(self.proj(text_tokens), dim=-1)  


class LIMN(nn.Module):
    def __init__(self, embed_dim=512, num_factors=8, num_heads=8, num_layers=4):
        '''
        Batch size: B
        Number of factors: U
        Embedding dimension: D
        '''

        super().__init__()

        self.U = num_factors
        self.D = embed_dim

        # learn weights to get fq, the fused representation of the multimodal query, the query features.
        self.get_w = WeightLearning(embed_dim)

        #U learnable condition masks 
        self.condition_masks_q = nn.Parameter(safe_normalize(torch.randn(self.U, self.D), dim=1))
        self.condition_masks_t = nn.Parameter(safe_normalize(torch.randn(self.U, self.D), dim=1))

        #factor identifier token
        self.mu_q = nn.Parameter(safe_normalize(torch.randn(self.U, self.D), dim=1))  # [U, D]
        self.mu_t = nn.Parameter(safe_normalize(torch.randn(self.U, self.D), dim=1))  # [U, D]

        #audio local tokens projector
        self.audio_token_projector = AudioTokenProjector(in_dim=768, out_dim=512)

        #text local tokens projector
        self.text_token_projector = TextTokenProjector(in_dim=768, out_dim=512)

        #identifier tokens for the matching tokens
        self.m_q = nn.Parameter(safe_normalize((torch.randn(self.D)), dim=0))  # m_q
        self.m_t = nn.Parameter(safe_normalize((torch.randn(self.D)), dim=0))  # m_t

        # Transformer block
        self.transformer_block = TransformerBlock(hidden_size=self.D, num_heads=num_heads)

    def forward(self, fr, fm, ft, er, em, et, eq_len, et_len):
        """
        fr: [B, D] - reference song features
        fm: [B, D] - modification text features
        ft: [B, D] - target song features
        er: [B, L, 768] - local audio tokens for reference song
        em: [B, L', 768] - local text tokens for modification text
        et: [B, L'', 768] - local audio tokens for target song
        eq_len: [B] - lengths of the local tokens for the query
        et_len: [B] - lengths of the local tokens for the target song
        """

        #building the fused query features
        w = self.get_w(fr, fm)  # [B, D] 
        fq = (w * fr) + ((1-w) * fm)
        #fq = fr  # or fm


        #print("[DEBUG] w stats:", w.min().item(), w.max().item(), w.mean().item(), w.std().item())


        #print("[DEBUG] fq mean/std:", fq.mean().item(), fq.std().item())


        #latent factor tokens, uq and ut
        uq = fq.unsqueeze(1) * self.condition_masks_q.unsqueeze(0)  # -> [B, U, D]
        ut = ft.unsqueeze(1) * self.condition_masks_t.unsqueeze(0)  # -> [B, U, D]

        #print("[DEBUG] condition_masks_q std:", self.condition_masks_q.std().item())
        #print("[DEBUG] condition_masks_t std:", self.condition_masks_t.std().item())

        #final query latent factor tokens
        Xq = uq + self.mu_q.unsqueeze(0)  # [B, U, D]
        Xt = ut + self.mu_t.unsqueeze(0)  # [B, U, D]

        #print("[DEBUG] mu_q std:", self.mu_q.std().item())
        #print("[DEBUG] mu_t std:", self.mu_t.std().item())

        #local audio tokens, shape = [B, L, 768] -> [B, L, 512]
        er = self.audio_token_projector(er)
        et = self.audio_token_projector(et)
        
        #local text token, shape = [B, L', 768] -> [B, L', 512]
        em = self.text_token_projector(em)

        #print("[DEBUG] audio_token_projector (er) std:", er.std().item())
        #print("[DEBUG] audio_token_projector (et) std:", et.std().item())
        #print("[DEBUG] text_token_projector (em) std:", em.std().item())

        #matching tokens, shape = [B, D]
        aq = fq + self.m_q
        at = ft + self.m_t

        #print("[DEBUG] m_q std:", self.m_q.std().item())
        #print("[DEBUG] m_t std:", self.m_t.std().item())
        #print("[DEBUG] aq std:", aq.std().item())
        #print("[DEBUG] at std:", at.std().item())

        #aggregate tokens in transformer
        x_hat_q, a_hat_q, self.attn_weights_q = self.transformer_block(torch.cat([er, em], dim=1), Xq, aq, eq_len)
        x_hat_t, a_hat_t, attn_weights_t = self.transformer_block(et, Xt, at, et_len)

        return x_hat_q, a_hat_q, x_hat_t, a_hat_t


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, dropout=0.25):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

    @staticmethod
    def build_attention_masks(local_lengths, num_factors):
        """
        local_lengths: [B] tensor with number of local tokens per sample
        num_factors: number of latent factor tokens (U)
        Returns:
            attention_masks: [B, T_max, T_max] float mask for Transformer
        """
        B = local_lengths.size(0)
        L_max = local_lengths.max().item()
        T_max = L_max + num_factors + 1  # total tokens: local + factors + match

        #create attention masks
        masks = torch.zeros((B, T_max, T_max), dtype=torch.bool)

        for b in range(B):
            L = local_lengths[b].item()
            match_idx = L + num_factors  # index of match token

            # Match token should not attend to local tokens
            masks[b, match_idx, :L] = True
            
            # Local tokens should not attend to match token
            masks[b, :L, match_idx] = True

            # (rest is default: allow all attention)

        # Create padding masks. this let's pytorch ignore the padding tokens in the attention
        padding_masks = torch.zeros((B, T_max), dtype=torch.bool)  # False = attend, True = ignore

        for b in range(B):
            valid_len = local_lengths[b].item() + num_factors + 1  # L + U + match
            padding_masks[b, valid_len:] = True  # mark the rest as padding

        return masks, padding_masks

    def forward(self, tokens, attn_mask, pad_mask):
        # tokens: [B, T, D]
        # attn_mask: [B, T, T] bool mask with True where attention disallowed

        B, T, D = tokens.size()
        outputs = []
        #since you may have different token lengths in the batch, we will iterate over the batch
        #this happens because of different modification text lengths, generating different local text token lengths
        for b in range(B):
            sample_tokens = tokens[b:b+1]  # [1, T, D]
            mask_sample = attn_mask[b]  # [T, T]
            pad_mask_sample = pad_mask[b].unsqueeze(0)  # [1, T]
            
            out_sample, weights = self.attn(
                sample_tokens, sample_tokens, sample_tokens,
                attn_mask=mask_sample,
                key_padding_mask=pad_mask_sample,
                need_weights=True
            )

            outputs.append(out_sample)
            #stack weights
            if b == 0:
                weight_outputs = weights.unsqueeze(0)
            else:
                weight_outputs = torch.cat([weight_outputs, weights.unsqueeze(0)], dim=0)

        weight_outputs = weight_outputs.mean(dim=1)
        if weight_outputs.requires_grad:
            weight_outputs.retain_grad()  # <-- Add this line

        return torch.cat(outputs, dim=0), weight_outputs

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size * 3),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)
                
    def forward(self, tokens, attn_mask, pad_mask):
        # Attention and norm
        x, attn_weights = self.attn1(tokens, attn_mask, pad_mask)
        x = x + tokens
        x = self.norm1(x)

        # Apply MLP and layer normalization
        x_mlp = self.mlp(x)
        x = x + x_mlp
        x = self.norm_mlp(x)
        
        return x, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, num_layers = 2, num_heads=8):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(num_layers)
        ])
                
    def forward(self, local_tokens, feature_tokens, match_token, local_lengths):
        #pad local_tokens to the maximum length in the batch [B, L_max, D]
        local_tokens = nn.utils.rnn.pad_sequence(local_tokens, batch_first=True, padding_value=0.0)    

        #get number of feature tokens
        num_factors = feature_tokens.size(1) # U

        #get the masks for the attention
        attn_mask, pad_mask = AttentionBlock.build_attention_masks(local_lengths, num_factors)
        attn_mask = attn_mask.to(local_tokens.device)  # move to the same device as local_tokens
        pad_mask = pad_mask.to(local_tokens.device)

        #concat local tokens, feature tokens and match token
        match_token = match_token.unsqueeze(1)
        tokens = torch.cat([local_tokens, feature_tokens, match_token], dim=1) # [B, T, D]
        
        # Pass sequentially through all transformer layers
        for layer in self.layers:
            x, attn_weights = layer(tokens, attn_mask, pad_mask)

        # Extract the output for the match token and the feature tokens x
        x_hat = []
        a_hat = []

        for b in range(x.size(0)):
            L = local_lengths[b].item()
            x_hat.append(x[b, L:L+num_factors, :])     # shape [U, D]
            a_hat.append(x[b, L+num_factors, :])       # shape [D]

        x_hat = torch.stack(x_hat, dim=0)  # [B, U, D]
        a_hat = torch.stack(a_hat, dim=0)  # [B, D]

        return x_hat, a_hat, attn_weights  # [B, U, D], [B, D]
    

