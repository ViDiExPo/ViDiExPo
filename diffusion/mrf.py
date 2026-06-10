"""
Mutual Re-Interaction Fusion (MRF) — ViDiExPo

Implements bidirectional cross-attention between factorized identity and
semantic embeddings BEFORE diffusion conditioning, as described in
Section 3.2.1 (Equations 18–30) and Algorithm 1 (Supplementary S3).

Key design:
- Operates on DSID-produced factorized embeddings (E_id, E_sem ∈ R^{B×512})
- Token expansion via learnable W_exp ∈ R^{512×(L·d)}
- Bidirectional cross-attention:
    Identity-to-Semantic:  E'_id  = Softmax(Q_id K_sem^T / sqrt(d)) V_sem
    Semantic-to-Identity:  E'_sem = Softmax(Q_sem K_id^T / sqrt(d)) V_id
- Concatenation + projection: E_XStream = Concat(E'_id, E'_sem) W_f
- Integration with U-Net via secondary cross-attention (Eqs. 27-30)
- MRF block replaces every cross-attention block in the trainable U-Net branch
- Applied at all resolution levels H×W, H/2×W/2, ..., H/8×W/8

Ablation results (Table 9) confirm:
- Full MRF: IDsim=0.52, Semdist=0.29, ExpAcc=82%, Poseerr=2.50
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualReInteractionFusion(nn.Module):
    """
    Mutual Re-Interaction Fusion (MRF) module.

    A single MRF block applied at one U-Net resolution level.
    Each cross-attention block in the trainable U-Net branch is replaced
    by an instance of this module (Section 3.2.2).

    Args:
        emb_dim:    DSID embedding dimension (512)
        L:          number of interaction tokens
        d:          unified latent dimension for cross-attention
        unet_ch:    number of channels in U-Net features at this level (C_q)
        num_heads:  number of attention heads (multi-head extension)
    """

    def __init__(
        self,
        emb_dim: int = 512,
        L: int = 16,
        d: int = 512,
        unet_ch: int = 1280,
        num_heads: int = 8,
    ):
        super().__init__()
        self.L = L
        self.d = d
        self.unet_ch = unet_ch
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        assert d % num_heads == 0, "d must be divisible by num_heads"

        # ---- Token expansion  (shared for identity and semantic) [Eq. 19] ----
        # W_exp ∈ R^{512 × (L*d)}
        self.W_exp = nn.Linear(emb_dim, L * d, bias=False)

        # ---- Identity pathway projections  [Eq. 21] ----
        self.W_id_Q = nn.Linear(d, d, bias=False)
        self.W_id_K = nn.Linear(d, d, bias=False)
        self.W_id_V = nn.Linear(d, d, bias=False)

        # ---- Semantic pathway projections  [Eq. 22] ----
        self.W_sem_Q = nn.Linear(d, d, bias=False)
        self.W_sem_K = nn.Linear(d, d, bias=False)
        self.W_sem_V = nn.Linear(d, d, bias=False)

        # ---- Fusion projection  W_f ∈ R^{2d×d}  [Eq. 26] ----
        self.W_f = nn.Linear(2 * d, d, bias=False)

        # ---- U-Net feature projection  W_Q^{diff} ∈ R^{Cq×d}  [Eq. 27] ----
        self.W_diff_Q = nn.Linear(unet_ch, d, bias=False)

        # ---- Cross-stream to U-Net projections  [Eq. 28] ----
        self.W_diff_K = nn.Linear(d, d, bias=False)
        self.W_diff_V = nn.Linear(d, d, bias=False)

        # ---- Output projection  W_o ∈ R^{d×Cq}  [Eq. 30] ----
        self.W_o = nn.Linear(d, unet_ch, bias=False)

        # Layer norms for stability
        self.norm_xstream = nn.LayerNorm(d)
        self.norm_out     = nn.LayerNorm(unet_ch)

        self.scale = math.sqrt(self.head_dim)

    # ------------------------------------------------------------------
    # Internal multi-head cross-attention helper
    # ------------------------------------------------------------------

    def _cross_attention(
        self,
        Q: torch.Tensor,   # [B, N_q, d]
        K: torch.Tensor,   # [B, N_k, d]
        V: torch.Tensor,   # [B, N_k, d]
    ) -> torch.Tensor:
        """Scaled dot-product attention with optional multi-head split."""
        B, N_q, _ = Q.shape
        _, N_k, _ = K.shape
        h, dh = self.num_heads, self.head_dim

        # Reshape to [B, h, N, dh]
        Q = Q.view(B, N_q, h, dh).transpose(1, 2)
        K = K.view(B, N_k, h, dh).transpose(1, 2)
        V = V.view(B, N_k, h, dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self.scale          # [B, h, N_q, N_k]
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous()         # [B, N_q, h, dh]
        return out.view(B, N_q, self.d)

    # ------------------------------------------------------------------
    # MRF forward  (Algorithm 1, Supplementary S3)
    # ------------------------------------------------------------------

    def forward(
        self,
        E_id:    torch.Tensor,    # [B, 512]  identity embedding from DSID
        E_sem:   torch.Tensor,    # [B, 512]  semantic embedding from DSID
        F_unet:  torch.Tensor,    # [B, C_q, H, W]  U-Net intermediate features
    ) -> torch.Tensor:
        """
        Args:
            E_id:   identity embedding  [B, 512]
            E_sem:  semantic embedding  [B, 512]
            F_unet: U-Net feature map   [B, C_q, H, W]

        Returns:
            F_fus_unet: fused diffusion features [B, C_q, H, W]  (Eq. 30)
        """
        B, C_q, H, W = F_unet.shape

        # ---- Token expansion  [Eq. 19] ----
        # E_id, E_sem ∈ R^{B×512}  ->  E~_id, E~_sem ∈ R^{B×L×d}
        E_tilde_id  = self.W_exp(E_id).view(B, self.L, self.d)    # [B, L, d]
        E_tilde_sem = self.W_exp(E_sem).view(B, self.L, self.d)   # [B, L, d]

        # ---- QKV projections  [Eqs. 21–22] ----
        Q_id  = self.W_id_Q(E_tilde_id)
        K_id  = self.W_id_K(E_tilde_id)
        V_id  = self.W_id_V(E_tilde_id)

        Q_sem = self.W_sem_Q(E_tilde_sem)
        K_sem = self.W_sem_K(E_tilde_sem)
        V_sem = self.W_sem_V(E_tilde_sem)

        # ---- Bidirectional cross-attention  [Eqs. 23–24] ----
        # Identity-to-Semantic:  E'_id = Softmax(Q_id K_sem^T / sqrt(d)) V_sem
        E_prime_id  = self._cross_attention(Q_id,  K_sem, V_sem)  # [B, L, d]

        # Semantic-to-Identity:  E'_sem = Softmax(Q_sem K_id^T / sqrt(d)) V_id
        E_prime_sem = self._cross_attention(Q_sem, K_id,  V_id)   # [B, L, d]

        # ---- Concatenation and projection  [Eqs. 25–26] ----
        E_XStream       = torch.cat([E_prime_id, E_prime_sem], dim=-1)  # [B, L, 2d]
        E_tilde_XStream = self.norm_xstream(self.W_f(E_XStream))        # [B, L, d]

        # ---- U-Net feature token sequence  [Eq. 27] ----
        # Reshape [B, C_q, H, W] -> [B, H*W, C_q] -> [B, H*W, d]
        F_flat   = F_unet.flatten(2).transpose(1, 2)                    # [B, HW, C_q]
        Q_diff   = self.W_diff_Q(F_flat)                                # [B, HW, d]

        # ---- U-Net cross-attention  [Eqs. 28–29] ----
        K_diff = self.W_diff_K(E_tilde_XStream)                         # [B, L, d]
        V_diff = self.W_diff_V(E_tilde_XStream)                         # [B, L, d]

        # Scale by sqrt(d) for U-Net cross-attention [Eq. 29]
        scale = math.sqrt(self.d)
        attn  = F.softmax((Q_diff @ K_diff.transpose(-2, -1)) / scale, dim=-1)
        F_att = attn @ V_diff                                            # [B, HW, d]

        # ---- Residual connection  [Eq. 30] ----
        # Project F_att back to U-Net channel space
        F_out = self.W_o(F_att)                                          # [B, HW, C_q]
        F_out = F_out.transpose(1, 2).view(B, C_q, H, W)

        F_fus_unet = self.norm_out(
            (F_unet + F_out).flatten(2).transpose(1, 2)
        ).transpose(1, 2).view(B, C_q, H, W)

        return F_fus_unet


# ---------------------------------------------------------------------------
# Multi-resolution MRF block set (one per U-Net resolution level)
# Placed at every cross-attention in the trainable branch (Section 3.2.2)
# ---------------------------------------------------------------------------

class MRFIntegration(nn.Module):
    """
    Set of MRF modules for all U-Net resolution levels.

    Standard SDXL resolution levels:
        H×W, H/2×W/2, H/4×W/4, H/8×W/8  (downsampling)
        H/8×W/8, H/4×W/4, H/2×W/2, H×W  (upsampling)

    Each cross-attention block in the trainable branch is replaced by
    an MRF module (Figure 4 bottom).

    Channel dimensions match stable-diffusion-xl-base-1.0 (v1.0.0)
    which is the pre-trained backbone used (Section 4.1).
    """

    # SDXL U-Net cross-attention channel dims (down + mid + up blocks)
    SDXL_CROSS_ATTN_CHANNELS = [
        320, 640, 1280,          # down blocks (per resolution)
        1280,                    # mid block
        1280, 640, 320,          # up blocks
    ]

    def __init__(
        self,
        emb_dim: int = 512,
        L: int = 16,
        d: int = 512,
        num_heads: int = 8,
        unet_channels: list[int] | None = None,
    ):
        super().__init__()
        if unet_channels is None:
            unet_channels = self.SDXL_CROSS_ATTN_CHANNELS

        self.mrf_blocks = nn.ModuleList([
            MutualReInteractionFusion(
                emb_dim=emb_dim,
                L=L,
                d=d,
                unet_ch=ch,
                num_heads=num_heads,
            )
            for ch in unet_channels
        ])

    def forward(
        self,
        E_id:  torch.Tensor,
        E_sem: torch.Tensor,
        unet_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Apply MRF at each U-Net resolution level.

        Args:
            E_id:          [B, 512]
            E_sem:         [B, 512]
            unet_features: list of [B, C_i, H_i, W_i] tensors,
                           one per cross-attention location

        Returns:
            list of fused feature maps, same shapes as input
        """
        assert len(unet_features) == len(self.mrf_blocks), (
            f"Expected {len(self.mrf_blocks)} feature maps, "
            f"got {len(unet_features)}"
        )
        return [
            mrf(E_id, E_sem, feat)
            for mrf, feat in zip(self.mrf_blocks, unet_features)
        ]
