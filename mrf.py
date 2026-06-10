"""
Mutual Re-Interaction Fusion (MRF) — ViDiExPo
Section 3.2 (MRF subsections: Identity-Semantic Interaction Extraction +
Diffusion Fusion + Fine-tuning)

Design (paper, revised):
─────────────────────────────────────────────────────────────────────────────
INPUTS from DSID
  E_id, E_sem  ∈ R^{B×512}          (Eq. 1 / Section 3.2)
  F_id, F_sem  ∈ R^{B×2048×8×8}     (last encoder feature maps, Eq. feature_maps)

IDENTITY-SEMANTIC INTERACTION EXTRACTION (Section 3.2.1)
  1. Linear query projections        W_Q^id, W_Q^sem ∈ R^{512×512}
       Q_id  = E_id  W_Q^id              [B, 512]
       Q_sem = E_sem W_Q^sem             [B, 512]
  2. Unsqueeze to token form
       Q_id, Q_sem  ∈ R^{B×1×512}
  3. Reshape feature maps to token sequences (each spatial loc = one token)
       F_id, F_sem  ∈ R^{B×64×2048}     (Eq. flatten)
  4. Key/value projections           W_K, W_V ∈ R^{2048×512}
       K_id, V_id, K_sem, V_sem  ∈ R^{B×64×512}
  5. Bidirectional single-query cross-attention
       I_{id←sem} = Softmax(Q_id  K_sem^T / √d) V_sem  ∈ R^{B×1×512}
       I_{sem←id} = Softmax(Q_sem K_id^T  / √d) V_id   ∈ R^{B×1×512}
  6. Concat + project
       I       = Concat(I_{id←sem}, I_{sem←id})  ∈ R^{B×1×1024}
       I_cross = I W_I,   W_I ∈ R^{1024×512}     ∈ R^{B×1×512}

DIFFUSION FUSION (Section 3.2.2) — semantically hierarchical conditioning
  Three conditioning tokens (all R^{B×1×512}):
       E_sem   → Up Block 1  (d=1280, global semantics)
       E_id    → Up Block 3  (d=640,  structural identity)
       I_cross → Up Block 4  (d=320,  extracted interaction)

  At each stage s with U-Net feature F_unet ∈ R^{B×C_s×H_s×W_s}:
       Q_s  = Reshape(F_unet)         ∈ R^{B×H_s*W_s×C_s}  → projected to d
       K_s  = C^(s) W_K^(s)           ∈ R^{B×1×d}
       V_s  = C^(s) W_V^(s)           ∈ R^{B×1×d}
       F_fus = F_unet + Proj(Softmax(Q_s K_s^T / √d) V_s)   (Eq. fusion)

─────────────────────────────────────────────────────────────────────────────
KEY DIFFERENCES from the old implementation
  Old: token expansion W_exp (B×512 → B×L×d), full L-token QKV paths
  New: single query token (Unsqueeze), feature-map KV paths (F_id/F_sem)

  Old: shared W_exp for both id and sem, computed K/V from expanded tokens
  New: separate W_Q^id / W_Q^sem; K/V from reshaped backbone feature maps

  Old: concat along last dim, project with W_f ∈ R^{2d×d}
  New: identical concat-project step → I_cross ∈ R^{B×1×512}  (kept)

  Old: MRF applied uniformly at EVERY cross-attention level (7 levels)
  New: three dedicated stages at Up Block 1 / 3 / 4  (C=1280/640/320)
       each with its own semantically matched conditioning token
─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MRF Interaction Extractor
# Implements Section 3.2.1: produces I_cross ∈ R^{B×1×512}
# ---------------------------------------------------------------------------

class MRFInteractionExtractor(nn.Module):
    """
    Identity-Semantic Interaction Extraction (Section 3.2.1).

    Takes:
        E_id,  E_sem  ∈ R^{B×512}          — DSID factorized embeddings
        F_id,  F_sem  ∈ R^{B×2048×8×8}     — last encoder feature maps

    Produces:
        Q_id,  Q_sem  ∈ R^{B×1×512}        — query tokens (reused as
                                              conditioning embeddings for
                                              E_id / E_sem downstream)
        I_cross       ∈ R^{B×1×512}        — bidirectional interaction

    Args:
        emb_dim  : DSID embedding dim         (512)
        feat_ch  : encoder last-stage channels (2048)
        d        : cross-attention latent dim  (512)
        feat_hw  : spatial size of feature map (8 → 8×8 = 64 tokens)
    """

    def __init__(
        self,
        emb_dim: int = 512,
        feat_ch: int = 2048,
        d: int = 512,
        feat_hw: int = 8,
    ):
        super().__init__()
        self.d       = d
        self.n_tok   = feat_hw * feat_hw   # 64 spatial tokens

        # Query projections  W_Q^id, W_Q^sem ∈ R^{512×512}  (Eq. query_proj)
        self.W_Q_id  = nn.Linear(emb_dim, d, bias=False)
        self.W_Q_sem = nn.Linear(emb_dim, d, bias=False)

        # KV projections from feature maps  W_K, W_V ∈ R^{2048×512}  (Eq. kv_proj)
        self.W_K_id  = nn.Linear(feat_ch, d, bias=False)
        self.W_V_id  = nn.Linear(feat_ch, d, bias=False)
        self.W_K_sem = nn.Linear(feat_ch, d, bias=False)
        self.W_V_sem = nn.Linear(feat_ch, d, bias=False)

        # Concat projection  W_I ∈ R^{1024×512}  (Eq. ddep)
        self.W_I = nn.Linear(2 * d, d, bias=False)

    def _single_query_attn(
        self,
        Q: torch.Tensor,   # [B, 1, d]
        K: torch.Tensor,   # [B, N, d]
        V: torch.Tensor,   # [B, N, d]
    ) -> torch.Tensor:
        """Scaled dot-product attention with a single query token."""
        scale  = math.sqrt(self.d)
        attn   = F.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1)  # [B, 1, N]
        return attn @ V                                                  # [B, 1, d]

    def forward(
        self,
        E_id:  torch.Tensor,   # [B, 512]
        E_sem: torch.Tensor,   # [B, 512]
        F_id:  torch.Tensor,   # [B, 2048, 8, 8]
        F_sem: torch.Tensor,   # [B, 2048, 8, 8]
    ):
        """
        Returns
        -------
        Q_id   : [B, 1, 512]  — identity query token (conditioning embedding)
        Q_sem  : [B, 1, 512]  — semantic query token  (conditioning embedding)
        I_cross: [B, 1, 512]  — bidirectional interaction descriptor
        """
        B = E_id.shape[0]

        # ---- 1. Query projections  (Eq. query_proj) ----
        # E_id/E_sem ∈ R^{B×512} → Q ∈ R^{B×512} → unsqueeze → R^{B×1×512}
        Q_id  = self.W_Q_id(E_id).unsqueeze(1)    # [B, 1, 512]  (Eq. unsqueeze)
        Q_sem = self.W_Q_sem(E_sem).unsqueeze(1)   # [B, 1, 512]

        # ---- 2. Reshape feature maps to token sequences  (Eq. flatten) ----
        # [B, 2048, 8, 8] → [B, 64, 2048]
        F_id_tok  = F_id.flatten(2).transpose(1, 2)    # [B, 64, 2048]
        F_sem_tok = F_sem.flatten(2).transpose(1, 2)   # [B, 64, 2048]

        # ---- 3. KV projections from feature maps  (Eq. kv_proj) ----
        K_id  = self.W_K_id(F_id_tok)     # [B, 64, 512]
        V_id  = self.W_V_id(F_id_tok)     # [B, 64, 512]
        K_sem = self.W_K_sem(F_sem_tok)   # [B, 64, 512]
        V_sem = self.W_V_sem(F_sem_tok)   # [B, 64, 512]

        # ---- 4. Bidirectional cross-attention  (Eqs. did_sem / dsem_id) ----
        # I_{id←sem}: identity queries attending to semantic feature tokens
        I_id_from_sem  = self._single_query_attn(Q_id,  K_sem, V_sem)  # [B, 1, 512]
        # I_{sem←id}: semantic queries attending to identity feature tokens
        I_sem_from_id  = self._single_query_attn(Q_sem, K_id,  V_id)   # [B, 1, 512]

        # ---- 5. Concat + project  (Eqs. concat / ddep) ----
        I       = torch.cat([I_id_from_sem, I_sem_from_id], dim=-1)  # [B, 1, 1024]
        I_cross = self.W_I(I)                                         # [B, 1, 512]

        return Q_id, Q_sem, I_cross


# ---------------------------------------------------------------------------
# Single-Stage Diffusion Fusion Block
# Implements one cross-attention injection at one U-Net decoder stage
# ---------------------------------------------------------------------------

class DiffusionFusionBlock(nn.Module):
    """
    Fuses one conditioning token into one U-Net decoder stage via
    cross-attention with residual addition  (Eq. fusion).

    F_fus = F_unet + Proj( Softmax(Q_s K_s^T / √d) V_s )

    Args:
        unet_ch : U-Net channel count at this stage  (1280 / 640 / 320)
        cond_dim: conditioning token dimension        (512)
        d       : cross-attention latent dimension    (512)
    """

    def __init__(
        self,
        unet_ch:  int = 1280,
        cond_dim: int = 512,
        d:        int = 512,
    ):
        super().__init__()
        self.d       = d
        self.unet_ch = unet_ch

        # Project U-Net spatial features into query space
        self.W_Q = nn.Linear(unet_ch, d, bias=False)

        # Project conditioning token into key / value
        self.W_K = nn.Linear(cond_dim, d, bias=False)
        self.W_V = nn.Linear(cond_dim, d, bias=False)

        # Project attention output back to U-Net channel space  (Proj in Eq. fusion)
        self.proj_out = nn.Linear(d, unet_ch, bias=False)

        self.norm = nn.LayerNorm(unet_ch)

    def forward(
        self,
        F_unet: torch.Tensor,    # [B, C_s, H_s, W_s]
        cond:   torch.Tensor,    # [B, 1, cond_dim]
    ) -> torch.Tensor:
        """
        Returns fused feature map of same shape as F_unet.
        """
        B, C, H, W = F_unet.shape

        # Reshape U-Net features to query tokens  [B, H*W, C]
        F_flat = F_unet.flatten(2).transpose(1, 2)   # [B, HW, C]
        Q_s    = self.W_Q(F_flat)                     # [B, HW, d]

        # Project conditioning token into K / V  [B, 1, d]
        K_s = self.W_K(cond)   # [B, 1, d]
        V_s = self.W_V(cond)   # [B, 1, d]

        # Cross-attention  (single key/value token)  [B, HW, d]
        scale  = math.sqrt(self.d)
        attn   = F.softmax((Q_s @ K_s.transpose(-2, -1)) / scale, dim=-1)  # [B, HW, 1]
        F_att  = attn @ V_s                                                  # [B, HW, d]

        # Project back to U-Net channel space and residual addition
        F_out  = self.proj_out(F_att)              # [B, HW, C]
        F_out  = self.norm(F_flat + F_out)         # residual + LN
        return F_out.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# MutualReInteractionFusion — top-level module
# Combines interaction extraction + hierarchical diffusion fusion
# ---------------------------------------------------------------------------

class MutualReInteractionFusion(nn.Module):
    """
    Full MRF module (Section 3.2).

    Usage
    -----
    Instantiate once, attach to the ViDiExPo pipeline.

    Forward inputs
    ~~~~~~~~~~~~~~
    E_id, E_sem  : DSID factorized embeddings  [B, 512]
    F_id, F_sem  : last encoder feature maps   [B, 2048, 8, 8]
    unet_features: dict mapping stage channel to U-Net feature map
                   {1280: [B, 1280, H1, W1],
                     640: [B,  640, H3, W3],
                     320: [B,  320, H4, W4]}

    Forward outputs
    ~~~~~~~~~~~~~~~
    fused_features: same dict with fused feature maps
    I_cross       : bidirectional interaction descriptor  [B, 1, 512]
                    (may be useful for logging / ablation)

    Hierarchical conditioning assignment  (Eqs. esem_block / eid_block / ddep_block)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    E_sem   → Up Block 1  (C=1280, global semantics)
    E_id    → Up Block 3  (C= 640, structural identity)
    I_cross → Up Block 4  (C= 320, extracted interaction)

    Args:
        emb_dim  : DSID embedding dim       (512)
        feat_ch  : encoder last-stage dim   (2048)
        d        : cross-attention latent   (512)
        feat_hw  : feature map spatial size (8)
    """

    # Mapping: stage_channel → which conditioning token to use
    # Keys are the unet_ch values for Up Blocks 1, 3, 4
    STAGE_TOKEN = {
        1280: "E_sem",    # Up Block 1 — global semantics
         640: "E_id",     # Up Block 3 — structural identity
         320: "I_cross",  # Up Block 4 — extracted interaction
    }

    def __init__(
        self,
        emb_dim:  int = 512,
        feat_ch:  int = 2048,
        d:        int = 512,
        feat_hw:  int = 8,
    ):
        super().__init__()

        # ---- Interaction extraction ----
        self.extractor = MRFInteractionExtractor(
            emb_dim=emb_dim,
            feat_ch=feat_ch,
            d=d,
            feat_hw=feat_hw,
        )

        # ---- Hierarchical diffusion fusion blocks ----
        # One per Up Block (1280 / 640 / 320)
        self.fusion_blocks = nn.ModuleDict({
            str(ch): DiffusionFusionBlock(unet_ch=ch, cond_dim=d, d=d)
            for ch in self.STAGE_TOKEN
        })

    def forward(
        self,
        E_id:          torch.Tensor,          # [B, 512]
        E_sem:         torch.Tensor,          # [B, 512]
        F_id:          torch.Tensor,          # [B, 2048, 8, 8]
        F_sem:         torch.Tensor,          # [B, 2048, 8, 8]
        unet_features: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        E_id, E_sem     : DSID embeddings
        F_id, F_sem     : encoder feature maps
        unet_features   : {1280: tensor, 640: tensor, 320: tensor}

        Returns
        -------
        fused_features  : {1280: tensor, 640: tensor, 320: tensor}
        I_cross         : [B, 1, 512]
        """
        # ---- Step 1: Interaction extraction (Section 3.2.1) ----
        # Q_id / Q_sem are the unsqueezed conditioning tokens used in fusion
        Q_id, Q_sem, I_cross = self.extractor(E_id, E_sem, F_id, F_sem)

        # Conditioning token routing (Eq. conditioning)
        token_map = {
            1280: Q_sem,    # E_sem → Up Block 1
             640: Q_id,     # E_id  → Up Block 3
             320: I_cross,  # I_cross → Up Block 4
        }

        # ---- Step 2: Hierarchical diffusion fusion (Section 3.2.2) ----
        fused = {}
        for ch, F_unet in unet_features.items():
            cond  = token_map[ch]                      # [B, 1, 512]
            block = self.fusion_blocks[str(ch)]
            fused[ch] = block(F_unet, cond)

        return fused, I_cross


# ---------------------------------------------------------------------------
# MRFIntegration — convenience wrapper kept for pipeline compatibility
# ---------------------------------------------------------------------------

class MRFIntegration(nn.Module):
    """
    Thin wrapper around MutualReInteractionFusion for drop-in compatibility
    with ViDiExPoDiffusionPipeline / ViDiExPoUNetWrapper.

    The pipeline calls:
        fused_features = self.mrf(E_id, E_sem, F_id, F_sem, unet_features)

    unet_features is a dict {1280: ..., 640: ..., 320: ...} extracted
    from the SDXL U-Net decoder via forward hooks at:
        Up Block 1  (channel dim 1280)
        Up Block 3  (channel dim  640)
        Up Block 4  (channel dim  320)
    """

    def __init__(
        self,
        emb_dim:  int = 512,
        feat_ch:  int = 2048,
        d:        int = 512,
        feat_hw:  int = 8,
    ):
        super().__init__()
        self.mrf = MutualReInteractionFusion(
            emb_dim=emb_dim,
            feat_ch=feat_ch,
            d=d,
            feat_hw=feat_hw,
        )

    def forward(
        self,
        E_id:          torch.Tensor,
        E_sem:         torch.Tensor,
        F_id:          torch.Tensor,
        F_sem:         torch.Tensor,
        unet_features: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        return self.mrf(E_id, E_sem, F_id, F_sem, unet_features)
