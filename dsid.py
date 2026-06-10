"""
Dynamic Semantic and Identity Disentanglement (DSID) Module
ViDiExPo: Video-Driven Disentangled Identity–Semantic Fusion
for Controllable Expression and Pose in Diffusion

Architecture:
- Symmetric dual-pathway design: source and target pathways
- Identity Encoder (f_id): captures identity-consistent features
- Semantic Encoder (f_sem): captures dynamic semantic variation via HAL
- Metric Learning (ML): triplet loss + AAM-Softmax for identity discrimination
- Mutual Information Disentanglement (MID): CLUB-based MI minimization
- Hierarchical Aggregation Layer (HAL): multi-scale semantic aggregation
- Wrapper Layer (W) + Image Decoder (D): reconstruction during training only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ---------------------------------------------------------------------------
# Hierarchical Aggregation Layer (HAL)
# Applied exclusively to the semantic encoder (Section 3.1.5)
# ---------------------------------------------------------------------------

class HAL(nn.Module):
    """
    Hierarchical Aggregation Layer.

    Aggregates features from all n stages of the image encoder via
    learnable weighted average pooling:

        E_sem = HAL({phi_i}_{i=1}^{n}) = sum_i w_i * AvgPool(phi_i)

    where {w_i} are learnable scalars initialised uniformly at 1/n
    and optimised jointly with the semantic encoder (Eq. 16).

    Applied ONLY to the semantic encoder — not the identity encoder.
    """

    def __init__(self, stage_channels: list[int], out_dim: int = 512):
        """
        Args:
            stage_channels: list of channel dims for each ResNet stage
            out_dim: output embedding dimension (512 per paper)
        """
        super().__init__()
        n = len(stage_channels)
        # Learnable scalar weights, initialised uniformly
        self.weights = nn.Parameter(torch.ones(n) / n)
        # Per-stage linear projections to shared out_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, out_dim),
            )
            for c in stage_channels
        ])

    def forward(self, stage_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stage_features: list of feature tensors [B, C_i, H_i, W_i]
        Returns:
            E_sem: [B, out_dim]
        """
        w = F.softmax(self.weights, dim=0)          # normalise: sum = 1
        agg = sum(w[i] * self.projections[i](stage_features[i])
                  for i in range(len(stage_features)))
        return agg


# ---------------------------------------------------------------------------
# CLUB Mutual Information Upper Bound Estimator (Section 3.1.4)
# Reference: Cheng et al. 2020, "CLUB: A Contrastive Log-ratio Upper Bound"
# ---------------------------------------------------------------------------

class CLUBEstimator(nn.Module):
    """
    Contrastive Log-ratio Upper Bound (CLUB) of mutual information.

    Estimates MI upper bound via a conditional density network q_phi(y|x):

        CLUB(X; Y) = E[log q(y|x)] - E[log q(y|x')]

    where x' is a sample independent of y.
    Used as L_MID to minimise I(E_id; E_sem) (Eq. 15).
    """

    def __init__(self, x_dim: int = 512, y_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        # Conditional density network mu(x) and log_var(x)
        self.mu_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim),
            nn.Tanh(),   # bound log-variance
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute CLUB upper-bound estimate.
        Args:
            x: identity embedding  [B, x_dim]
            y: semantic embedding  [B, y_dim]
        Returns:
            CLUB MI estimate (scalar, to be minimised)
        """
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)

        # Positive term: log q(y_i | x_i)
        pos = -0.5 * (logvar + (y - mu).pow(2) / logvar.exp()).sum(-1)

        # Negative term: mean over all j != i of log q(y_j | x_i)
        mu_expand = mu.unsqueeze(1)                 # [B, 1, D]
        y_expand  = y.unsqueeze(0)                  # [1, B, D]
        logvar_expand = logvar.unsqueeze(1)         # [B, 1, D]
        neg = -0.5 * (logvar_expand + (y_expand - mu_expand).pow(2)
                      / logvar_expand.exp()).sum(-1).mean(-1)   # [B]

        return (pos - neg).mean()

    def learning_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Loss for updating the auxiliary network parameters only.
        Maximises log q(y|x) as a fitting objective.
        """
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)
        return 0.5 * (logvar + (y - mu).pow(2) / logvar.exp()).sum(-1).mean()


# ---------------------------------------------------------------------------
# AAM-Softmax (Additive Angular Margin Softmax) (Section 3.1.3)
# Reference: Deng et al. 2019, ArcFace
# ---------------------------------------------------------------------------

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax / ArcFace).

    Used as an auxiliary angular-margin regulariser for identity discrimination.
    margin m = 0.2, scale s = 30 (cosine distance), as per Section 3.1.3.
    """

    def __init__(self, in_features: int, num_classes: int,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise both embeddings and class centres
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cosine = x_norm @ w_norm.T                     # [B, C]

        # Add angular margin to the target class
        theta = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7).acos()
        theta_m = (theta + self.margin).clamp(max=torch.pi)
        cos_m   = theta_m.cos()

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1)
        logits  = self.scale * (one_hot * cos_m + (1 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# ResNet-50 Backbone with Stage-wise Feature Extraction
# Pretrained on VGGFace2 per Section S1.1 of the supplementary
# ---------------------------------------------------------------------------

class ImageEncoderBackbone(nn.Module):
    """
    ResNet-50 backbone that exposes intermediate stage features
    needed by HAL for multi-scale aggregation.

    Stages exposed (channels): [256, 512, 1024, 2048]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1  = base.layer1   # -> [B, 256,  H/4,  W/4]
        self.layer2  = base.layer2   # -> [B, 512,  H/8,  W/8]
        self.layer3  = base.layer3   # -> [B, 1024, H/16, W/16]
        self.layer4  = base.layer4   # -> [B, 2048, H/32, W/32]
        self.stage_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns list of [phi1, phi2, phi3, phi4] stage feature maps.
        """
        x = self.stem(x)
        phi1 = self.layer1(x)
        phi2 = self.layer2(phi1)
        phi3 = self.layer3(phi2)
        phi4 = self.layer4(phi3)
        return [phi1, phi2, phi3, phi4]


# ---------------------------------------------------------------------------
# Identity Encoder  f_id
# ---------------------------------------------------------------------------

class IdentityEncoder(nn.Module):
    """
    Identity encoder for one pathway.

    Architecture:
        ImageEncoderBackbone -> AdaptiveAvgPool -> FC -> 512-dim embedding

    Optimised by metric learning (triplet + AAM-Softmax).
    HAL is NOT applied here (Section 3.1.5).
    """

    def __init__(self, pretrained: bool = True, emb_dim: int = 512):
        super().__init__()
        self.backbone = ImageEncoderBackbone(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            E_id: [B, 512]
        """
        stages = self.backbone(x)
        feat   = stages[-1]                  # use deepest stage for identity
        return self.fc(self.pool(feat))

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both the embedding and the last-stage feature map.

        Returns
        -------
        E_id : [B, 512]
        F_id : [B, 2048, 8, 8]   — last ResNet-50 stage (stage4) output
        """
        stages = self.backbone(x)
        feat   = stages[-1]                  # [B, 2048, H/32, W/32]
        return self.fc(self.pool(feat)), feat


# ---------------------------------------------------------------------------
# Semantic Encoder  f_sem   (with HAL)
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """
    Semantic encoder with Hierarchical Aggregation Layer (HAL).

    Architecture:
        ImageEncoderBackbone -> HAL({phi_i}) -> 512-dim embedding

    HAL aggregates features from all 4 ResNet stages with learnable
    scalar weights (Eq. 16), capturing localized semantic cues
    (expression, mouth config, gaze direction).
    Uses average pooling as the final HAL variant (best Sem_dist per
    ablation Table 8).
    """

    def __init__(self, pretrained: bool = True, emb_dim: int = 512):
        super().__init__()
        self.backbone = ImageEncoderBackbone(pretrained=pretrained)
        self.hal = HAL(
            stage_channels=self.backbone.stage_channels,
            out_dim=emb_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            E_sem: [B, 512]
        """
        stages = self.backbone(x)
        return self.hal(stages)

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both the embedding and the last-stage feature map.

        Returns
        -------
        E_sem : [B, 512]
        F_sem : [B, 2048, 8, 8]   — last ResNet-50 stage (stage4) output
        """
        stages = self.backbone(x)
        return self.hal(stages), stages[-1]


# ---------------------------------------------------------------------------
# Wrapper Layer  W  and lightweight Image Decoder  D
# Used ONLY during training for reconstruction objective (Eq. 7)
# ---------------------------------------------------------------------------

class WrapperLayer(nn.Module):
    """
    Combines source identity embedding and target semantic embedding
    into a joint conditioning vector for the decoder.
    """

    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
        )

    def forward(self, E_id: torch.Tensor, E_sem: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([E_id, E_sem], dim=-1))


class ImageDecoder(nn.Module):
    """
    Lightweight convolutional decoder for reconstruction training.
    Reconstructs target frame from W(E_id^s, E_sem^t).
    Discarded at inference (Section 3.1.1).
    """

    def __init__(self, cond_dim: int = 512, out_channels: int = 3,
                 base_channels: int = 64):
        super().__init__()
        self.fc = nn.Linear(cond_dim, base_channels * 8 * 8 * 8)
        self.decode = nn.Sequential(
            nn.Unflatten(1, (base_channels * 8, 8, 8)),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels,     4, 2, 1),
            nn.BatchNorm2d(base_channels),     nn.ReLU(),
            nn.ConvTranspose2d(base_channels,   base_channels // 2,  4, 2, 1),
            nn.BatchNorm2d(base_channels // 2), nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 2, out_channels,     4, 2, 1),
            nn.Tanh(),
        )  # 8 -> 256 px through 5 ups

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.fc(z))


# ---------------------------------------------------------------------------
# Full DSID Module (training + inference modes)
# ---------------------------------------------------------------------------

class DSIDModule(nn.Module):
    """
    Dynamic Semantic and Identity Disentanglement (DSID) Module.

    Training mode (Figure 2, left):
        - Four encoders: f_id^s, f_sem^s, f_id^t, f_sem^t  (separate weights)
        - Reconstruction: hat_I_t = D(W(E_id^s, E_sem^t))
        - Losses: L_recon, L_percep, L_adv, L_MID, L_ML

    Inference mode (Figure 2, right):
        - Only two encoders retained: f_id^s (from identity image) and
          f_sem^t (from reference image)
        - Returns (E_id, E_sem) for MRF conditioning

    Identity-controlled contrastive supervision signal (Section 3.1.2):
        - Paired frames from same video: ID(I^s) = ID(I^t), Sem(I^s) ≠ Sem(I^t)
        - No temporal ordering assumed; static-image inference compatible
    """

    def __init__(
        self,
        emb_dim: int = 512,
        num_identities: int = 4242,     # number of training identities
        pretrained_backbone: bool = True,
        training_mode: bool = True,
    ):
        super().__init__()
        self.training_mode = training_mode
        self.emb_dim = emb_dim

        # --- Source pathway encoders ---
        self.f_id_s   = IdentityEncoder(pretrained=pretrained_backbone, emb_dim=emb_dim)
        self.f_sem_s  = SemanticEncoder(pretrained=pretrained_backbone, emb_dim=emb_dim)

        # --- Target pathway encoders ---
        self.f_id_t   = IdentityEncoder(pretrained=pretrained_backbone, emb_dim=emb_dim)
        self.f_sem_t  = SemanticEncoder(pretrained=pretrained_backbone, emb_dim=emb_dim)

        if training_mode:
            # Training-only components
            self.wrapper  = WrapperLayer(emb_dim=emb_dim)
            self.decoder  = ImageDecoder(cond_dim=emb_dim)

            # Metric learning
            self.aam_softmax = AAMSoftmax(
                in_features=emb_dim,
                num_classes=num_identities,
                margin=0.2,
                scale=30.0,
            )

            # MID — one CLUB estimator per pathway direction
            self.club_s   = CLUBEstimator(x_dim=emb_dim, y_dim=emb_dim)
            self.club_t   = CLUBEstimator(x_dim=emb_dim, y_dim=emb_dim)

    def forward(
        self,
        I_s: torch.Tensor,
        I_t: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass.

        Training:
            I_s: source frame  [B, 3, H, W]
            I_t: target frame  [B, 3, H, W]  (same identity, different semantics)
            labels: identity class indices    [B]

        Inference:
            I_s: identity image (I_id)         [B, 3, H, W]
            I_t: None  (semantic from f_sem^t is extracted separately)

        Returns:
            dict with 'E_id', 'E_sem', and training losses if in training mode
        """
        if not self.training_mode or I_t is None:
            # ---- Inference: extract factorized embeddings ----
            E_id  = self.f_id_s(I_s)
            return {"E_id": E_id}

        # ---- Training ----
        # Source pathway
        E_id_s  = self.f_id_s(I_s)
        E_sem_s = self.f_sem_s(I_s)

        # Target pathway
        E_id_t  = self.f_id_t(I_t)
        E_sem_t = self.f_sem_t(I_t)

        # Reconstruction: hat_I_t = D(W(E_id^s, E_sem^t))  [Eq. 7]
        z        = self.wrapper(E_id_s, E_sem_t)
        I_hat_t  = self.decoder(z)

        # MID losses (applied independently to both branches, Eq. 15)
        L_MID = (self.club_s(E_id_s, E_sem_s) +
                 self.club_t(E_id_t, E_sem_t)) * 0.5

        # CLUB auxiliary fitting losses (for updating CLUB network params)
        L_club_fit = (self.club_s.learning_loss(E_id_s, E_sem_s) +
                      self.club_t.learning_loss(E_id_t, E_sem_t)) * 0.5

        # AAM-Softmax identity discrimination
        L_aam = torch.tensor(0.0, device=I_s.device)
        if labels is not None:
            L_aam = (self.aam_softmax(E_id_s, labels) +
                     self.aam_softmax(E_id_t, labels)) * 0.5

        return {
            # Inference-relevant embeddings (source id, target sem)
            "E_id":       E_id_s,
            "E_sem":      E_sem_t,
            # Training-relevant extras
            "E_id_t":     E_id_t,
            "E_sem_s":    E_sem_s,
            "I_hat_t":    I_hat_t,
            # Losses
            "L_MID":      L_MID,
            "L_club_fit": L_club_fit,
            "L_aam":      L_aam,
        }

    def extract_semantic_embedding(self, I_ref: torch.Tensor) -> torch.Tensor:
        """
        Inference-time semantic embedding extraction from reference image.
        Uses f_sem^t (target semantic encoder).

        Args:
            I_ref: reference image  [B, 3, H, W]
        Returns:
            E_sem: [B, 512]
        """
        return self.f_sem_t(I_ref)

    def extract_identity_embedding(self, I_id: torch.Tensor) -> torch.Tensor:
        """
        Inference-time identity embedding extraction from identity image.
        Uses f_id^s (source identity encoder).

        Args:
            I_id: identity image  [B, 3, H, W]
        Returns:
            E_id: [B, 512]
        """
        return self.f_id_s(I_id)

    def extract_identity_embedding_and_features(
        self, I_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-time identity embedding + last-stage feature map.
        Required by MRF for key/value construction (Section 3.2.1, Eq. feature_maps).

        Args:
            I_id: identity image  [B, 3, H, W]
        Returns:
            E_id : [B, 512]
            F_id : [B, 2048, 8, 8]
        """
        return self.f_id_s.forward_with_features(I_id)

    def extract_semantic_embedding_and_features(
        self, I_ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-time semantic embedding + last-stage feature map.
        Required by MRF for key/value construction (Section 3.2.1, Eq. feature_maps).

        Args:
            I_ref: reference image  [B, 3, H, W]
        Returns:
            E_sem : [B, 512]
            F_sem : [B, 2048, 8, 8]
        """
        return self.f_sem_t.forward_with_features(I_ref)

    def set_inference_mode(self):
        """Switch to inference mode (discard training-only components)."""
        self.training_mode = False
        # Remove training-only modules to save memory
        for attr in ["wrapper", "decoder", "aam_softmax", "club_s", "club_t"]:
            if hasattr(self, attr):
                delattr(self, attr)
