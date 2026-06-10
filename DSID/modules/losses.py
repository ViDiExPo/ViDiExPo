"""
DSID Loss Functions — ViDiExPo

Total DSID loss (Eq. 17):
    L_DSID = λ1*L_recon + λ2*L_percep + λ3*L_adv + λ4*L_MID + λ5*L_ML

Default weights per Section 4.1:
    λ1=1.0, λ2=0.1, λ3=1.0, λ4=0.1, λ5=0.1

Reconstruction, perceptual, and adversarial losses follow LIA [45].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


# ---------------------------------------------------------------------------
# VGG-19 Perceptual Loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-19 feature activations.
    Used in both DSID (L_percep) and the diffusion pipeline (L_id, Eq. 32).
    """

    _VGG_LAYERS = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 18,
        "relu4_3": 27,
    }

    def __init__(self, layers: list[str] | None = None, normalize: bool = True):
        super().__init__()
        if layers is None:
            layers = list(self._VGG_LAYERS.keys())
        self.normalize = normalize

        vgg = vgg19(weights="IMAGENET1K_V1")
        vgg.eval()
        features = list(vgg.features.children())

        self.slices = nn.ModuleList()
        prev_idx = 0
        for lname in layers:
            end_idx = self._VGG_LAYERS[lname] + 1
            self.slices.append(nn.Sequential(*features[prev_idx:end_idx]))
            prev_idx = end_idx

        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = self._preprocess(pred)
        target = self._preprocess(target)
        loss = torch.tensor(0.0, device=pred.device)
        for sl in self.slices:
            pred   = sl(pred)
            target = sl(target)
            loss   = loss + F.l1_loss(pred, target)
        return loss


# ---------------------------------------------------------------------------
# Adversarial Loss (following LIA [45])
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator (following LIA).
    Used for adversarial loss L_adv during DSID training.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for _ in range(n_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, 4, 2, 1),
                nn.InstanceNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdversarialLoss(nn.Module):
    """Least-squares GAN adversarial losses."""

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        loss_real = F.mse_loss(real_logits, torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# Triplet Loss  (metric learning component, Section 3.1.3)
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """
    Triplet loss with L2 distance metric.
    Margin t = 0.01 (conservative complement to AAM-Softmax).

    L_ML = max(0, d(a, S_p) - d(a, S_n) + t)   [Eq. 14]
    """

    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        loss  = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


# ---------------------------------------------------------------------------
# Full DSID Loss (Eq. 17)
# ---------------------------------------------------------------------------

class DSIDLoss(nn.Module):
    """
    Total DSID training loss:
        L_DSID = λ1*L_recon + λ2*L_percep + λ3*L_adv + λ4*L_MID + λ5*L_ML

    Weights λ1=1.0, λ2=0.1, λ3=1.0, λ4=0.1, λ5=0.1  (Section 4.1)
    """

    def __init__(
        self,
        lambda1: float = 1.0,   # L_recon
        lambda2: float = 0.1,   # L_percep
        lambda3: float = 1.0,   # L_adv
        lambda4: float = 0.1,   # L_MID
        lambda5: float = 0.1,   # L_ML
    ):
        super().__init__()
        self.lambdas = (lambda1, lambda2, lambda3, lambda4, lambda5)
        self.perceptual  = VGGPerceptualLoss()
        self.adversarial = AdversarialLoss()
        self.triplet     = TripletLoss(margin=0.01)

    def forward(
        self,
        I_hat_t: torch.Tensor,      # reconstructed target frame
        I_t:     torch.Tensor,      # ground-truth target frame
        L_MID:   torch.Tensor,      # CLUB MI estimate (from DSID forward)
        L_aam:   torch.Tensor,      # AAM-Softmax loss (from DSID forward)
        fake_logits: torch.Tensor,  # discriminator output on I_hat_t
        # For triplet loss:
        E_id_anchor:   torch.Tensor,
        E_id_positive: torch.Tensor,
        E_id_negative: torch.Tensor,
    ) -> dict:
        λ1, λ2, λ3, λ4, λ5 = self.lambdas

        # Reconstruction loss
        L_recon = F.l1_loss(I_hat_t, I_t)

        # Perceptual loss
        L_percep = self.perceptual(I_hat_t, I_t)

        # Adversarial loss (generator side)
        L_adv = self.adversarial.generator_loss(fake_logits)

        # Metric learning: triplet + AAM-Softmax
        L_triplet = self.triplet(E_id_anchor, E_id_positive, E_id_negative)
        L_ML = L_triplet + L_aam

        # Total loss
        L_total = (λ1 * L_recon + λ2 * L_percep + λ3 * L_adv +
                   λ4 * L_MID   + λ5 * L_ML)

        return {
            "L_total":  L_total,
            "L_recon":  L_recon.detach(),
            "L_percep": L_percep.detach(),
            "L_adv":    L_adv.detach(),
            "L_MID":    L_MID.detach(),
            "L_ML":     L_ML.detach(),
        }
