"""
DSID Training Script — ViDiExPo

Trains the Dynamic Semantic and Identity Disentanglement (DSID) module.

Training setup (Section 4.1 + Supplementary S1.1):
    - Datasets: HDTF, VoxCeleb, VFHQ (4,242 unique IDs, 17,108 clips, 55 hrs)
    - Optimizer: Adam, lr=1e-4, cosine decay
    - Batch size: 4 per GPU × 4 GPUs = effective 16
    - Iterations: 100k
    - Identity pairs: dataset-level sampling (not in-batch)
    - Loss weights: λ1=1.0, λ2=0.1, λ3=1.0, λ4=0.1, λ5=0.1

Loss convergence:
    - L_MID: 3.53±0.75  →  1.03±0.25  (across 3 seeds)
    - L_DSID: 4.026 → 1.224 (−69.6%) over 100k iterations

Usage:
    python train_dsid.py --config configs/dsid_train.yaml
"""

import os
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from DSID.modules.dsid import DSIDModule, PatchDiscriminator
from DSID.modules.losses import DSIDLoss, AdversarialLoss
from DSID.data.video_frame_dataset import VideoFrameDataset
from utils.logger import DSIDLogger
from utils.checkpoint import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def get_default_config() -> dict:
    """
    Default DSID training configuration matching paper Section 4.1.
    """
    return {
        "training": {
            "num_iterations": 100_000,
            "batch_size_per_gpu": 4,
            "effective_batch_size": 16,      # 4 GPUs
            "learning_rate": 1e-4,
            "adam_betas": [0.5, 0.999],
            "cosine_decay": True,
            "checkpoint_freq": 5000,
            "log_freq": 100,
            "num_workers": 4,
            "seed": 42,
        },
        "loss_weights": {
            "lambda1": 1.0,   # L_recon
            "lambda2": 0.1,   # L_percep
            "lambda3": 1.0,   # L_adv
            "lambda4": 0.1,   # L_MID
            "lambda5": 0.1,   # L_ML
        },
        "dsid": {
            "emb_dim": 512,
            "num_identities": 4242,          # unique IDs in training set
            "pretrained_backbone": True,     # ResNet-50 pretrained on VGGFace2
        },
        "metric_learning": {
            "triplet_margin": 0.01,          # conservative (AAM-Softmax dominant)
            "aam_margin": 0.2,
            "aam_scale": 30.0,
        },
        "dataset": {
            "frame_size": 256,
            "sources": ["HDTF", "VoxCeleb", "VFHQ"],
            "min_face_resolution": 256,
            "max_yaw_deg": 60.0,
            "clips_per_identity": 3,         # 2-3 per Section S1.1
        },
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_dsid(
    config: dict,
    log_dir: str,
    checkpoint_path: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
):
    """
    DSID training loop.

    Args:
        config:           training configuration dict
        log_dir:          directory for logs and checkpoints
        checkpoint_path:  path to resume from (or None)
        local_rank:       GPU rank for DDP
        world_size:       total number of GPUs
    """
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Reproducibility ----
    seed = config["training"]["seed"] + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ---- Dataset ----
    dataset = VideoFrameDataset(
        data_roots=config["dataset"].get("data_roots", []),
        frame_size=config["dataset"]["frame_size"],
        min_face_resolution=config["dataset"]["min_face_resolution"],
        max_yaw_deg=config["dataset"]["max_yaw_deg"],
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset,
                                                         num_replicas=world_size,
                                                         rank=local_rank)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ---- Models ----
    cfg = config["dsid"]
    dsid = DSIDModule(
        emb_dim=cfg["emb_dim"],
        num_identities=cfg["num_identities"],
        pretrained_backbone=cfg["pretrained_backbone"],
        training_mode=True,
    ).to(device)

    discriminator = PatchDiscriminator(in_channels=3).to(device)

    # ---- DDP wrapping ----
    if world_size > 1:
        dsid          = DDP(dsid,          device_ids=[local_rank])
        discriminator = DDP(discriminator, device_ids=[local_rank])

    # ---- Losses ----
    lw = config["loss_weights"]
    dsid_loss = DSIDLoss(
        lambda1=lw["lambda1"],
        lambda2=lw["lambda2"],
        lambda3=lw["lambda3"],
        lambda4=lw["lambda4"],
        lambda5=lw["lambda5"],
    ).to(device)
    adv_loss = AdversarialLoss()

    # ---- Optimizers ----
    lr = config["training"]["learning_rate"]
    betas = tuple(config["training"]["adam_betas"])

    optimizer_dsid = torch.optim.Adam(dsid.parameters(), lr=lr, betas=betas)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(),
                                      lr=lr * 0.5, betas=betas)
    # Separate optimizer for CLUB auxiliary network (updates club params only)
    club_params = (
        list(dsid.module.club_s.parameters()) +
        list(dsid.module.club_t.parameters())
        if hasattr(dsid, "module") else
        list(dsid.club_s.parameters()) +
        list(dsid.club_t.parameters())
    )
    optimizer_club = torch.optim.Adam(club_params, lr=lr, betas=betas)

    # ---- LR Schedulers ----
    n_iters = config["training"]["num_iterations"]
    scheduler_dsid = CosineAnnealingLR(optimizer_dsid, T_max=n_iters)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=n_iters)

    # ---- Resume ----
    start_iter = 0
    if checkpoint_path is not None:
        start_iter = load_checkpoint(
            checkpoint_path, dsid, discriminator,
            optimizer_dsid, optimizer_disc, scheduler_dsid
        )
        if is_main:
            print(f"[DSID] Resumed from iteration {start_iter}")

    # ---- Logger ----
    logger = DSIDLogger(log_dir=log_dir) if is_main else None

    # ---- Training loop ----
    data_iter = iter(loader)
    dsid.train()
    discriminator.train()

    for iteration in range(start_iter, n_iters):
        # Fetch batch (restart loader if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(iteration)
            data_iter = iter(loader)
            batch = next(data_iter)

        I_s      = batch["source"].to(device, non_blocking=True)
        I_t      = batch["target"].to(device, non_blocking=True)
        labels   = batch["identity_label"].to(device, non_blocking=True)
        I_neg    = batch["negative"].to(device, non_blocking=True)

        # ---- DSID forward ----
        out = dsid(I_s, I_t, labels)

        I_hat_t  = out["I_hat_t"]
        E_id_s   = out["E_id"]
        E_sem_t  = out["E_sem"]
        E_id_t   = out["E_id_t"]
        L_MID    = out["L_MID"]
        L_aam    = out["L_aam"]

        # Negative identity embedding for triplet loss
        E_id_neg = (dsid.module if hasattr(dsid, "module") else dsid).f_id_s(I_neg)

        # ---- Discriminator step ----
        optimizer_disc.zero_grad()
        real_logits = discriminator(I_t.detach())
        fake_logits_d = discriminator(I_hat_t.detach())
        L_disc = adv_loss.discriminator_loss(real_logits, fake_logits_d)
        L_disc.backward()
        optimizer_disc.step()

        # ---- CLUB auxiliary network update ----
        optimizer_club.zero_grad()
        L_club_fit = out["L_club_fit"]
        L_club_fit.backward(retain_graph=True)
        optimizer_club.step()

        # ---- DSID generator step ----
        optimizer_dsid.zero_grad()
        fake_logits_g = discriminator(I_hat_t)

        loss_dict = dsid_loss(
            I_hat_t       = I_hat_t,
            I_t           = I_t,
            L_MID         = L_MID,
            L_aam         = L_aam,
            fake_logits   = fake_logits_g,
            E_id_anchor   = E_id_s,
            E_id_positive = E_id_t,
            E_id_negative = E_id_neg,
        )
        loss_dict["L_total"].backward()
        torch.nn.utils.clip_grad_norm_(dsid.parameters(), max_norm=1.0)
        optimizer_dsid.step()

        # Update LR schedulers
        scheduler_dsid.step()
        scheduler_disc.step()

        # ---- Logging ----
        if is_main and (iteration % config["training"]["log_freq"] == 0):
            log_dict = {k: v.item() for k, v in loss_dict.items()}
            log_dict["L_disc"] = L_disc.item()
            log_dict["L_MID_raw"] = L_MID.item()
            log_dict["lr"] = optimizer_dsid.param_groups[0]["lr"]
            logger.log(iteration, log_dict)

            if iteration % (config["training"]["log_freq"] * 10) == 0:
                logger.log_images(
                    iteration,
                    source=I_s[:4],
                    target=I_t[:4],
                    reconstructed=I_hat_t[:4],
                )

        # ---- Checkpointing ----
        if is_main and (
            iteration % config["training"]["checkpoint_freq"] == 0
            or iteration == n_iters - 1
        ):
            save_checkpoint(
                log_dir, iteration,
                dsid=dsid,
                discriminator=discriminator,
                optimizer_dsid=optimizer_dsid,
                optimizer_disc=optimizer_disc,
                scheduler_dsid=scheduler_dsid,
            )
            if is_main:
                print(f"[DSID] Saved checkpoint at iteration {iteration}")

    if is_main:
        print("[DSID] Training complete.")
        # Save inference-only weights (without training components)
        save_checkpoint(
            log_dir, n_iters,
            dsid=dsid,
            inference_only=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViDiExPo DSID Training")
    parser.add_argument("--config",      type=str, default="configs/dsid_train.yaml")
    parser.add_argument("--log_dir",     type=str, default="logs/dsid")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--local_rank",  type=int, default=0)
    parser.add_argument("--world_size",  type=int, default=4)    # 4× RTX 4090
    args = parser.parse_args()

    # Load config
    import yaml
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
        print(f"[DSID] Config not found at {args.config}, using defaults.")

    # Initialize distributed training
    if args.world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    train_dsid(
        config=config,
        log_dir=args.log_dir,
        checkpoint_path=args.checkpoint,
        local_rank=args.local_rank,
        world_size=args.world_size,
    )


if __name__ == "__main__":
    main()
