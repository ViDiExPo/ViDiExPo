"""
ViDiExPo Utility Modules — Logging and Checkpointing
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# DSID Logger
# ---------------------------------------------------------------------------

class DSIDLogger:
    """
    Logging for DSID training.
    Tracks all five loss components: L_recon, L_percep, L_adv, L_MID, L_ML.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "dsid_train_log.jsonl"
        self.history  = defaultdict(list)
        self._start   = time.time()

    def log(self, iteration: int, metrics: dict):
        entry = {"iteration": iteration, "time": time.time() - self._start}
        entry.update(metrics)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.history["iteration"].append(iteration)
        for k, v in metrics.items():
            self.history[k].append(v)

        # Console summary
        loss_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
            if k.startswith("L_") or k == "lr"
        )
        print(f"[DSID] iter {iteration:6d}  {loss_str}")

    def log_images(
        self,
        iteration: int,
        source: torch.Tensor,
        target: torch.Tensor,
        reconstructed: torch.Tensor,
    ):
        """Save visualization grid every N iterations."""
        import torchvision.utils as vutils
        vis_dir = self.log_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        grid = vutils.make_grid(
            torch.cat([source, target, reconstructed], dim=0),
            nrow=source.shape[0],
            normalize=True,
            value_range=(-1, 1),
        )
        vutils.save_image(grid, vis_dir / f"iter_{iteration:07d}.png")


# ---------------------------------------------------------------------------
# Diffusion Logger
# ---------------------------------------------------------------------------

class DiffusionLogger:
    """
    Logging for diffusion fine-tuning.
    Tracks L_SD, L_id, L_sem (Eq. 34).
    """

    def __init__(self, log_dir: str):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "diffusion_train_log.jsonl"
        self._start   = time.time()

    def log(self, step: int, metrics: dict):
        entry = {"step": step, "time": time.time() - self._start}
        entry.update(metrics)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        loss_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
            if isinstance(v, (int, float))
        )
        print(f"[Diffusion] step {step:7d}  {loss_str}")
