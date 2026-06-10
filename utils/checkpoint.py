"""
ViDiExPo Checkpoint Utilities

Handles saving and loading of DSID and MRF checkpoints with full state recovery.
"""

import os
import torch
from pathlib import Path


def save_checkpoint(
    log_dir: str,
    iteration: int | str,
    dsid=None,
    discriminator=None,
    mrf=None,
    optimizer_dsid=None,
    optimizer_disc=None,
    optimizer=None,
    scheduler_dsid=None,
    inference_only: bool = False,
):
    """
    Save a training checkpoint.

    For inference-only export, saves only the encoder weights needed
    for deployment (f_id^s, f_sem^t, HAL, and MRF modules).
    """
    state = {"iteration": iteration}

    def get_state_dict(module):
        """Handle DDP wrapping."""
        if module is None:
            return None
        return (module.module.state_dict()
                if hasattr(module, "module") else module.state_dict())

    if inference_only:
        # Export only inference-relevant components
        if dsid is not None:
            dsid_sd = get_state_dict(dsid)
            # Filter to only retain f_id_s and f_sem_t
            inference_keys = {
                k: v for k, v in dsid_sd.items()
                if k.startswith("f_id_s.") or k.startswith("f_sem_t.")
            }
            state["dsid_inference"] = inference_keys
        if mrf is not None:
            state["mrf"] = get_state_dict(mrf)
        fname = f"viexpo_inference_{iteration}.pth"
    else:
        if dsid          is not None: state["dsid"]           = get_state_dict(dsid)
        if discriminator is not None: state["discriminator"]  = get_state_dict(discriminator)
        if mrf           is not None: state["mrf"]            = get_state_dict(mrf)
        if optimizer_dsid is not None: state["optimizer_dsid"] = optimizer_dsid.state_dict()
        if optimizer_disc is not None: state["optimizer_disc"] = optimizer_disc.state_dict()
        if optimizer      is not None: state["optimizer"]      = optimizer.state_dict()
        if scheduler_dsid is not None: state["scheduler_dsid"] = scheduler_dsid.state_dict()
        fname = f"checkpoint_{iteration}.pth"

    path = Path(log_dir) / fname
    torch.save(state, path)
    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    dsid=None,
    discriminator=None,
    mrf=None,
    optimizer_dsid=None,
    optimizer_disc=None,
    optimizer=None,
    scheduler_dsid=None,
    strict: bool = False,
) -> int:
    """
    Load a training checkpoint and restore all states.
    Returns the saved iteration number (for resuming).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    def load_state(module, key):
        if module is not None and key in ckpt:
            target = module.module if hasattr(module, "module") else module
            target.load_state_dict(ckpt[key], strict=strict)

    load_state(dsid,          "dsid")
    load_state(discriminator, "discriminator")
    load_state(mrf,           "mrf")

    if optimizer_dsid is not None and "optimizer_dsid" in ckpt:
        optimizer_dsid.load_state_dict(ckpt["optimizer_dsid"])
    if optimizer_disc is not None and "optimizer_disc" in ckpt:
        optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler_dsid is not None and "scheduler_dsid" in ckpt:
        scheduler_dsid.load_state_dict(ckpt["scheduler_dsid"])

    return ckpt.get("iteration", 0)
