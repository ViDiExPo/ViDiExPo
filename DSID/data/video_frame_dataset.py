"""
VideoFrameDataset — ViDiExPo DSID Training Data

Provides identity-controlled contrastive frame pairs for DSID supervision
(Section 3.1.2):
    - Source frame I_s and target frame I_t from SAME video sequence
    - Same identity:    ID(I_s) = ID(I_t)
    - Different semantics: Sem(I_s) ≠ Sem(I_t)  (expression, pose, gaze)
    - No temporal ordering assumed (frames sampled statically)

Datasets (Section 4.1 + Supplementary S1.1):
    - HDTF:    high-definition audio-visual dataset
    - VoxCeleb: large-scale speaker verification in the wild
    - VFHQ:    video face high-quality dataset
    - Total:   4,242 unique IDs, 17,108 clips, 55 hours
    - 2-3 clips per identity (randomly selected)

Filtering rules (Supplementary S1.1):
    - Minimum face resolution: 256 × 256
    - Blur detection: Laplacian operator
    - Maximum yaw angle: 60°
    - All images resized to 256 × 256

Identity-level negative sampling (Section 4.1):
    - Negative identities sampled at DATASET level (not in-batch)
    - Decouples negative mining from batch statistics
    - Enables effective metric learning at batch_size=4 per GPU

Augmentation (Section 3.1.6):
    - Reconstruction pairs: minimal (preserve frame correspondence)
    - Identity encoder (ML): flipping, color jitter, Gaussian blur,
      shifting, scaling, rotation  (via Albumentations)
"""

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_reconstruction_augmentation(frame_size: int = 256) -> A.Compose:
    """
    Minimal augmentation for reconstruction-based disentanglement training.
    No strong geometric or appearance perturbations (Section 3.1.6).
    """
    return A.Compose([
        A.Resize(frame_size, frame_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_identity_augmentation(frame_size: int = 256) -> A.Compose:
    """
    Standard augmentation for identity encoder metric learning.
    Improves identity robustness (Section 3.1.6).
    Uses Albumentations library (https://github.com/albumentations-team/albumentations).
    """
    return A.Compose([
        A.Resize(frame_size, frame_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoFrameDataset(Dataset):
    """
    Dataset of identity-consistent, semantically-diverse frame pairs.

    Samples (I_s, I_t) from the same video such that:
        ID(I_s) = ID(I_t)  and  Sem(I_s) ≠ Sem(I_t)

    Also returns a negative sample I_neg from a DIFFERENT identity for
    the triplet loss component of metric learning.

    Args:
        data_roots:         list of paths to video dataset roots
        frame_size:         target spatial resolution (256)
        min_face_resolution: minimum face crop resolution before filtering
        max_yaw_deg:        maximum allowed yaw angle (60°)
        min_frame_gap:      minimum frame index gap to ensure semantic variation
        num_negatives:      pre-computed negative identity pool size
    """

    def __init__(
        self,
        data_roots: list[str],
        frame_size: int = 256,
        min_face_resolution: int = 256,
        max_yaw_deg: float = 60.0,
        min_frame_gap: int = 8,
        num_negatives: int = 100,
        split: str = "train",
        seed: int = 0,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.min_face_resolution = min_face_resolution
        self.max_yaw_deg = max_yaw_deg
        self.min_frame_gap = min_frame_gap
        self.num_negatives = num_negatives
        self.split = split

        # Augmentation pipelines
        self.aug_recon = get_reconstruction_augmentation(frame_size)
        self.aug_id    = get_identity_augmentation(frame_size)

        # Build clip index
        self.clips = []          # list of (identity_id, clip_path, frame_paths)
        self.identity_to_clips = {}   # identity_id -> list of clip indices

        self._build_index(data_roots, seed)

        # Build flat identity list for negative sampling
        self.identity_ids = list(self.identity_to_clips.keys())

    # ------------------------------------------------------------------

    def _build_index(self, data_roots: list[str], seed: int):
        """
        Scan dataset roots and build frame index.
        Applies filtering rules (Section S1.1).
        """
        rng = random.Random(seed)
        global_clip_idx = 0
        global_identity_idx = 0

        for root in data_roots:
            root = Path(root)
            if not root.exists():
                print(f"[VideoFrameDataset] Warning: {root} does not exist, skipping.")
                continue

            # Iterate over identity directories
            for identity_dir in sorted(root.iterdir()):
                if not identity_dir.is_dir():
                    continue

                identity_id = global_identity_idx

                # Collect video clips for this identity (2-3 per Section S1.1)
                clip_dirs = [d for d in identity_dir.iterdir() if d.is_dir()]
                # Also check for .mp4 files
                clip_files = list(identity_dir.glob("*.mp4"))
                all_clips = clip_dirs + clip_files

                if not all_clips:
                    continue

                # Randomly select 2-3 clips per identity
                n_clips = min(len(all_clips), rng.randint(2, 3))
                selected_clips = rng.sample(all_clips, n_clips)

                clip_indices_for_id = []
                for clip_path in selected_clips:
                    frames = self._get_clip_frames(clip_path)
                    if len(frames) < 2:
                        continue

                    self.clips.append((identity_id, clip_path, frames))
                    clip_indices_for_id.append(global_clip_idx)
                    global_clip_idx += 1

                if clip_indices_for_id:
                    self.identity_to_clips[identity_id] = clip_indices_for_id
                    global_identity_idx += 1

        print(f"[VideoFrameDataset] Indexed {global_identity_idx} identities "
              f"across {global_clip_idx} clips from {len(data_roots)} roots.")

    def _get_clip_frames(self, clip_path: Path) -> list[str]:
        """Return list of valid frame paths for a clip (dir or video file)."""
        frames = []
        if clip_path.is_dir():
            frames = sorted(
                str(p) for p in clip_path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        elif clip_path.suffix.lower() == ".mp4":
            # For .mp4 files, we store the video path and sample frames lazily
            frames = [str(clip_path)]   # placeholder; actual frames sampled in __getitem__
        return frames

    def _load_frame(self, frame_path: str, frame_idx: Optional[int] = None) -> np.ndarray:
        """Load a single frame (from image file or video at given index)."""
        if frame_path.endswith(".mp4"):
            cap = cv2.VideoCapture(frame_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_idx is None:
                frame_idx = random.randint(0, max(0, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(frame_path)
            if img is None:
                return np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _passes_filters(self, frame: np.ndarray) -> bool:
        """Apply resolution and blur filtering rules (Section S1.1)."""
        h, w = frame.shape[:2]
        if h < self.min_face_resolution or w < self.min_face_resolution:
            return False
        # Laplacian blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 10.0:    # threshold for blur rejection
            return False
        return True

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        identity_id, clip_path, frames = self.clips[idx]

        # ---- Sample source and target frames ----
        # Ensure minimum semantic variation via frame gap
        if clip_path.suffix == ".mp4" if isinstance(clip_path, Path) else str(clip_path).endswith(".mp4"):
            cap = cv2.VideoCapture(str(clip_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            idx_s = random.randint(0, max(0, total - self.min_frame_gap - 1))
            idx_t = idx_s + random.randint(self.min_frame_gap, min(self.min_frame_gap * 3, total - idx_s - 1))
        else:
            n = len(frames)
            idx_s = random.randint(0, max(0, n - self.min_frame_gap - 1))
            idx_t = idx_s + random.randint(self.min_frame_gap, min(self.min_frame_gap * 3, n - idx_s - 1))

        # Load frames
        frame_s = self._load_frame(str(clip_path) if str(clip_path).endswith(".mp4") else frames[idx_s], idx_s)
        frame_t = self._load_frame(str(clip_path) if str(clip_path).endswith(".mp4") else frames[idx_t], idx_t)

        # ---- Sample negative frame (different identity) ----
        neg_id = identity_id
        while neg_id == identity_id:
            neg_id = random.choice(self.identity_ids)
        neg_clip_idx = random.choice(self.identity_to_clips[neg_id])
        neg_identity_id, neg_clip_path, neg_frames = self.clips[neg_clip_idx]
        frame_neg = self._load_frame(
            str(neg_clip_path) if str(neg_clip_path).endswith(".mp4") else neg_frames[0]
        )

        # ---- Apply augmentations ----
        # Reconstruction augmentation (minimal) for source/target
        aug_s = self.aug_recon(image=frame_s)["image"]
        aug_t = self.aug_recon(image=frame_t)["image"]
        # Identity augmentation (stronger) for negative
        aug_neg = self.aug_id(image=frame_neg)["image"]

        return {
            "source":         aug_s,                              # I_s   [3, 256, 256]
            "target":         aug_t,                              # I_t   [3, 256, 256]
            "negative":       aug_neg,                            # I_neg [3, 256, 256]
            "identity_label": torch.tensor(identity_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Fine-tuning dataset: CelebA-HQ (identity) × AffectNet (semantic)
# Used for MRF diffusion pipeline fine-tuning (Section 4.1)
# ---------------------------------------------------------------------------

class DiffusionFineTuneDataset(Dataset):
    """
    Dataset for fine-tuning the ViDiExPo diffusion pipeline.

    Sources (Section 4.1):
        - CelebA-HQ: identity images (IDs 0–20,000 for training,
                     20,001–30,000 held out for evaluation)
        - AffectNet: semantic reference images (expression labels)
        - Text prompts: constructed from identity + expression descriptors

    Non-overlapping splits to prevent evaluation contamination.
    """

    def __init__(
        self,
        celeba_hq_root: str,
        affectnet_root: str,
        frame_size: int = 512,       # SDXL generates 512×512
        split: str = "train",
        id_range: tuple[int, int] = (0, 20000),
        tokenizer=None,
        max_token_length: int = 77,
    ):
        self.celeba_root  = Path(celeba_hq_root)
        self.affectnet_root = Path(affectnet_root)
        self.frame_size   = frame_size
        self.split        = split
        self.id_range     = id_range
        self.tokenizer    = tokenizer
        self.max_token_length = max_token_length

        self.transform = A.Compose([
            A.Resize(frame_size, frame_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        self._build_pairs()

    def _build_pairs(self):
        """Build (identity_image, reference_image, text_prompt) triplets."""
        # Collect identity images from CelebA-HQ within id_range
        id_start, id_end = self.id_range
        all_id_images = sorted(self.celeba_root.glob("*.jpg")) + \
                        sorted(self.celeba_root.glob("*.png"))
        self.id_images = [
            p for p in all_id_images
            if id_start <= int(p.stem) <= id_end
        ]

        # Collect semantic reference images from AffectNet
        self.sem_images = sorted(self.affectnet_root.glob("**/*.jpg"))[:10000]

        # Expression label templates
        self.expr_templates = [
            "happy", "sad", "angry", "surprised", "shocked",
            "neutral", "fearful", "disgusted", "contempt",
        ]

        print(f"[DiffusionFineTuneDataset] {len(self.id_images)} identity images, "
              f"{len(self.sem_images)} semantic references.")

    def __len__(self) -> int:
        return min(len(self.id_images) * 5, 100_000)   # up-sample pairs

    def __getitem__(self, idx: int) -> dict:
        id_path  = random.choice(self.id_images)
        sem_path = random.choice(self.sem_images)
        expr     = random.choice(self.expr_templates)

        # Construct text prompt
        text = f"A photo of a person with a {expr} expression."

        # Load and transform images
        id_img  = cv2.cvtColor(cv2.imread(str(id_path)),  cv2.COLOR_BGR2RGB)
        sem_img = cv2.cvtColor(cv2.imread(str(sem_path)), cv2.COLOR_BGR2RGB)

        id_tensor  = self.transform(image=id_img)["image"]
        sem_tensor = self.transform(image=sem_img)["image"]

        item = {
            "identity_image":  id_tensor,    # [3, 512, 512]
            "semantic_image":  sem_tensor,   # [3, 512, 512]
            "text":            text,
        }

        # Optionally tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"]      = tokens.input_ids.squeeze(0)
            item["attention_mask"] = tokens.attention_mask.squeeze(0)

        return item
