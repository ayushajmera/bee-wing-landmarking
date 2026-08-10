"""
unet_landmarks.py
------------------
Small from-scratch U-Net and helper utilities for 9 landmark detection.

This file intentionally does NOT use segmentation_models_pytorch.
It uses plain PyTorch modules only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# U-Net from scratch
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad/crop safety for odd input sizes.
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dx != 0 or dy != 0:
            x = F.pad(
                x,
                [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetLandmarks(nn.Module):
    """
    U-Net for landmark heatmap regression.

    Input:  [B, 3, H, W]
    Output: [B, num_landmarks, H, W] logits

    Apply torch.sigmoid(output) to get heatmap probabilities.
    """

    def __init__(self, num_landmarks: int = 9, base_ch: int = 32, dropout: float = 0.05):
        super().__init__()
        self.num_landmarks = int(num_landmarks)

        self.inc = DoubleConv(3, base_ch, dropout=0.0)
        self.down1 = Down(base_ch, base_ch * 2, dropout=dropout)
        self.down2 = Down(base_ch * 2, base_ch * 4, dropout=dropout)
        self.down3 = Down(base_ch * 4, base_ch * 8, dropout=dropout)
        self.down4 = Down(base_ch * 8, base_ch * 16, dropout=dropout)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8, dropout=dropout)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4, dropout=dropout)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2, dropout=dropout)
        self.up4 = Up(base_ch * 2, base_ch, base_ch, dropout=0.0)

        self.outc = nn.Conv2d(base_ch, self.num_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# -----------------------------------------------------------------------------
# Annotation parsing
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    image_path: Path
    points: np.ndarray  # [num_landmarks, 2], x/y in original image coordinates; NaN allowed


def find_images(image_dir: Path) -> Dict[str, Path]:
    image_dir = Path(image_dir)
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    mapping: Dict[str, Path] = {}
    for p in sorted(paths):
        mapping[p.name] = p
        mapping[p.stem] = p
    return mapping


def _standardize_points(df: pd.DataFrame, num_landmarks: int) -> np.ndarray:
    """
    Read x/y columns from a dataframe. If a landmark/id column exists, sort by it.
    Otherwise, use row order as landmark 1..N.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    if "x" not in cols_lower or "y" not in cols_lower:
        raise ValueError("Annotation CSV must contain x and y columns.")

    # Sort by landmark id if present; otherwise row order is used.
    id_col = None
    for cand in ["landmark", "landmark_id", "id", "point", "point_id", "index"]:
        if cand in cols_lower:
            id_col = cols_lower[cand]
            break

    d = df.copy()
    if id_col is not None:
        d = d.sort_values(id_col)

    arr = np.full((num_landmarks, 2), np.nan, dtype=np.float32)
    xy = d[[cols_lower["x"], cols_lower["y"]]].astype(float).to_numpy(dtype=np.float32)
    n = min(num_landmarks, len(xy))
    arr[:n] = xy[:n]
    return arr


def load_samples(
    image_dir: str | Path,
    num_landmarks: int = 9,
    label_dir: str | Path | None = None,
    master_csv: str | Path | None = None,
) -> List[Sample]:
    """
    Supported annotation layouts:

    A) Per-image CSV files:
       image_dir/LACM_001.jpg
       label_dir/LACM_001.csv
       CSV columns: x,y  with exactly 9 rows in landmark order.

    B) Master long CSV:
       columns: image,x,y  and optionally landmark/id.
       If no landmark/id exists, row order within each image is landmark order.

    C) Master wide CSV:
       columns: image,x1,y1,x2,y2,...,x9,y9
    """
    image_map = find_images(Path(image_dir))
    samples: List[Sample] = []

    if label_dir is None and master_csv is None:
        raise ValueError("Pass either label_dir or master_csv.")

    if label_dir is not None:
        label_dir = Path(label_dir)
        for csv_path in sorted(label_dir.glob("*.csv")):
            img = image_map.get(csv_path.stem)
            if img is None:
                print(f"Warning: no matching image found for {csv_path.name}; skipped")
                continue
            df = pd.read_csv(csv_path)
            pts = _standardize_points(df, num_landmarks=num_landmarks)
            samples.append(Sample(img, pts))

    if master_csv is not None:
        master_csv = Path(master_csv)
        df = pd.read_csv(master_csv)
        cols_lower = {c.lower().strip(): c for c in df.columns}

        image_col = None
        for cand in ["image", "filename", "file", "image_name", "name", "img", "path", "image_id"]:
            if cand in cols_lower:
                image_col = cols_lower[cand]
                break
        if image_col is None:
            raise ValueError(
                "Master CSV needs an image column such as image, filename, image_name, or path."
            )

        # Wide format: image,x1,y1,...,x9,y9
        wide_ok = all(f"x{i}" in cols_lower and f"y{i}" in cols_lower for i in range(1, num_landmarks + 1))
        if wide_ok:
            for _, row in df.iterrows():
                key = str(row[image_col])
                key_stem = Path(key).stem
                img = image_map.get(Path(key).name) or image_map.get(key_stem)
                if img is None:
                    print(f"Warning: no matching image found for {key}; skipped")
                    continue
                pts = np.full((num_landmarks, 2), np.nan, dtype=np.float32)
                for i in range(1, num_landmarks + 1):
                    pts[i - 1, 0] = float(row[cols_lower[f"x{i}"]])
                    pts[i - 1, 1] = float(row[cols_lower[f"y{i}"]])
                samples.append(Sample(img, pts))
        else:
            # Long format: image,x,y[,landmark]
            if "x" not in cols_lower or "y" not in cols_lower:
                raise ValueError("Long master CSV must contain columns image,x,y.")
            # Preserve original row order for no-ID CSVs.
            df = df.copy()
            df["__row_order__"] = np.arange(len(df))
            for key, group in df.groupby(image_col, sort=False):
                key_str = str(key)
                img = image_map.get(Path(key_str).name) or image_map.get(Path(key_str).stem)
                if img is None:
                    print(f"Warning: no matching image found for {key_str}; skipped")
                    continue
                pts = _standardize_points(group.sort_values("__row_order__"), num_landmarks=num_landmarks)
                samples.append(Sample(img, pts))

    # Remove accidental duplicates, preferring first occurrence.
    seen = set()
    unique: List[Sample] = []
    for s in samples:
        if s.image_path in seen:
            continue
        seen.add(s.image_path)
        unique.append(s)

    if not unique:
        raise RuntimeError("No image/annotation pairs found. Check paths and file names.")

    return unique


# -----------------------------------------------------------------------------
# Heatmaps and preprocessing
# -----------------------------------------------------------------------------

def make_gaussian_heatmaps(
    points_xy: np.ndarray,
    image_h: int,
    image_w: int,
    sigma: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert N landmark coordinates into N separate heatmap channels.

    Returns:
        heatmaps: [N, H, W]
        visible:  [N] 1 if coordinate is valid, else 0
    """
    points_xy = np.asarray(points_xy, dtype=np.float32)
    n = points_xy.shape[0]
    heatmaps = np.zeros((n, image_h, image_w), dtype=np.float32)
    visible = np.zeros((n,), dtype=np.float32)

    radius = int(round(3 * sigma))
    size = radius * 2 + 1
    ax = np.arange(size, dtype=np.float32) - radius
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    kernel /= max(float(kernel.max()), 1e-8)

    for i, (x, y) in enumerate(points_xy):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        if x_i < 0 or x_i >= image_w or y_i < 0 or y_i >= image_h:
            continue
        visible[i] = 1.0

        x1 = max(0, x_i - radius)
        x2 = min(image_w, x_i + radius + 1)
        y1 = max(0, y_i - radius)
        y2 = min(image_h, y_i + radius + 1)

        kx1 = radius - (x_i - x1)
        kx2 = kx1 + (x2 - x1)
        ky1 = radius - (y_i - y1)
        ky2 = ky1 + (y2 - y1)

        heatmaps[i, y1:y2, x1:x2] = np.maximum(
            heatmaps[i, y1:y2, x1:x2],
            kernel[ky1:ky2, kx1:kx2],
        )

    return heatmaps, visible


def resize_image_and_points(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    old_h, old_w = image_rgb.shape[:2]
    image_resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    pts = points_xy.astype(np.float32).copy()
    pts[:, 0] *= image_size / float(old_w)
    pts[:, 1] *= image_size / float(old_h)
    return image_resized, pts


def apply_random_affine(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    max_rotate_deg: float = 5.0,
    max_shift_frac: float = 0.03,
    max_scale_delta: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    angle = np.random.uniform(-max_rotate_deg, max_rotate_deg)
    scale = 1.0 + np.random.uniform(-max_scale_delta, max_scale_delta)
    tx = np.random.uniform(-max_shift_frac, max_shift_frac) * w
    ty = np.random.uniform(-max_shift_frac, max_shift_frac) * h

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale).astype(np.float32)
    M[0, 2] += tx
    M[1, 2] += ty

    image_out = cv2.warpAffine(
        image_rgb,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    pts = points_xy.copy().astype(np.float32)
    valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if valid.any():
        homo = np.concatenate([pts[valid], np.ones((valid.sum(), 1), dtype=np.float32)], axis=1)
        pts[valid] = homo @ M.T
    return image_out, pts


def apply_photometric_aug(image_rgb: np.ndarray) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    # Brightness and contrast.
    alpha = 1.0 + np.random.uniform(-0.15, 0.15)
    beta = np.random.uniform(-15, 15)
    img = img * alpha + beta
    # Low Gaussian sensor noise.
    if np.random.rand() < 0.3:
        img += np.random.normal(0, 4.0, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_tensor_normalized(image_rgb: np.ndarray) -> torch.Tensor:
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


class WingLandmarkDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 512,
        num_landmarks: int = 9,
        sigma: float = 6.0,
        train: bool = True,
        affine_prob: float = 0.7,
        photometric_prob: float = 0.7,
    ):
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.num_landmarks = int(num_landmarks)
        self.sigma = float(sigma)
        self.train = bool(train)
        self.affine_prob = float(affine_prob)
        self.photometric_prob = float(photometric_prob)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {sample.image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image, pts = resize_image_and_points(image, sample.points, self.image_size)

        if self.train:
            if np.random.rand() < self.affine_prob:
                image, pts = apply_random_affine(image, pts)
            if np.random.rand() < self.photometric_prob:
                image = apply_photometric_aug(image)

        heatmaps, visible = make_gaussian_heatmaps(
            pts,
            image_h=self.image_size,
            image_w=self.image_size,
            sigma=self.sigma,
        )

        return {
            "image": to_tensor_normalized(image),
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "visible": torch.from_numpy(visible).float(),
            "points": torch.from_numpy(pts).float(),
            "name": sample.image_path.name,
        }


# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

def heatmap_weighted_mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visible: torch.Tensor,
    pos_weight: float = 30.0,
) -> torch.Tensor:
    """
    Weighted MSE on sigmoid heatmaps.
    visible masks missing landmark channels, if any.
    """
    pred = torch.sigmoid(logits)
    weights = 1.0 + (float(pos_weight) - 1.0) * targets
    se = weights * (pred - targets) ** 2

    # visible: [B, C] -> [B, C, 1, 1]
    mask = visible[:, :, None, None].to(se.dtype)
    se = se * mask
    denom = mask.sum() * targets.shape[-1] * targets.shape[-2]
    return se.sum() / denom.clamp_min(1.0)


def heatmaps_to_points(
    heatmaps: np.ndarray,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Landmark-specific decoding: one argmax per channel.

    Args:
        heatmaps: [C, H, W] probabilities after sigmoid.

    Returns:
        points: [C, 2] x/y coordinates in heatmap space. NaN if below threshold.
        confs:  [C] max probability per channel.
    """
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [C,H,W] heatmaps, got {heatmaps.shape}")

    c, h, w = heatmaps.shape
    points = np.full((c, 2), np.nan, dtype=np.float32)
    confs = np.zeros((c,), dtype=np.float32)
    for i in range(c):
        hm = heatmaps[i]
        flat_idx = int(np.argmax(hm))
        y, x = divmod(flat_idx, w)
        conf = float(hm[y, x])
        confs[i] = conf
        if conf >= conf_threshold:
            points[i] = [x, y]
    return points, confs


def evaluate_landmarks(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    visible: np.ndarray,
    tolerance_px: float = 20.0,
) -> Dict[str, float]:
    valid = (visible > 0.5) & np.isfinite(gt_points[:, 0]) & np.isfinite(gt_points[:, 1])
    if not valid.any():
        return {"pck": 0.0, "mean_error": float("nan"), "n": 0.0}

    pred = pred_points[valid]
    gt = gt_points[valid]
    ok_pred = np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])

    errors = np.full((valid.sum(),), np.inf, dtype=np.float32)
    errors[ok_pred] = np.linalg.norm(pred[ok_pred] - gt[ok_pred], axis=1)
    pck = float(np.mean(errors <= tolerance_px))
    finite_errors = errors[np.isfinite(errors)]
    mean_error = float(finite_errors.mean()) if len(finite_errors) else float("inf")
    return {"pck": pck, "mean_error": mean_error, "n": float(valid.sum())}
