"""
inference_unet_landmarks.py
---------------------------
Streamlit/API-friendly inference helpers for the from-scratch 9-landmark U-Net.

Use this with weights produced by train_unet_landmarks.py, e.g.
best_unet_landmarks.pth.

Required files in the same folder:
    - unet_landmarks.py
    - inference_unet_landmarks.py
    - app_unet_landmarks.py
    - best_unet_landmarks.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from unet_landmarks import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNetLandmarks,
    heatmaps_to_points,
)


def load_landmark_model(weights_path: str | Path, device: str | None = None):
    """Load the from-scratch U-Net landmark checkpoint."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(weights_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected checkpoint from train_unet_landmarks.py containing "
            "'model_state_dict'. This is not the old SMP junction-detector format."
        )

    num_landmarks = int(checkpoint.get("num_landmarks", 9))
    image_size = int(checkpoint.get("image_size", 512))
    base_ch = int(checkpoint.get("base_ch", 32))
    dropout = float(checkpoint.get("dropout", 0.05))

    model = UNetLandmarks(
        num_landmarks=num_landmarks,
        base_ch=base_ch,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "device": device,
        "num_landmarks": num_landmarks,
        "image_size": image_size,
        "base_ch": base_ch,
        "dropout": dropout,
        "best_pck": checkpoint.get("best_pck"),
        "best_mean_error": checkpoint.get("best_mean_error"),
    }
    return model, meta


def preprocess_bgr(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """BGR image -> normalized model tensor [1,3,H,W]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.no_grad()
def detect_landmarks(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    meta: dict,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict exactly one point per heatmap channel.

    Returns
    -------
    points_xy:
        [num_landmarks, 2] in original image coordinates, top-left origin.
    confs:
        [num_landmarks] max probability for each landmark channel.
    heatmap_fullres:
        [H,W] max heatmap across landmark channels, resized to original image size.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    original_h, original_w = image_bgr.shape[:2]
    image_size = int(meta.get("image_size", 512))
    device = next(model.parameters()).device

    x = preprocess_bgr(image_bgr, image_size).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)  # [C,H,W]

    points_model, confs = heatmaps_to_points(probs, conf_threshold=conf_threshold)

    points_xy = points_model.copy().astype(np.float32)
    points_xy[:, 0] *= original_w / float(image_size)
    points_xy[:, 1] *= original_h / float(image_size)

    max_hm = probs.max(axis=0)
    heatmap_fullres = cv2.resize(
        max_hm,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_fullres = np.clip(heatmap_fullres, 0.0, 1.0)

    return points_xy, confs.astype(np.float32), heatmap_fullres


def create_landmark_overlay(
    image_bgr: np.ndarray,
    points_xy: np.ndarray,
    confs: np.ndarray | None = None,
    heatmap: np.ndarray | None = None,
    alpha: float = 0.25,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw numbered 9-landmark overlay."""
    overlay = image_bgr.copy()

    if heatmap is not None:
        hm = np.clip(heatmap, 0.0, 1.0)
        hm_uint8 = (hm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 1.0 - alpha, heatmap_color, alpha, 0)

    for i, point in enumerate(points_xy, start=1):
        x, y = point
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))

        cv2.circle(overlay, (x_i, y_i), 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x_i, y_i), 6, (0, 0, 255), -1, cv2.LINE_AA)

        if show_labels:
            label = str(i)
            cv2.putText(
                overlay,
                label,
                (x_i + 8, y_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (x_i + 8, y_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return overlay
