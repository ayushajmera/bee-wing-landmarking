%%writefile /content/unet_landmark_project/predict_unet_landmarks.py
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from inference_unet_landmarks import (
    load_landmark_model,
    detect_landmarks,
    create_landmark_overlay,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))

    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    return ((image - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)


def predict_one_image(
    image_path: Path,
    weights_path: Path,
    out_dir: Path,
    conf_threshold: float = 0.0,
    show_heatmap: bool = True,
    show_labels: bool = True,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]

    model, meta = load_landmark_model(weights_path)

    points, confs, heatmap = detect_landmarks(
        image_bgr=image_bgr,
        model=model,
        meta=meta,
        conf_threshold=conf_threshold,
    )

    overlay = create_landmark_overlay(
        image_bgr=image_bgr,
        points_xy=points,
        confs=confs,
        heatmap=heatmap if show_heatmap else None,
        show_labels=show_labels,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, ((x, y), conf) in enumerate(zip(points, confs), start=1):
        row = {
            "image": image_path.name,
            "landmark": idx,
            "x": float(x) if np.isfinite(x) else np.nan,
            "y_image_top_left": float(y) if np.isfinite(y) else np.nan,
            "y_bottom_left": float(h - y) if np.isfinite(y) else np.nan,
            "confidence": float(conf),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = out_dir / f"{image_path.stem}_unet_landmarks.csv"
    out_png = out_dir / f"{image_path.stem}_unet_landmarks_overlay.png"
    out_heatmap = out_dir / f"{image_path.stem}_unet_heatmap.png"

    df.to_csv(out_csv, index=False)
    cv2.imwrite(str(out_png), overlay)
    cv2.imwrite(str(out_heatmap), normalize_to_uint8(heatmap))

    print(f"Image: {image_path.name}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved overlay: {out_png}")
    print(f"Saved heatmap: {out_heatmap}")
    print(f"Landmarks predicted: {len(df)}")

    return out_csv, out_png, out_heatmap


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
        help="Path to one wing image.",
    )

    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to best_unet_landmarks.pth.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory for CSV and overlay.",
    )

    parser.add_argument(
        "--conf-threshold",
        default=0.0,
        type=float,
        help="Keep 0.0 to always output all 9 landmarks.",
    )

    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Do not blend heatmap into overlay.",
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw landmark numbers.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    predict_one_image(
        image_path=image_path,
        weights_path=weights_path,
        out_dir=out_dir,
        conf_threshold=args.conf_threshold,
        show_heatmap=not args.no_heatmap,
        show_labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()