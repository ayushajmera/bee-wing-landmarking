%%writefile /content/unet_landmark_project/check_annotations.py
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_matching_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

        p_upper = image_dir / f"{stem}{ext.upper()}"
        if p_upper.exists():
            return p_upper

    # fallback recursive search
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
            return p

    return None


def read_points(csv_path: Path, num_landmarks: int = 9) -> np.ndarray:
    df = pd.read_csv(csv_path)

    cols_lower = {c.lower().strip(): c for c in df.columns}

    if "x" not in cols_lower or "y" not in cols_lower:
        raise ValueError(f"{csv_path} must contain x and y columns")

    # If an ID/landmark column exists, sort by it.
    id_col = None
    for cand in ["landmark", "landmark_id", "id", "point", "point_id", "index"]:
        if cand in cols_lower:
            id_col = cols_lower[cand]
            break

    if id_col is not None:
        df = df.sort_values(id_col)

    pts = df[[cols_lower["x"], cols_lower["y"]]].astype(float).to_numpy(dtype=np.float32)

    if len(pts) != num_landmarks:
        print(f"WARNING: {csv_path.name} has {len(pts)} points, expected {num_landmarks}")

    return pts[:num_landmarks]


def draw_overlay(
    image_path: Path,
    csv_path: Path,
    out_path: Path,
    csv_origin: str = "bottom-left",
    num_landmarks: int = 9,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]

    pts = read_points(csv_path, num_landmarks=num_landmarks)

    # IMPORTANT:
    # Madeleine's CSV coordinates use bottom-left origin.
    # OpenCV image coordinates use top-left origin.
    if csv_origin == "bottom-left":
        pts[:, 1] = h - pts[:, 1]

    overlay = image_bgr.copy()

    for i, (x, y) in enumerate(pts, start=1):
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        x_i = int(round(float(x)))
        y_i = int(round(float(y)))

        cv2.circle(overlay, (x_i, y_i), 10, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x_i, y_i), 6, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(
            overlay,
            str(i),
            (x_i + 10, y_i - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            str(i),
            (x_i + 10, y_i - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", required=True, type=str)
    parser.add_argument("--label-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)

    parser.add_argument("--max-images", default=20, type=int)
    parser.add_argument("--num-landmarks", default=9, type=int)

    parser.add_argument(
        "--csv-origin",
        choices=["bottom-left", "top-left"],
        default="bottom-left",
        help="Use bottom-left for Madeleine's CSVs.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    csv_files = sorted(label_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in {label_dir}")

    count = 0

    for csv_path in csv_files:
        image_path = find_matching_image(image_dir, csv_path.stem)

        if image_path is None:
            print(f"WARNING: no matching image found for {csv_path.name}")
            continue

        out_path = out_dir / f"{csv_path.stem}_annotation_check.png"

        draw_overlay(
            image_path=image_path,
            csv_path=csv_path,
            out_path=out_path,
            csv_origin=args.csv_origin,
            num_landmarks=args.num_landmarks,
        )

        print(f"Saved: {out_path}")

        count += 1

        if count >= args.max_images:
            break

    print(f"Done. Generated {count} annotation check images.")


if __name__ == "__main__":
    main()