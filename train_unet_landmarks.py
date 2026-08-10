from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from unet_landmarks import (
    Sample,
    UNetLandmarks,
    WingLandmarkDataset,
    evaluate_landmarks,
    heatmap_weighted_mse_loss,
    heatmaps_to_points,
    load_samples,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_samples(
    samples: List[Sample],
    val_split: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    val_n = max(1, int(round(n * val_split))) if n > 1 else 0

    val_samples = samples[:val_n]
    train_samples = samples[val_n:]

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("Need at least 2 labelled images for train/validation split.")

    return train_samples, val_samples


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    image_size: int,
    tolerance_px: float,
    pos_weight: float,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_pck = 0.0
    total_mean_error = 0.0
    total_images = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["heatmaps"].to(device)
        visible = batch["visible"].to(device)

        gt_points = batch["points"].cpu().numpy()
        visible_np = batch["visible"].cpu().numpy()

        logits = model(images)

        loss = heatmap_weighted_mse_loss(
            logits,
            targets,
            visible,
            pos_weight=pos_weight,
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        total_loss += float(loss.item())

        for i in range(probs.shape[0]):
            pred_points, _ = heatmaps_to_points(
                probs[i],
                conf_threshold=0.0,
            )

            metrics = evaluate_landmarks(
                pred_points=pred_points,
                gt_points=gt_points[i],
                visible=visible_np[i],
                tolerance_px=tolerance_px,
            )

            total_pck += metrics["pck"]

            if np.isfinite(metrics["mean_error"]):
                total_mean_error += metrics["mean_error"]
            else:
                total_mean_error += image_size

            total_images += 1

    return {
        "loss": total_loss / max(len(loader), 1),
        "pck": total_pck / max(total_images, 1),
        "mean_error": total_mean_error / max(total_images, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train from-scratch U-Net for 9 bee-wing landmarks. "
            "This version does NOT flip y coordinates. Use it only when your "
            "CSV coordinates are already in image top-left coordinates, or when "
            "your unet_landmarks.py/load_samples() already applies the y-flip."
        )
    )

    parser.add_argument("--image-dir", required=True, type=str)
    parser.add_argument("--label-dir", default=None, type=str)
    parser.add_argument("--master-csv", default=None, type=str)
    parser.add_argument("--save-path", required=True, type=str)

    parser.add_argument("--num-landmarks", default=9, type=int)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--sigma", default=6.0, type=float)

    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    parser.add_argument("--base-ch", default=32, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)

    parser.add_argument("--val-split", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--pos-weight", default=30.0, type=float)
    parser.add_argument("--tolerance-px", default=20.0, type=float)

    args = parser.parse_args()

    if args.label_dir is None and args.master_csv is None:
        raise ValueError("Pass either --label-dir or --master-csv.")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    samples = load_samples(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        master_csv=args.master_csv,
        num_landmarks=args.num_landmarks,
    )

    print("Loaded samples:", len(samples))
    print("Y-coordinate handling: NO y-flip inside train_unet_landmarks.py")
    print("Use this only if labels are already correct in image coordinates.")

    train_samples, val_samples = split_samples(
        samples,
        val_split=args.val_split,
        seed=args.seed,
    )

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    train_ds = WingLandmarkDataset(
        train_samples,
        image_size=args.image_size,
        num_landmarks=args.num_landmarks,
        sigma=args.sigma,
        train=True,
    )

    val_ds = WingLandmarkDataset(
        val_samples,
        image_size=args.image_size,
        num_landmarks=args.num_landmarks,
        sigma=args.sigma,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = UNetLandmarks(
        num_landmarks=args.num_landmarks,
        base_ch=args.base_ch,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_pck = -1.0
    best_mean_error = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)
            visible = batch["visible"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = heatmap_weighted_mse_loss(
                logits,
                targets,
                visible,
                pos_weight=args.pos_weight,
            )

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            image_size=args.image_size,
            tolerance_px=args.tolerance_px,
            pos_weight=args.pos_weight,
        )

        val_pck = val_metrics["pck"]
        val_mean_error = val_metrics["mean_error"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"PCK={val_pck:.3f} "
            f"mean_err={val_mean_error:.2f}px"
        )

        is_best = False
        if val_pck > best_pck:
            is_best = True
        elif val_pck == best_pck and val_mean_error < best_mean_error:
            is_best = True

        if is_best:
            best_pck = val_pck
            best_mean_error = val_mean_error

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_landmarks": args.num_landmarks,
                "image_size": args.image_size,
                "sigma": args.sigma,
                "base_ch": args.base_ch,
                "dropout": args.dropout,
                "best_pck": best_pck,
                "best_mean_error": best_mean_error,
                "epoch": epoch,
                "csv_origin": "already_top_left_or_handled_elsewhere",
            }

            torch.save(checkpoint, save_path)

            print(
                f"Saved best model: {save_path} "
                f"PCK={best_pck:.3f} "
                f"mean_err={best_mean_error:.2f}px"
            )

    print("Done.")
    print(f"Best PCK={best_pck:.3f}, best mean error={best_mean_error:.2f}px")
    print("Best weights:", save_path)


if __name__ == "__main__":
    main()
