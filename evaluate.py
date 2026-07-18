"""
Evaluate Segmentation Against Ground Truth Masks
=================================================
Computes Dice, IoU, and HD95 for all predicted masks.

Usage
-----
# NIfTI masks (4D Flow):
python evaluate.py \
  --pred_dir ./results/subject_001 \
  --gt_dir   ./data/4dflow/subject_001 \
  --fmt nifti \
  --output_dir ./results/eval

# PNG masks (DSA):
python evaluate.py \
  --pred_dir ./results/patient_001 \
  --gt_dir   ./data/dsa/patient_001 \
  --fmt png \
  --output_dir ./results/eval
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation")
    p.add_argument("--pred_dir",    type=str, required=True)
    p.add_argument("--gt_dir",      type=str, required=True)
    p.add_argument("--fmt",         type=str, default="png",
                   choices=["png", "nifti"])
    p.add_argument("--output_dir",  type=str, default="./results/eval")
    p.add_argument("--voxel_size",  type=float, default=1.0,
                   help="Isotropic voxel size in mm (for HD95)")
    return p.parse_args()


def load_mask_png(path: str) -> np.ndarray:
    from PIL import Image
    return (np.array(Image.open(path).convert("L")) > 127).astype(bool)


def load_mask_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj).astype(bool)


def main():
    args = parse_args()
    from src.metrics import compute_all

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)

    # Discover predicted masks
    if args.fmt == "png":
        pred_files = sorted(pred_dir.glob("*_mask.png"))
        def load_pred(p): return load_mask_png(str(p))
        def find_gt(p):
            stem = p.stem.replace("_mask", "")
            return gt_dir / f"{stem}.png"
    else:
        pred_files = sorted(pred_dir.glob("*.nii.gz")) + \
                     sorted(pred_dir.glob("*.nii"))
        def load_pred(p): return load_mask_nifti(str(p))
        def find_gt(p): return gt_dir / "mask_gt.nii.gz"

    all_results = []
    dices, ious, hd95s = [], [], []

    for pf in pred_files:
        gt_path = find_gt(pf)
        if not gt_path.exists():
            print(f"  No GT found for {pf.name}, skipping.")
            continue
        pred = load_pred(pf)
        gt   = load_mask_nifti(str(gt_path)) if args.fmt == "nifti" \
               else load_mask_png(str(gt_path))
        metrics = compute_all(pred, gt, voxel_spacing=args.voxel_size)
        metrics["file"] = pf.name
        all_results.append(metrics)
        dices.append(metrics["Dice"])
        ious.append(metrics["IoU"])
        if not np.isinf(metrics["HD95_mm"]):
            hd95s.append(metrics["HD95_mm"])
        print(f"  {pf.name:<40}  "
              f"Dice={metrics['Dice']:.3f}  "
              f"IoU={metrics['IoU']:.3f}  "
              f"HD95={metrics['HD95_mm']:.1f} mm")

    summary = {
        "n_samples"   : len(all_results),
        "Dice_mean"   : float(np.mean(dices)) if dices else None,
        "Dice_std"    : float(np.std(dices))  if dices else None,
        "IoU_mean"    : float(np.mean(ious))  if ious  else None,
        "IoU_std"     : float(np.std(ious))   if ious  else None,
        "HD95_mean_mm": float(np.mean(hd95s)) if hd95s else None,
        "HD95_std_mm" : float(np.std(hd95s))  if hd95s else None,
        "per_sample"  : all_results,
    }

    out_file = out_dir / "metrics.json"
    with open(str(out_file), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"  Dice : {summary['Dice_mean']:.3f} ± {summary['Dice_std']:.3f}")
    print(f"  IoU  : {summary['IoU_mean']:.3f} ± {summary['IoU_std']:.3f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
