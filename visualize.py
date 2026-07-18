"""
Visualise Segmentation Results
================================
Overlay predicted masks on images and save as PNG or animated GIF.

Usage
-----
python visualize.py \
  --image ./data/dsa/patient_001/frame_001.png \
  --mask  ./results/patient_001/frame_001_mask.png \
  --output ./results/overlay.png

# Animated GIF for 3D MRI slices:
python visualize.py \
  --nifti ./data/4dflow/subject_001/magnitude.nii.gz \
  --mask_nifti ./results/subject_001_mask.nii.gz \
  --output ./results/subject_001.gif
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Visualise segmentation results")
    # 2D image mode
    p.add_argument("--image",       type=str, default="")
    p.add_argument("--mask",        type=str, default="")
    # 3D NIfTI GIF mode
    p.add_argument("--nifti",       type=str, default="")
    p.add_argument("--mask_nifti",  type=str, default="")
    p.add_argument("--axis",        type=int, default=2,
                   help="Axis along which to extract slices for the GIF")
    p.add_argument("--output",      type=str, default="./results/overlay.png")
    p.add_argument("--alpha",       type=float, default=0.4,
                   help="Mask overlay transparency")
    p.add_argument("--color",       type=str, default="red",
                   help="Mask overlay colour")
    return p.parse_args()


def make_overlay(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4, color: str = "red") -> np.ndarray:
    """Create an RGBA overlay of a mask on a grayscale image."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) /
                 (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    rgb = np.stack([image, image, image], axis=-1)
    colour_map = {"red": [255, 0, 0], "green": [0, 255, 0],
                  "blue": [0, 0, 255], "yellow": [255, 255, 0]}
    c = colour_map.get(color, [255, 0, 0])
    overlay = rgb.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * rgb[mask.astype(bool)] + alpha * np.array(c)
    ).astype(np.uint8)
    return overlay


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 2D image mode ----
    if args.image and args.mask:
        img  = np.array(Image.open(args.image).convert("L"))
        mask = (np.array(Image.open(args.mask).convert("L")) > 127)
        overlay = make_overlay(img, mask, alpha=args.alpha, color=args.color)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input Image")
        axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Predicted Mask")
        axes[2].imshow(overlay); axes[2].set_title("Overlay")
        for ax in axes: ax.axis("off")
        patch = mpatches.Patch(color=args.color, label="Segmented region")
        axes[2].legend(handles=[patch], loc="lower right", fontsize=8)
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        print(f"Saved overlay to {out_path}")
        return

    # ---- NIfTI / GIF mode ----
    if args.nifti and args.mask_nifti:
        import nibabel as nib
        vol  = np.asarray(nib.load(args.nifti).dataobj)
        mask = np.asarray(nib.load(args.mask_nifti).dataobj).astype(bool)
        n    = vol.shape[args.axis]

        frames = []
        for i in range(n):
            slc  = np.take(vol,  i, axis=args.axis)
            mslc = np.take(mask, i, axis=args.axis)
            ov   = make_overlay(slc, mslc, alpha=args.alpha, color=args.color)
            frames.append(Image.fromarray(ov))

        gif_path = str(out_path.with_suffix(".gif"))
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       loop=0, duration=150)
        print(f"Saved animated GIF ({n} frames) to {gif_path}")
        return

    print("Provide --image + --mask  OR  --nifti + --mask_nifti.")


if __name__ == "__main__":
    main()
