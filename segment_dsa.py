"""
Segment DSA Images with SAM or MedSAM
=======================================
Paper 2: Segmentation of cerebral DSA images in IIH and venous sinus
         stenosis using SAM and MedSAM, SPIE 2024.

Usage
-----
# SAM with bounding box:
python segment_dsa.py \
  --input_dir  ./data/dsa/patient_001 \
  --checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --model_type sam \
  --output_dir ./results/patient_001 \
  --bbox 50 60 200 220

# MedSAM with bounding box:
python segment_dsa.py \
  --input_dir  ./data/dsa/patient_001 \
  --checkpoint ./checkpoints/medsam_vit_b.pth \
  --model_type medsam \
  --output_dir ./results/patient_001 \
  --bbox 50 60 200 220
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.sam_segmenter   import SAMSegmenter
from src.preprocessing   import load_image_dir, normalize_window
from src.postprocessing  import refine_mask


def parse_args():
    p = argparse.ArgumentParser(
        description="SAM / MedSAM segmentation of DSA images")
    p.add_argument("--input_dir",   type=str, required=True,
                   help="Directory of input DSA images (PNG/JPG)")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to SAM or MedSAM checkpoint")
    p.add_argument("--model_type",  type=str, default="sam",
                   choices=["sam", "medsam"],
                   help="'sam' uses ViT-H, 'medsam' uses ViT-B")
    p.add_argument("--output_dir",  type=str, default="./results")
    p.add_argument("--bbox",        type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Bounding box prompt [x0 y0 x1 y1]")
    p.add_argument("--point",       type=int, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="Single foreground point prompt")
    p.add_argument("--no_refine",   action="store_true")
    p.add_argument("--device",      type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAM model type mapping
    model_type_map = {"sam": "vit_h", "medsam": "vit_b"}
    vit_type = model_type_map.get(args.model_type, "vit_h")

    # Load model
    segmenter = SAMSegmenter(
        checkpoint=args.checkpoint,
        model_type=vit_type,
        device=args.device,
    )

    # Load images
    images, paths = load_image_dir(args.input_dir, grayscale=True)
    print(f"Found {len(images)} DSA images in {args.input_dir}")

    for img_np, img_path in zip(images, paths):
        name = Path(img_path).stem
        print(f"  Segmenting: {name}")

        # Choose inference mode
        if args.bbox is not None:
            mask = segmenter.predict_bbox(img_np, args.bbox)
        elif args.point is not None:
            mask = segmenter.predict_points(img_np, [args.point])
        else:
            # Auto mode: pick largest mask
            results = segmenter.predict_auto(img_np)
            if not results:
                print(f"    No mask found for {name}, skipping.")
                continue
            results.sort(key=lambda r: r["area"], reverse=True)
            mask = results[0]["segmentation"]

        if not args.no_refine:
            mask = refine_mask(mask)

        # Save mask as PNG
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(str(out_dir / f"{name}_mask.png"))

        # Save overlay
        overlay = np.stack([img_np] * 3, axis=-1).astype(np.uint8)
        overlay[mask, 0] = 255     # red overlay on segmented region
        overlay[mask, 1] = 0
        overlay[mask, 2] = 0
        Image.fromarray(overlay).save(str(out_dir / f"{name}_overlay.png"))

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
