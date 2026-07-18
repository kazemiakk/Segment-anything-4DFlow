"""
Segment 4D Flow MRI Magnitude Volume with SAM 2
================================================
Paper 1: Semi-automated segmentation of magnitude images in 4D flow MRI
         using Segment Anything Model 2 (SAM 2), SPIE 2025.

Usage
-----
python segment_4dflow.py \
  --input    ./data/4dflow/subject_001/magnitude.nii.gz \
  --checkpoint ./checkpoints/sam2_hiera_large.pt \
  --model_cfg  sam2_hiera_l \
  --output   ./results/subject_001_mask.nii.gz \
  --prompt_slice 10 \
  --prompt_point 128 128
"""

import argparse
from pathlib import Path

import numpy as np

from src.sam2_segmenter  import SAM2VolumeSegmenter
from src.preprocessing   import load_nifti, save_nifti
from src.postprocessing  import refine_mask


def parse_args():
    p = argparse.ArgumentParser(
        description="SAM 2 segmentation of 4D Flow MRI magnitude volumes")
    p.add_argument("--input",        type=str, required=True,
                   help="Path to input magnitude NIfTI (.nii/.nii.gz)")
    p.add_argument("--checkpoint",   type=str,
                   default="./checkpoints/sam2_hiera_large.pt")
    p.add_argument("--model_cfg",    type=str, default="sam2_hiera_l",
                   help="SAM 2 model config name")
    p.add_argument("--output",       type=str, default="",
                   help="Output mask NIfTI path")
    p.add_argument("--prompt_slice", type=int, default=5,
                   help="Z-slice index for the segmentation prompt")
    p.add_argument("--prompt_point", type=int, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="Point prompt coordinates (x y) on the reference slice")
    p.add_argument("--prompt_bbox",  type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Bounding box prompt on the reference slice")
    p.add_argument("--time_phase",   type=int, default=0,
                   help="Cardiac phase to use for segmentation (for 4D input)")
    p.add_argument("--device",       type=str, default="")
    p.add_argument("--no_refine",    action="store_true",
                   help="Skip post-processing mask refinement")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load volume ----
    print(f"Loading: {args.input}")
    volume, nib_img = load_nifti(args.input)
    print(f"  Volume shape: {volume.shape}")

    # ---- Build segmenter ----
    segmenter = SAM2VolumeSegmenter(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )

    # ---- Build prompts ----
    prompt_points = [args.prompt_point] if args.prompt_point else None
    prompt_bbox   = args.prompt_bbox

    if prompt_points is None and prompt_bbox is None:
        # Default: centre-point heuristic
        H, W = volume.shape[:2]
        prompt_points = [[W // 2, H // 2]]
        print(f"  Using centre-point prompt: {prompt_points[0]}")

    # ---- Segment ----
    if volume.ndim == 4:
        mask = segmenter.segment_4dflow(
            volume,
            prompt_slice_idx=args.prompt_slice,
            prompt_points=prompt_points,
            prompt_bbox=prompt_bbox,
            time_phase=args.time_phase,
        )
    else:
        mask = segmenter.segment_volume(
            volume,
            prompt_slice_idx=args.prompt_slice,
            prompt_points=prompt_points,
            prompt_bbox=prompt_bbox,
        )

    print(f"  Mask shape: {mask.shape}  Voxels segmented: {mask.sum()}")

    # ---- Refine ----
    if not args.no_refine:
        mask = refine_mask(mask)

    # ---- Save ----
    out_path = args.output or str(
        Path(args.input).parent / "predicted_mask.nii.gz")
    save_nifti(mask, nib_img, out_path)


if __name__ == "__main__":
    main()
