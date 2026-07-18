"""
SAM 2 Wrapper for 3D/4D MRI Volume Segmentation
=================================================
Uses SAM 2's video predictor to propagate a segmentation mask
through the depth (Z) or time (T) dimension of a medical imaging volume.

Workflow (Paper 1 — SAM 2 for 4D Flow MRI):
  1. Load a 4D magnitude volume (X, Y, Z, T) or 3D (X, Y, Z).
  2. Extract 2D slices along the propagation axis (Z or T).
  3. User provides a point/bbox prompt on a single reference slice.
  4. SAM 2 propagates the mask to all other slices.
  5. Output is a full 3D or 4D binary mask volume.

Requirements
------------
Install SAM 2:
  pip install git+https://github.com/facebookresearch/sam2.git

Download checkpoint:
  sam2_hiera_large.pt  from https://dl.fbaipublicfiles.com/segment_anything_2/
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def _to_uint8_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """Normalise a 2D array to uint8 RGB (H, W, 3)."""
    lo, hi = slice_2d.min(), slice_2d.max()
    if hi - lo < 1e-8:
        norm = np.zeros_like(slice_2d, dtype=np.uint8)
    else:
        norm = ((slice_2d - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


class SAM2VolumeSegmenter:
    """
    SAM 2 video predictor adapted for volumetric medical image segmentation.

    Parameters
    ----------
    checkpoint : str
        Path to sam2_*.pt checkpoint.
    model_cfg : str
        SAM 2 model config name, e.g. 'sam2_hiera_l' (large),
        'sam2_hiera_b+' (base+), 'sam2_hiera_s' (small).
    device : str
        'cuda', 'mps', or 'cpu'. Auto-detected if empty.
    """

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str = "sam2_hiera_l",
        device: str = "",
    ):
        self.checkpoint = checkpoint
        self.model_cfg  = model_cfg
        self.device     = self._get_device(device)
        self.predictor  = None
        self._load_model()

    @staticmethod
    def _get_device(requested: str) -> str:
        import torch
        if requested:
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint, device=self.device
            )
            print(f"Loaded SAM 2 ({self.model_cfg}) on {self.device}")
        except ImportError:
            raise ImportError(
                "Install SAM 2: "
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )

    def segment_volume(
        self,
        volume: np.ndarray,
        prompt_slice_idx: int,
        prompt_points: Optional[List[List[int]]] = None,
        prompt_labels: Optional[List[int]] = None,
        prompt_bbox: Optional[List[int]] = None,
        propagation_axis: int = 2,
    ) -> np.ndarray:
        """
        Propagate a segmentation mask through the full volume.

        Parameters
        ----------
        volume : np.ndarray
            3D array (H, W, D) or 4D (H, W, D, T). Will be sliced along
            propagation_axis.
        prompt_slice_idx : int
            Index of the reference slice to provide the prompt on.
        prompt_points : [[x, y], ...] pixel coordinates on the reference slice.
        prompt_labels : 1=foreground, 0=background per point. Default all 1.
        prompt_bbox : [x0, y0, x1, y1] bounding box on the reference slice.
        propagation_axis : int
            Axis along which to extract 2D slices (default=2, i.e. Z-axis).

        Returns
        -------
        mask_volume : np.ndarray of bool, same spatial shape as volume
        """
        import torch

        # Extract slices along the propagation axis
        n_slices = volume.shape[propagation_axis]
        slices = [
            np.take(volume, i, axis=propagation_axis)
            for i in range(n_slices)
        ]
        rgb_frames = [_to_uint8_rgb(s if s.ndim == 2 else s[..., 0])
                      for s in slices]

        mask_volume = np.zeros(
            [s if ax != propagation_axis else n_slices
             for ax, s in enumerate(volume.shape[:3])],
            dtype=bool
        )

        with torch.inference_mode():
            state = self.predictor.init_state_from_frames(rgb_frames)

            # Add prompt on the reference slice
            obj_id = 1
            if prompt_bbox is not None:
                box = np.array(prompt_bbox, dtype=np.float32)
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=prompt_slice_idx,
                    obj_id=obj_id,
                    box=box,
                )
            if prompt_points is not None:
                pts = np.array(prompt_points, dtype=np.float32)
                lbs = np.array(
                    prompt_labels if prompt_labels else [1] * len(prompt_points),
                    dtype=np.int32,
                )
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=prompt_slice_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbs,
                )

            # Propagate through all slices
            for frame_idx, obj_ids, masks in \
                    self.predictor.propagate_in_video(state):
                if obj_id in obj_ids:
                    idx = obj_ids.index(obj_id)
                    m = masks[idx].cpu().numpy().squeeze() > 0.5
                    idx_tuple = [slice(None)] * mask_volume.ndim
                    idx_tuple[propagation_axis] = frame_idx
                    mask_volume[tuple(idx_tuple)] = m

        return mask_volume

    def segment_4dflow(
        self,
        magnitude: np.ndarray,
        prompt_slice_idx: int,
        prompt_points: Optional[List[List[int]]] = None,
        prompt_bbox: Optional[List[int]] = None,
        time_phase: int = 0,
    ) -> np.ndarray:
        """
        Convenience wrapper for 4D flow MRI magnitude volumes.

        Segments on a chosen time phase and propagates through Z.

        Parameters
        ----------
        magnitude : (H, W, Z, T) 4D magnitude volume
        prompt_slice_idx : Z-slice index for the prompt
        prompt_points : [[x, y], ...] on the reference slice
        prompt_bbox   : [x0, y0, x1, y1] on the reference slice
        time_phase    : cardiac phase to use for segmentation (default 0)

        Returns
        -------
        mask_3d : (H, W, Z) boolean mask
        """
        # Use a single time phase for segmentation, propagate through Z
        vol_3d = magnitude[..., time_phase]    # (H, W, Z)
        return self.segment_volume(
            vol_3d,
            prompt_slice_idx=prompt_slice_idx,
            prompt_points=prompt_points,
            prompt_bbox=prompt_bbox,
            propagation_axis=2,
        )
