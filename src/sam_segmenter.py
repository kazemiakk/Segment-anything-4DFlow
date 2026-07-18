"""
SAM / MedSAM Wrapper for Prompted Segmentation
================================================
Provides a unified interface for running SAM (ViT-H) and MedSAM
with bounding-box or point prompts on 2D images.

Supports:
  - SAM  : facebook/segment-anything
  - MedSAM: bowang-lab/MedSAM (ViT-B based, fine-tuned on medical images)

Usage
-----
from src.sam_segmenter import SAMSegmenter

seg = SAMSegmenter(
    checkpoint="./checkpoints/sam_vit_h_4b8939.pth",
    model_type="vit_h",          # "vit_h" | "vit_l" | "vit_b" (MedSAM uses "vit_b")
)
mask = seg.predict_bbox(image_np, bbox=[x0, y0, x1, y1])
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


class SAMSegmenter:
    """
    Unified wrapper for SAM and MedSAM inference on 2D images.

    Parameters
    ----------
    checkpoint : str
        Path to the model .pth checkpoint file.
    model_type : str
        SAM model type: 'vit_h', 'vit_l', or 'vit_b'.
        MedSAM uses 'vit_b'.
    device : str
        'cuda', 'mps', or 'cpu'. Auto-detected if empty.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_h",
        device: str = "",
    ):
        self.checkpoint  = checkpoint
        self.model_type  = model_type
        self.device      = self._get_device(device)
        self.predictor   = None
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
        """Load the SAM / MedSAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            sam.to(self.device)
            self.predictor = SamPredictor(sam)
            print(f"Loaded SAM ({self.model_type}) from {self.checkpoint}")
        except ImportError:
            raise ImportError(
                "Install segment-anything: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    def set_image(self, image: np.ndarray):
        """
        Set the image for inference. Call once per image before predict_*.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:
            image = ((image - image.min()) /
                     (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        self.predictor.set_image(image)

    def predict_bbox(
        self,
        image: np.ndarray,
        bbox: List[int],
        multimask: bool = False,
    ) -> np.ndarray:
        """
        Predict a mask from a bounding box prompt.

        Parameters
        ----------
        image : (H, W) or (H, W, 3) ndarray
        bbox  : [x_min, y_min, x_max, y_max] in pixel coordinates
        multimask : if True, returns best of 3 SAM masks

        Returns
        -------
        mask : (H, W) boolean ndarray
        """
        import torch
        self.set_image(image)
        box = np.array(bbox, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=multimask,
        )
        best = masks[np.argmax(scores)]
        return best.astype(bool)

    def predict_points(
        self,
        image: np.ndarray,
        points: List[List[int]],
        labels: Optional[List[int]] = None,
        multimask: bool = False,
    ) -> np.ndarray:
        """
        Predict a mask from point prompts.

        Parameters
        ----------
        image  : (H, W) or (H, W, 3) ndarray
        points : [[x1, y1], [x2, y2], ...] foreground (and background) points
        labels : 1=foreground, 0=background for each point. Defaults to all 1.
        multimask : if True, returns best of 3 SAM masks

        Returns
        -------
        mask : (H, W) boolean ndarray
        """
        self.set_image(image)
        pts = np.array(points, dtype=np.float32)
        lbs = np.array(labels if labels is not None else [1] * len(points),
                       dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            multimask_output=multimask,
        )
        best = masks[np.argmax(scores)]
        return best.astype(bool)

    def predict_auto(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> List[dict]:
        """
        Run automatic mask generation (no prompts).

        Returns a list of mask dictionaries (SAM format).
        """
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("Install segment-anything first.")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:
            image = ((image - image.min()) /
                     (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        generator = SamAutomaticMaskGenerator(
            self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return generator.generate(image)
