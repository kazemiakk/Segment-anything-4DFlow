"""
Segmentation Metrics
=====================
- dice_coefficient   : Dice similarity coefficient (DSC)
- iou_score          : Intersection over Union (IoU / Jaccard)
- hausdorff_distance : 95th percentile Hausdorff distance
- compute_all        : Compute all metrics at once
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred: np.ndarray, gt: np.ndarray,
                     smooth: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient.

    DSC = 2 * |Pred ∩ GT| / (|Pred| + |GT|)

    Parameters
    ----------
    pred, gt : bool or binary ndarray (any shape)
    smooth   : smoothing constant to avoid division by zero

    Returns
    -------
    dsc : float in [0, 1]
    """
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    intersection = (p & g).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray,
              smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard index).

    IoU = |Pred ∩ GT| / |Pred ∪ GT|

    Returns
    -------
    iou : float in [0, 1]
    """
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    intersection = (p & g).sum()
    union = (p | g).sum()
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                           voxel_spacing: float = 1.0) -> float:
    """
    95th-percentile Hausdorff Distance.

    Parameters
    ----------
    pred, gt       : bool ndarray (2D or 3D)
    voxel_spacing  : isotropic voxel size in mm (default 1.0)

    Returns
    -------
    hd95 : float (mm)
    """
    pred_pts = np.argwhere(pred.astype(bool))
    gt_pts   = np.argwhere(gt.astype(bool))

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    # Forward and backward directed distances
    d_fwd = np.array([np.min(np.linalg.norm(gt_pts - p, axis=1))
                      for p in pred_pts])
    d_bwd = np.array([np.min(np.linalg.norm(pred_pts - g, axis=1))
                      for g in gt_pts])

    hd95 = np.percentile(np.concatenate([d_fwd, d_bwd]), 95) * voxel_spacing
    return float(hd95)


def compute_all(
    pred: np.ndarray,
    gt:   np.ndarray,
    voxel_spacing: float = 1.0,
) -> Dict[str, float]:
    """
    Compute Dice, IoU, and HD95 together.

    Returns
    -------
    metrics : dict with keys 'Dice', 'IoU', 'HD95_mm'
    """
    return {
        "Dice"   : dice_coefficient(pred, gt),
        "IoU"    : iou_score(pred, gt),
        "HD95_mm": hausdorff_distance_95(pred, gt, voxel_spacing),
    }
