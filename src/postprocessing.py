"""
Post-processing for Segmentation Masks
=======================================
- keep_largest_component : Remove small spurious regions
- fill_holes             : Fill enclosed holes in a binary mask
- smooth_mask            : Apply morphological closing to smooth boundaries
- threshold_mask         : Convert probability map to binary mask
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Parameters
    ----------
    mask : boolean or uint8 ndarray (2D or 3D)

    Returns
    -------
    cleaned : bool ndarray, same shape as mask
    """
    labeled, n_comps = ndimage.label(mask.astype(bool))
    if n_comps == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, range(1, n_comps + 1))
    largest = np.argmax(sizes) + 1
    return labeled == largest


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask.

    Parameters
    ----------
    mask : bool or uint8 ndarray (2D or 3D)

    Returns
    -------
    filled : bool ndarray
    """
    return ndimage.binary_fill_holes(mask.astype(bool))


def smooth_mask(
    mask: np.ndarray,
    iterations: int = 2,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Smooth mask boundaries using morphological closing.

    Parameters
    ----------
    mask       : bool ndarray
    iterations : number of dilation+erosion iterations
    structure  : structuring element (default: cross for 2D, cube for 3D)

    Returns
    -------
    smoothed : bool ndarray
    """
    from scipy.ndimage import binary_closing
    return binary_closing(mask.astype(bool),
                          structure=structure,
                          iterations=iterations).astype(bool)


def threshold_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Threshold a probability/logit map to obtain a binary mask.

    Parameters
    ----------
    prob_map  : float ndarray in [0, 1]
    threshold : decision threshold (default 0.5)

    Returns
    -------
    binary mask : bool ndarray
    """
    return prob_map >= threshold


def refine_mask(
    mask: np.ndarray,
    remove_small: bool = True,
    fill: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Full post-processing pipeline: largest component → fill holes → smooth.

    Parameters
    ----------
    mask        : raw binary mask
    remove_small: if True, keep only the largest connected component
    fill        : if True, fill enclosed holes
    smooth      : if True, apply morphological closing

    Returns
    -------
    refined mask : bool ndarray
    """
    out = mask.astype(bool)
    if remove_small:
        out = keep_largest_component(out)
    if fill:
        out = fill_holes(out)
    if smooth:
        out = smooth_mask(out)
    return out


# Missing import fix
from typing import Optional
