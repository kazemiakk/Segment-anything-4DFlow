"""
Preprocessing Utilities for MRI & DSA Images
=============================================
- load_nifti         : Load a NIfTI (.nii / .nii.gz) volume
- save_nifti         : Save a mask as NIfTI
- normalize_window   : Intensity windowing + normalisation
- load_image_dir     : Load a directory of 2D images (PNG/JPG/TIFF)
- extract_slice      : Extract a 2D slice from a 3D volume
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# NIfTI helpers (requires nibabel)
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> Tuple[np.ndarray, object]:
    """
    Load a NIfTI file.

    Returns
    -------
    data   : np.ndarray — image array
    header : nibabel header object (for preserving affine when saving)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel: pip install nibabel")

    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img


def save_nifti(mask: np.ndarray, reference_img, out_path: str):
    """
    Save a binary mask as a NIfTI file, reusing the affine from reference_img.

    Parameters
    ----------
    mask          : np.ndarray (boolean or uint8)
    reference_img : nibabel image object (provides affine & header)
    out_path      : output file path (.nii or .nii.gz)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Install nibabel: pip install nibabel")

    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine,
                        reference_img.header),
        out_path,
    )
    print(f"Saved mask to {out_path}")


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def normalize_window(
    image: np.ndarray,
    window_center: Optional[float] = None,
    window_width:  Optional[float] = None,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Apply window/level intensity normalisation, then scale to out_range.

    If window_center and window_width are None, uses the global min/max.

    Parameters
    ----------
    image         : input array (any shape)
    window_center : centre of the intensity window
    window_width  : width of the intensity window
    out_range     : (low, high) output range

    Returns
    -------
    normalised float32 array
    """
    img = image.astype(np.float32)

    if window_center is None or window_width is None:
        lo, hi = img.min(), img.max()
    else:
        lo = window_center - window_width / 2.0
        hi = window_center + window_width / 2.0

    img = np.clip(img, lo, hi)
    denom = hi - lo if hi > lo else 1.0
    img = (img - lo) / denom                           # [0, 1]
    img = img * (out_range[1] - out_range[0]) + out_range[0]
    return img


# ---------------------------------------------------------------------------
# Image directory loader (for DSA sequences)
# ---------------------------------------------------------------------------

def load_image_dir(
    directory: str,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp"),
    grayscale: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from a directory, sorted by filename.

    Returns
    -------
    images : list of (H, W) or (H, W, 3) uint8 arrays
    paths  : corresponding file paths
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Install Pillow: pip install Pillow")

    dir_path = Path(directory)
    paths = sorted([
        p for p in dir_path.iterdir()
        if p.suffix.lower() in extensions
    ])
    images = []
    for p in paths:
        img = Image.open(str(p))
        if grayscale:
            img = img.convert("L")
        images.append(np.array(img))
    return images, [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Volume slicing
# ---------------------------------------------------------------------------

def extract_slice(
    volume: np.ndarray,
    index: int,
    axis: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume and optionally normalise to [0, 255].

    Returns
    -------
    slice_2d : (H, W) uint8 array
    """
    slc = np.take(volume, index, axis=axis)
    if slc.ndim > 2:
        slc = slc[..., 0]           # take first channel if 4D
    if normalize:
        slc = normalize_window(slc)
        slc = (slc * 255).astype(np.uint8)
    return slc
