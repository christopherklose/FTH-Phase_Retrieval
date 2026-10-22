"""
Python library for some general functions regarding the detector, i.e., background correction 

2026
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

def correct_background(image, edge_range, std):
    """
    Single-frame background correction for visible light in the chamber.

    A row-wise background profile is estimated from the mean of the
    `edge_range` leftmost and rightmost columns, then smoothed with a
    Gaussian of width `std`. The left-edge profile is subtracted from the
    upper half of the image, and the right-edge profile from the lower half.

    Parameters
    ----------
    image : 2d array
        single camera frame
    edge_range : int
        nr of columns at the left and right edge used to estimate the background
    std : float
        std of the gaussian filter applied to the background profiles

    Returns
    -------
    image_corrected, background : ndarray (float), same shape as `image`
    """
    image = np.asarray(image, dtype=float)
    mid = image.shape[-2] // 2

    left = gaussian_filter1d(image[:, :edge_range].mean(axis=1), std)
    right = gaussian_filter1d(image[:, -edge_range:].mean(axis=1), std)

    profile = np.concatenate([left[:mid], right[mid:]])  # shape (H,)
    background = np.broadcast_to(profile[:, None], image.shape).copy()

    return image - background, background


def correct_quadrant_background(image, offset=10, length=500):
    """
    Subtract a constant background per quadrant, estimated as the
    nan-median of a (length x length) window in each outer corner,
    `offset` pixels from the image edges.

    Parameters
    ----------
    image : 2d array
        single camera frame
    offset : int
        distance in px of the corner windows from the image edges
    length : int
        side length in px of the corner windows

    Returns
    -------
    image_corrected, background : ndarray (float), same shape as `image`
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape
    m, n = H // 2, W // 2

    top, bottom = slice(offset, offset + length), slice(H - offset - length, H - offset)
    left, right = slice(offset, offset + length), slice(W - offset - length, W - offset)

    background = np.empty_like(image)
    background[:m, :n] = np.nanmedian(image[top, left])
    background[:m, n:] = np.nanmedian(image[top, right])
    background[m:, :n] = np.nanmedian(image[bottom, left])
    background[m:, n:] = np.nanmedian(image[bottom, right])

    return image - background, background