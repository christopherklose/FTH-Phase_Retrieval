import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, next_fast_len
from scipy.ndimage import fourier_shift
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.constants import h, c, e
from skimage.registration import phase_cross_correlation


def crop_center(image, center, square=True):
    """Return a symmetric crop around given center coordinate."""
    n0, n1 = image.shape
    c0, c1 = center
    m0, m1 = [min(c, s - c) for s, c in zip(image.shape, center)]
    if square:
        m0 = m1 = min(m0, m1)
    roi_center = np.s_[c0 - m0:c0 + m0, c1 - m1:c1 + m1]
    return image[roi_center]

def pad_for_fft(image):
    """Zeropad image to next FFT-efficient shape.
    
    Uses scipy.fft.next_fast_len() to calculate new shape and pads with zeros
    accordingly. Causes half-pixel misalignments for axes with odd length.
    """
    fastshape = [next_fast_len(s) for s in image.shape]
    quot_rem = [divmod(fs - s, 2) for fs, s in zip(fastshape, image.shape)]
    pad = [[q, q + r] for q, r in quot_rem]
    # print(f"fast shape: {fastshape}")
    # print(f"image shape: {image.shape}")
    # print(f"quot, rem: {quot_rem}")
    # print(f"padding: {pad}")
    return np.pad(image, pad)


def shift_phase(image, phase):
    """
    Multiply image with a complex phase factor
    
    Parameters
    ----------
    image : array
        input hologram
    phi: scalar
        phase to multiply to the hologram
    
    Returns
    -------
    shifted: array
        phase shifted image
    -------
    author: KG 2020
    """
    return image * np.exp(1j * phase)


def propagate(holo, propdist, detectordist, pixelsize, energy, int_mul=True):
    """
    Apply free-space propagator to input array.
    
    Parameters
    ----------
    holo : array
        input hologram
    propdist: scalar
        distance of propagation in meters
    detectordist: float
        real-space distance from sample to detector in meters
    pixelsize: float
        real-space size of detector pixels in meters
    energy: float
        photon energy in eV
    int_mul: bool, optional
        If True (default), coerce propdist to integer multiples of wavelength
    
    Returns
    -------
    propagated: array
        propagated hologram
    """
    
    wl = h * c / (energy * e)
    if int_mul:
        propdist = (propdist // wl) * wl
    
    dist_phase = 2 * np.pi * propdist / wl
    
    n0, n1 = holo.shape
    c0, c1 = [idx - s // 2 for idx, s in zip(np.ogrid[:n0, :n1], holo.shape)]
    q = c0 ** 2 + c1 ** 2
    phase = dist_phase * (1 - q * (pixelsize / detectordist) ** 2) ** 0.5
    return np.exp(1j * phase.astype(np.single)) * holo
    


def shift_image(image, shift):
    """
    Shifts image with sub-pixel precision using Fourier shift
    
    Parameters
    ----------
    image: array
        Input image to be shifted
    shift: (dy, dx) sequence of floats
        x and y translation in px

    Returns
    -------
    shifted: array
        Shifted image

    author: CK 2021
    """
    shifted = fourier_shift(fft2(image), shift)
    shifted = ifft2(shifted)
    return shifted.real


def scalar_norm(arr1, arr2):
    """Calculate normalized sum of element-wise product for two arrays."""
    # this einstein sum expression is equivalent to, but faster than,
    # (arr1 * arr2).sum() / (arr2 * arr2).sum()
    return np.einsum("ij, ij", arr1, arr2) / np.einsum("ij, ij", arr2, arr2)


def match_linreg(image, reference, roi=None):
    """
    Adjust values in image to match reference using linear regression.
    """
    if roi is None:
        roi = np.s_[()]
    reg = linregress(image[roi].flatten(), reference[roi].flatten())
    print("== Linear regression results ==")
    for att in ["intercept", "slope", "rvalue"]:
        print(f"{att:10s}: {getattr(reg, att):.3f}")
    return reg.intercept + reg.slope * image


def estimate_pedestal(image, threshold_percent=1.0, plot=False):
    """
    Estimate image baseline by calculating low percentile of non-zero values.
    
    Parameters
    ----------
    image: array
    threshold_percent: float
    plot: bool, default False
        If True, plot a histogram of intensity values in the range [0, 1000]
    """
    ped = np.percentile(image[image > 0], threshold_percent)
    
    if plot:
        fig, ax = plt.subplots()
        _ = ax.hist(image.flatten(), np.linspace(0, 1000, 100))
        ax.axvline(ped, c='r', label=f"pedestal (th={threshold_percent})")
        ax.set_yscale("log")
        ax.set_xlabel("pixel value")
        ax.set_ylabel("absolute count")
        ax.legend()
    return ped


def reconstruct(holo, inverse=False):
    if inverse:
        return ifftshift(ifft2(fftshift(holo)))
    else:
        return ifftshift(fft2(fftshift(holo)))


def roll_center(image, center):
    """Move given coordinate to image center by rolling axes."""
    delta = [s // 2 - c for s, c in zip(image.shape, center)]
    return np.roll(image, delta, axis=(0, 1))


def image_registration_skimage(image, reference, roi=None):
    '''
    Aligns two images with sub-pixel precission through image registration
    
    
    Parameters
    ----------
    image: array
        Moving image, will be aligned with respect to reference image
        
    reference: array
        static reference image
    
    roi: region of interest defining the region of the images used to calc
        the alignment
    
    Returns
    -------
    shifted: array
        Shifted/aligned moving image
    shift: tuple
        shift values
    -------
    author: CK 2021
    '''
    if roi is None:
        reg_im = image
        reg_ref = reference
    else:
        reg_im = image[roi]
        reg_ref = reference[roi]
    
    # Calculate Shift
    shift, error, diffphase = phase_cross_correlation(
        reg_ref, reg_im, upsample_factor=1000
        )
    return shift_image(image, shift), shift
