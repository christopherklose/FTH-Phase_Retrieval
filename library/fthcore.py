"""
Python Dictionary for FTH reconstructions

2016/2019/2020/2021/2024
@authors:   MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            FB: Felix Buettner (felix.buettner@helmholtz-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            CK: Christopher Klose (christopher.klose@mbi-berlin.de)
            SG: Simon Gaebel (simon.gaebel@mbi-berlin.de)
            
Update: 04.11.2024 
        cleaned up library by removing unnecessary functions
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import scipy.constants as cst
from skimage.draw import disk as circle

###########################################################################################

#                               LOAD DATA                                                 #

###########################################################################################




def make_square(image):
    '''
    Return the input image in a quadratic format by omitting some rows or columns.
    
    Parameters
    ----------
    image : array
        input image of shape (2,N)
    
    Returns
    -------
    im : array
        quadratic form of image
    -------
    author: KG 2020
    '''
    size = image.shape
    if size[0]<size[1]:
        return image[:, :size[0]]
    elif size[0]>size[1]:
        return image[:size[1], :]
    else:
        return image


###########################################################################################

#                               RECONSTRUCTION                                            #

###########################################################################################

def reconstruct(holo):
    '''
    Reconstruct the hologram by Fourier transformation
    
    Parameters
    ----------
    holo : array
        input hologram of shape (2,N)
    
    Returns
    -------
    image: array
        Fourier transformed and shifted image of the input hologram
    -------
    author: MS 2016
    '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(holo)))

def reconstructCDI(image):
    '''
    Reconstruct the image by fft. must be applied to retrieved images
    -------
    author: RB 2020
    '''
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(image)))


def adjust_wave(ar, ratio):
    """
    Function to zoom in and out of array without changing its size
    """
    size = ar.shape
    old_x = np.linspace(-size[0] / 2, size[0] / 2, size[0])
    old_y = np.linspace(-size[1] / 2, size[1] / 2, size[1])
    new_x = old_x * ratio
    new_y = old_y * ratio
    res_func = scipy.interpolate.RectBivariateSpline(old_x, old_y, ar, kx=3, ky=3)
    return res_func(new_x, new_y)


###########################################################################################

#                                 PROPAGATION                                             #

###########################################################################################

def propagate(holo, prop_l, experimental_setup, integer_wl_multiple=True):
    '''
    Propagate the hologram
    
    Parameters
    ----------
    holo : array
        input hologram
    prop_l: scalar
        distance of propagation in metre
    experimental_setup: dict
        experimental setup parameters in the following form: {'ccd_dist': [in metre], 'energy': [in eV], 'px_size': [in metre]}
    integer_wl_multiple: bool, optional
        Use a propagation, that is an integer multiple of the x-ray wave length, default is True.
    
    Returns
    -------
    prop_holo: array
        propagated hologram
    -------
    author: MS 2016
    '''
    wl = cst.h * cst.c / (experimental_setup['energy'] * cst.e)
    if integer_wl_multiple:
        prop_l = np.round(prop_l / wl) * wl

    l1, l2 = holo.shape
    q0, p0 = [s / 2 for s in holo.shape] # centre of the hologram
    q, p = np.mgrid[0:l1, 0:l2]  #grid over CCD pixel coordinates   
    pq_grid = (q - q0) ** 2 + (p - p0) ** 2 #grid over CCD pixel coordinates, (0,0) is the centre position
    dist_wl = 2 * prop_l * np.pi / wl
    phase = (dist_wl * np.sqrt(1 - (experimental_setup['px_size']/ experimental_setup['ccd_dist']) ** 2 * pq_grid))
    return np.exp(1j * phase) * holo

def propagate_realspace(image, prop_l, experimental_setup, integer_wl_multiple=True):
    '''
    Propagate the real space image (reconstruction)
    
    Parameters
    ----------
    image : array
        input image
    prop_l: scalar
        distance of propagation in metre
    experimental_setup: dict
        experimental setup parameters in the following form: {'ccd_dist': [in metre], 'energy': [in eV], 'px_size': [in metre]}
    integer_wl_multiple: bool, optional
        Use a propagation, that is an integer multiple of the x-ray wave length, default is True.
    
    Returns
    -------
    prop_im: array
        propagated image
    -------
    author: KG 2020
    '''
    holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    holo = propagate(holo, prop_l, experimental_setup, integer_wl_multiple = integer_wl_multiple) 
    return reconstruct(holo)


###########################################################################################

#                                   HIGH PASS FILTER                                      #

###########################################################################################

def highpass(data, amplitude, sigma):
    '''
    Creates a highpass Gauss filter with variable ampltitude and sigma and multiplies it to the given data.
    
    Parameters
    ----------
    data : array
        the hologram you want to apply the highpass filter to
    A : scalar
        ampltitude of the Gauss, please input a positive number because -A is taken as factor for the Gauss
    sigma: scalar
        sigma of the Gauss
    
    Returns
    -------
    data * HP : array
        given data multiplied with the Gauss high pass filter
    HP: array
        high pass filter
    -------
    author: KG 2020
    '''
    x0, y0 = [s//2 for s in data.shape]
    x,y = np.mgrid[-x0:x0, -y0:y0]
    HP = 1 - amplitude * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return (data * HP, HP)


###########################################################################################

#                                   Heraldo reconstruction                                #

###########################################################################################

def differential_operator(shape, center, experimental_setup, angle=0):
    """
    Calculates Fourier-space differential operator for heraldo reconstruction

    Parameter
    ---------
    shape: int tuple
        shape of output array
    center: tuple
        array center coordinates (y,x)
    experimental_setup: dict
        must contain detector pixel_size ["px_size"], distance ["ccd_dist"],
        wavelength ["lambda"]
    angle: float
        rotation angle of heraldo slit

    Returns:
    --------
    return: complex array
        differential operator in Fourier space
    """

    # Convert deg to rad
    angle = np.deg2rad(angle)

    # Create x,y grid to convert pixel in q-space
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[0]))

    # Center meshgrid
    y, x = y - center[0], x - center[1]

    # Multiplay with pixel size
    y, x = y * experimental_setup["px_size"], x * experimental_setup["px_size"]

    # Convert to q-space
    qy = (
        4
        * np.pi
        / experimental_setup["lambda"]
        * np.sin(0.5 * np.arctan(y / experimental_setup["ccd_dist"]))
    )
    qx = (
        4
        * np.pi
        / experimental_setup["lambda"]
        * np.sin(0.5 * np.arctan(x / experimental_setup["ccd_dist"]))
    )

    # Normalize q space to [-1,1]
    qy, qx = qy / np.max(np.abs(qy)), qx / np.max(np.abs(qx))

    return 2j * np.pi * qx * np.cos(angle) + 2j * np.pi * qy * np.sin(angle)


def reconstruct_heraldo(
    holo, experimental_setup, center=None, prop_dist=0, phase=0, angle=0
):
    """
    Reconstruction of holograms in heraldo reference scheme

    Parameter:
    ----------
    holo: array
        Centered input hologram
    experimental_setup: dict
        must contain detector pixel_size ["px_size"], distance ["ccd_dist"],
        wavelength ["lambda"]
    center: tuple or None
        array center coordinates (y,x)
    prop_dist: float
        propagation distance
    phase: float
        global phase shift of complex array
    angle: float
        rotation angle of heraldo slit in deg

    returns:
    image: array
        reconstructed image
    heraldo_operator: complex array
        differential operator in Fourier space
    """

    if center is None:
        center = np.array(holo.shape) / 2

    heraldo_operator = differential_operator(
        holo.shape, center, experimental_setup, angle=angle
    )
    holo = holo * heraldo_operator
    holo = fth.propagate(
        holo, prop_dist * 1e-6, experimental_setup=experimental_setup
    ) * np.exp(1j * phase)
    image = fth.reconstruct(holo)

    return image, heraldo_operator