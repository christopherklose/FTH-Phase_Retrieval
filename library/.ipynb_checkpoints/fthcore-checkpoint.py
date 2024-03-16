"""
Python Dictionary for FTH reconstructions

2016/2019/2020/2021
@authors:   MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            FB: Felix Buettner (felix.buettner@helmholtz-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import scipy.constants as cst
from skimage.draw import disk as circle

###########################################################################################

#                               LOAD DATA                                                 #

###########################################################################################

def load_both(pos, neg, auto_factor=True):
    '''
    Load images for a double helicity reconstruction
    
    Parameters
    ----------
    pos : array
        positive helicity image
    neg : array
        negative helicity image
    auto_factor: bool, optional
        determine the factor by which (pos+neg) is multiplied automatically, if False: factor is set to 0.5, default is True
    
    Returns
    -------
    holo : array
        quadratic difference hologram
    factor: scalar
        factor used for the difference hologram
    -------
    author: KG 2019
    '''
    size = pos.shape
    if auto_factor:
        offset_pos = (np.mean(pos[:10,:10]) + np.mean(pos[-10:,:10]) + np.mean(pos[:10,-10:]) + np.mean(pos[-10:,-10:]))/4
        offset_neg = (np.mean(neg[:10,:10]) + np.mean(neg[-10:,:10]) + np.mean(neg[:10,-10:]) + np.mean(neg[-10:,-10:]))/4
        topo = pos - offset_pos + neg - offset_neg
        pos = pos - offset_pos
        factor = np.sum(np.multiply(pos,topo))/np.sum(np.multiply(topo, topo))
        print('Auto factor = ' + str(factor))
    else:
        topo = pos + neg
        factor = 0.5
    
    holo = pos - factor * topo
    return (make_square(holo), factor)

def load_single(image, topo, helicity, auto_factor=False):
    '''
    Load image for a single helicity reconstruction
    
    Parameters
    ----------
    image : array
        data of the single helicity image
    topo : array
        topography data (sum of positive and negative helicity reference images)
    helicity: bool
        True/False for pos/neg single helicity image
    auto_factor: bool, optional
        determine the factor by which (pos+neg) is multiplied automatically, if False: factor is set to 0.5, default is True
    
    Returns
    -------
    holo : array
        quadratic difference hologram
    factor: scalar
        factor used for the difference hologram
    -------
    author: KG 2019
    '''
    #load the reference for topology
    topo = topo.astype(np.int64)
    #load the single image
    image = image.astype(np.int64)

    size = image.shape

    if auto_factor:
        offset_sing = (np.mean(image[:10,:10]) + np.mean(image[-10:,:10]) + np.mean(image[:10,-10:]) + np.mean(image[-10:,-10:]))/4
        image = image - offset_sing
        offset_topo = (np.mean(topo[:10,:10]) + np.mean(topo[-10:,:10]) + np.mean(topo[:10,-10:]) + np.mean(topo[-10:,-10:]))/4
        topo = topo - offset_topo
        factor = np.sum(np.multiply(image, topo))/np.sum(np.multiply(topo, topo))
        print('Auto factor = ' + str(factor))
    else:
        factor = 0.5

    if helicity:
        holo = image - factor * topo
    else:
        holo = -1 * (image - factor * topo)

    #make sure to return a quadratic image, otherwise the fft will distort the image
    return (make_square(holo), factor)


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

#                               PLOTTING                                                  #

###########################################################################################

def plot(image, scale = (2,98), color = 'gray', colorbar = True):
    '''
    plot the image with the given scale, colormap and with a colorbar
    
    Parameters
    ----------
    image : array
        input image of shape (2,N)
    scale: tuple of two scalars, optional
        percentage used for the scaling (passed to np.percentile()), default is (2,98)
    color: str, optional
        matplotlib colormap, default is 'gray'
    colorbar: bool, optional
        whether to plot a colorbar, default is True
    
    Returns
    -------
    fig : matplotlib figure
        figure
    ax: matplotlib axes
        axes of fig
    --------
    author: KG 2019
    '''
    mi, ma = np.percentile(image, scale)
    fig, ax = plt.subplots()
    im = ax.imshow(image, vmin = mi, vmax = ma, cmap = color)
    if colorbar:
        plt.colorbar(im)
    return (fig, ax)

def plot_ROI(image, ROI, scale = (0,100), color = 'gray', colorbar = True):
    '''
    Plot the ROI of the image
    
    Parameters
    ----------
    image : array
        input image of shape (2,N)
    ROI: numpy slice
        slice of the ROI
    scale: tuple of two scalars, optional
        percentage used for the scaling (passed to np.percentile()), default is (0, 100)
    color: str, optional
        matplotlib colormap, default is 'gray'
    colorbar: bool, optional
        whether to plot a colorbar, default is True
    
    Returns
    -------
    fig : matplotlib figure
        figure
    ax: matplotlib axes
        axes of fig
    --------
    author: KG 2019
    '''
    mi, ma = np.percentile(image[ROI], scale)
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(image[ROI]), vmin = mi, vmax = ma, cmap = color)
    if colorbar:
        plt.colorbar()
    return (fig, ax)



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
###########################################################################################

#                               CENTERING                                                 #

###########################################################################################

def integer(n):
    '''
    Return a rounded integer
    
    Parameters
    ----------
    n : scalar
        input number
    
    Returns
    -------
    r: int
        n rounded and cast as int
    -------
    author: KG 2020
    '''
    return np.int(np.round(n))

def set_center(image, center, fill=0):
    '''
    Move given coordinate to center of image.
    Rolls input image to new center. Wrapped values are set to zero by default.centering routine shifts the image in a cyclical fashion
    
    Parameters
    ----------
    image : array
        input image
    center: sequence of scalars
        coordinates of the center (x, y)
    fill: scalar, optional
        value of the wrapped pixels, default is zero
    
    Returns
    -------
    image_roll: array
        centerd image
    -------
    author: MS 2016/2021, KG 2019
    '''
    dx, dy = [int(s / 2 - c) for s, c in zip(image.shape, center)]
    im_roll = np.roll(image, (dx, dy), axis=(1, 0))
    x0, x1 = (dx, None) if dx < 0 else (None, dx)
    y0, y1 = (dy, None) if dy < 0 else (None, dy)
    im_roll[x0:x1, :] = fill
    im_roll[:, y0:y1] = fill
    return im_roll

def sub_pixel_centering(reco, dx, dy):
    '''
    Routine for subpixel centering
    
    Parameters
    ----------
    reco : array
        input image, reconstruction
    dx: scalar
        amount by which to shift the phase in x direction for subpixel center correction
    dy: scalar
        amount by which to shift the phase in y direction for subpixel center correction
    
    Returns
    -------
    reco_shift: array
        subpixel corrected hologram
    ------
    author: KG, 2020
    '''
    sx, sy = reco.shape
    x = np.arange(- sy//2, sy//2, 1)
    y = np.arange(- sx//2, sx//2, 1)
    xx, yy = np.meshgrid(x, y)
    return reco * np.exp(2j * np.pi * (xx * dx/sx + yy * dy/sy))

###########################################################################################

#                                 BEAM STOP MASK                                          #

###########################################################################################

def mask_beamstop(holo, bs_size, sigma = 3, center = None):
    '''
    A smoothed circular region of the imput image is set to zero.
    
    Parameters
    ----------
    holo : array
        input hologram
    bs_size: scalar
        beamstop diameter inpixels
    sigma: scalar or sequence of scalars, optional
        Passed to scipy.ndimage.gaussian_filter(). Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes. Default is 3.
    center: sequence of scalars, optional
        If given, the beamstop is masked at that position, otherwise the center of the image is taken. Default is None.
    
    Returns
    -------
    masked_holo: array
        hologram with smoothed beamstop edges
    -------
    author: MS 2016, KG 2019
    '''
    if center is None:
        x0, y0 = [integer(c/2) for c in holo.shape]
    else:
        x0, y0 = [integer(c) for c in center]

    #create the beamstop mask using scikit-image's circle function
    bs_mask = np.zeros(holo.shape)
    yy, xx = circle(y0, x0, bs_size/2)
    bs_mask[yy, xx] = 1
    bs_mask = np.logical_not(bs_mask).astype(np.float64)
    #smooth the mask with a gaussion filter    
    bs_mask = gaussian_filter(bs_mask, sigma, mode='constant', cval=1)
    return holo*bs_mask


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
    holo = propagate(holo, prop_l, ccd_dist = experimental_setup['ccd_dist'], energy = experimental_setup['energy'], integer_wl_multiple = integer_wl_multiple, px_size = experimental_setup['px_size']) 
    return reconstruct(holo)

###########################################################################################

#                                   PHASE SHIFTS                                          #

###########################################################################################

def global_phase_shift(holo, phi):
    '''
    multiply the hologram with a global phase
    
    Parameters
    ----------
    holo : array
        input hologram
    phi: scalar
        phase to multiply to the hologram
    
    Returns
    -------
    holo_phase: array
        phase shifted hologram
    -------
    author: KG 2020
    '''
    return holo*np.exp(1j*phi)


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

