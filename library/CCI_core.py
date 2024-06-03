"""
Python library for CCI correlaion analysis

2022/23
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import sys, os
from os.path import join
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import itertools
import scipy as scp

import h5py

#Fitting
import scipy.optimize

#Filters
from scipy.ndimage.filters import gaussian_filter

#Image registration
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from dipy.align.transforms import AffineTransform2D, TranslationTransform2D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

#Progress bar
from tqdm.auto import tqdm

#Clustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances

#colormap
from matplotlib.colors import LinearSegmentedColormap

#======================
#Physics
#======================
#photon energy - wavelength converter
def photon_energy_wavelength(value, input_unit = 'eV'):
    '''
    Converts photon energy to wavelength and vice vera
    
    Parameter
    =========
    value : scalar
        input value either in eV or nm
    unit : string
        Select input unit. Currently either nm or eV is supported
        
    Output
    ======
    lambda_Xray or energy_Xray: scalar
        Converted unit  
    ======
    author: ck 2023
    '''    

    if input_unit == 'eV':
        lambda_Xray = scipy.constants.h*scipy.constants.c/(value*scipy.constants.e)
        return lambda_Xray
    elif input_unit == 'nm':
        energy_Xray = scipy.constants.h*scipy.constants.c/(value*10**(-9)*scipy.constants.e)
        return energy_Xray

#Draw circle mask
def circle_mask(shape,center,radius,sigma=None):

    '''
    Draws circle mask with option to apply gaussian filter for smoothing
    
    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter
        
    Output
    ======
    mask: array
        binary mask, or smoothed binary mask        
    ======
    author: ck 2022
    '''
    
    #setup array
    x = np.linspace(0,shape[1]-1,shape[1])
    y = np.linspace(0,shape[0]-1,shape[0])
    X,Y = np.meshgrid(x,y)

    # define circle
    mask = np.sqrt(((X-center[1])**2+(Y-center[0])**2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if sigma != None:
        mask = gaussian_filter(mask,sigma)
           
    return mask


def shift_image(image,shift):
    '''
    Shifts image with sub-pixel precission in Fourier space
    
    
    Parameters
    ----------
    image: array
        Moving image, will be shifted by shift vector
        
    shift: vector
        x and y translation in px
    
    Returns
    -------
    image_shifted: array
        Shifted image
    -------
    author: CK 2021
    '''
    
    #Shift Image
    shift_image = fourier_shift(scp.fft.fft2(image,workers=-1), shift)
    shift_image = scp.fft.ifft2(shift_image,workers=-1)
    shift_image = shift_image.real

    return shift_image


def shift_image_stack(image_stack,shift,chunk_sz = 'none'):
    '''
    Shifts all images of a stack with sub-pixel precission in Fourier space
    
    
    Parameters
    ----------
    image_stack: nr_images x dim1 x dim2 array
        Moving image stack, will be shifted by shift vector
        
    shift: nr_images x 2 array 
        x and y translation in px for each image
        
    chunk_sz: int
        nr of images per chunk, needed in case of large image arrays which might not fit into gpu memory
    
    Returns
    -------
    shifted_image_stack: array
        Shifted image stack
    -------
    author: CK 2023
    '''
    
    if image_stack.ndim == 2:
        print('Warning: This is only a single 2d image!')
        image_stack = shift_image(image_stack, shift)
    elif image_stack.ndim == 3:
        if chunk_sz == 'none':        
            #Shift Image
            for frame in tqdm(range(image_stack.shape[0])):
                image_stack[frame] = shift_image(image_stack[frame], shift[frame])
        else:       
            # Limits for Chunk Image stacks
            chunk_it = np.append(
                np.arange(0, np.ceil(image_stack.shape[0] / chunk_sz) * chunk_sz, chunk_sz), image_stack.shape[0]).astype(int)

            #Vary chunk
            print("Shifting images...")
            for ii in tqdm(range(len(chunk_it) - 1)):
                # Chunk data
                tmp_stack = image_stack[chunk_it[ii] : chunk_it[ii + 1]].copy()
                shift_stack = shift[chunk_it[ii] : chunk_it[ii + 1]].copy()

                ##Shift images
                #tmp_stack = shift_image(tmp_stack, shift_stack)

                ##Vary frames
                for frames in tqdm(range(tmp_stack.shape[0])):
                    #Calc correction
                    tmp_stack[frames] = shift_image(image_stack[frames],shift_stack[frames,:])

                # Assign to images
                image_stack[chunk_it[ii]:chunk_it[ii+1]] = tmp_stack.copy()
    
    return image_stack


def image_registration(image_unproccessed,image_background, method= "phase_cross_correlation",static_mask = None, moving_mask = None, roi=None,im_out=False):
    '''
    Aligns two images with sub-pixel precission through image registration
    
    
    Parameters
    ----------
    image_unproccessed: array
        Moving image, will be aligned with respect to image_background
        
    image_background: array
        static reference image
    
    static_mask: array
        ignore masked pixel in static image
        
    moving_mask: array
        ignore masked pixel in moving image
    
    roi: region of interest defining the region of the images used to calc
        the alignment
        
    im_out: bool
        return also shifted image if true
    
    Returns
    -------
    image_corrected: array
        Shifted/aligned moving image
    shift: array
        shift (dy,dx)
    -------
    author: CK 2022/23
    '''
    
    # Different method to calc image registration
    if method == "phase_cross_correlation":
        #Calculate Shift
        if roi == None:
            shift, error, diffphase = phase_cross_correlation(image_background, image_unproccessed, upsample_factor=100)
        else:  
            shift, error, diffphase = phase_cross_correlation(image_background[roi[2]:roi[3], image_unproccessed[roi[2]:roi[3], roi[0]:roi[1]], roi[0]:roi[1]], upsample_factor=100)
        
    elif method == "dipy":
        # Define your metric
        nbins = 32
        sampling_prop = None  # all pixels
        metric = MutualInformationMetric(nbins, sampling_prop)  # Gaussian pyramide

        # What is your transformation type?
        transform = TranslationTransform2D()

        # How many resolutions?
        level_iters = [10000, 1000, 100]

        # Smoothing of each level
        sigmas = [2.0, 1.0, 0.0]

        # Subsampling
        factors = [2, 1, 1]

        # Bring it together
        affreg = AffineRegistration(
            metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors
        )

        # Calc your transformation
        affine = affreg.optimize(
            image_background,
            image_unproccessed,
            transform,
            static_mask=static_mask,
            moving_mask=moving_mask,
            params0=None,
        )

        # Take only translation from affine transformation
        shift = np.array([affine.get_affine()[0, 2], affine.get_affine()[1, 2]])
        shift = np.round(shift, 2)
    
    if im_out == True:
        #Shift Image
        image_corrected = shift_image(image_unproccessed, shift)

        return image_corrected, shift
    else:
        return shift
    
    
#Fit (lin)
def func(x, a, b):
    return a*x+b
    
def dyn_factor(image,image_ref,method = 'scalarproduct', crop=0 ,plot = False, print_out = False):
    '''
    Calculates intensity normalization factor between images
    
    
    Parameters
    ----------
    image: array
        first image
        
    image_ref: array
        reference image
    
    method: str
        Method for calculating scaling factor (scalarproduct,correlation)
    
    crop : int
        crop array from each side for calc of factor and offset
    
    plot : bool
        Plot fit if method is correlation
        
    print_out : bool
        print factor and offset
    
    Returns
    -------
    factor: scalar
        Intensity correction factor
    -------
    author: CK 2023
    '''
    
    #Do you crop the images?
    crop_s = np.s_[crop:-1-crop,crop:-1-crop]
    
    if method == 'scalarproduct':
        factor = np.sum(image[crop_s]*image_ref[crop_s])/np.sum(image_ref[crop_s]*image_ref[crop_s])
        offset = 0
        
        if print_out == True:
            print(f'Intensity correction factor:', factor)
        
    elif method == 'correlation':
        #Create y, x data
        xdata = np.concatenate(image_ref[crop_s])
        ydata = np.concatenate(image[crop_s])
        
        #Ignore all x,y = 0 values, e.g., if a mask is used
        ignore = np.logical_or((xdata==0),(ydata==0))
        xdata = xdata[np.argwhere(ignore == False)]
        ydata = ydata[np.argwhere(ignore == False)]
        
        xdata = np.squeeze(xdata,axis=1)
        ydata = np.squeeze(ydata,axis=1)
        
        #Fitting
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        factor = popt[0]
        offset = popt[1]
        
        if print_out == True:
            print(f'Linear Fit: {factor:0.4f}*x + {offset:0.4f}')
        
        if plot == True:
            fig, ax = plt.subplots()
            ax.plot()
            ax.scatter(xdata, ydata, s= 5)
            ax.plot(xdata,func(xdata,*popt),'r-')
            ax.set_xlabel('Intensity Ref')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Linear Fit: {factor:0.4f}*x + {offset:0.4f}')
            plt.tight_layout()
            
    return factor, offset

def calc_diff_stack(images, topos, chunk_sz="none", method="scalarproduct",crop=0):
    """
    Calculates scaled differences between images and topos


    Parameters
    ----------
    images: nr_images x dim1 x dim2 array
        image stack

    topo: nr_images x dim1 x dim2 array
        reference images which will be subtracted after intensity normalization

    chunk_sz: int
        nr of images per chunk, needed in case of large image arrays which might not fit into gpu memory

    method: str
        Method for calculating scaling factor (scalarproduct,correlation)
    
    crop : int
        crop array from each side for calc of factor and offset

    Returns
    -------
    shifted_image_stack: array
        Shifted image stack
    -------
    author: CK 2023
    """

    if images.ndim == 2:
        print("Warning: This is only a single 2d image!")
        # Calc difference holo
        factor, offset = dyn_factor(
            images,
            topos,
            method=method,
            crop=crop,
            print_out=True,
            plot=False,
        )
        image_stack = images / factor - topos - offset

    elif images.ndim == 3:
        factor = np.zeros(images.shape[0])
        offset = np.zeros(images.shape[0])

        if chunk_sz == "none":
            # Shift Image                
            for frames in tqdm(range(images.shape[0])):                
                factor[frames], offset[frames] = dyn_factor(
                    images,
                    topos,
                    method=method,
                    crop=crop,
                    print_out=False,
                    plot=False,
                )
                images[frames] = (
                    images[frames] / factor[frames] - topos[frames] - offset[frames]
                )
            
        else:
            # Limits for Chunk Image stacks
            chunk_it = np.append(
                np.arange(0, np.ceil(images.shape[0] / chunk_sz) * chunk_sz, chunk_sz),
                images.shape[0],
            ).astype(int)

            # Vary chunk
            print("Shifting images...")
            for ii in tqdm(range(len(chunk_it) - 1)):
                # Chunk data and load into gpu
                image_stack = images[chunk_it[ii] : chunk_it[ii + 1]].copy()
                factor_stack = factor[chunk_it[ii] : chunk_it[ii + 1]].copy()
                offset_stack = offset[chunk_it[ii] : chunk_it[ii + 1]].copy()
                topo_stack = topos[chunk_it[ii] : chunk_it[ii + 1]].copy()

                # Vary frames
                for frames in tqdm(range(image_stack.shape[0])):
                    # Calc difference holo
                    factor_stack[frames], offset_stack[frames] = dyn_factor(
                        image_stack[frames],
                        topo_stack[frames],
                        method=method,
                        crop=crop,
                        print_out=False,
                        plot=False,
                    )
                    image_stack[frames] = (
                        image_stack[frames] / factor_stack[frames]
                        - topo_stack[frames]
                        - offset_stack[frames]
                    )

                # Assign to images                
                factor[chunk_it[ii] : chunk_it[ii + 1]] = factor_stack
                images[chunk_it[ii] : chunk_it[ii + 1]] = image_stack
                offset[chunk_it[ii] : chunk_it[ii + 1]] = offset_stack

    return images, factor, offset

def reconstruct(image):
    '''
    Reconstruct the image by inverse fft
    -------
    author: CK 2022
    '''
    return scp.fft.ifftshift(scp.fft.ifft2(scp.fft.fftshift(image)))


def FFT(image):
    '''
    Fourier transform
    -------
    author: CK 2022
    '''
    return scp.fft.fftshift(scp.fft.fft2(scp.fft.ifftshift(image)))


######## CCI specific #############

def parula_cmap():

    '''
    Matlab 'parula' colormap rgb values
    
    Parameter
    =========
    None
    
    
    Output
    ======
    cm_data = rgb colormap as list        
    ======
    author: ck 2022
    '''
    
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
     [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
     [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
      0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
     [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
      0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
     [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
      0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
     [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
      0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
     [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
      0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
     [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
      0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
      0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
     [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
      0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
     [0.0589714286, 0.6837571429, 0.7253857143], 
     [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
     [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
      0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
     [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
      0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
     [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
      0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
     [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
      0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
     [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
     [0.7184095238, 0.7411333333, 0.3904761905], 
     [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
      0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
     [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
     [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
      0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
     [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
      0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
     [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
     [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
     [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
      0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
     [0.9763, 0.9831, 0.0538]]

    return cm_data


def parula_map():
    
    '''
    Matlab 'parula' colormap as python colormap
    
    Parameter
    =========
    None
    
    
    Output
    ======
    parula: colormap as
    ======
    author: ck 2022
    '''
    
    cm_data = parula_cmap()
    parula = LinearSegmentedColormap.from_list('parula', cm_data)
    
    return parula


def filter_reference(holo,mask,settings):
    '''
    Filter reference-induced modulations from fth holograms
    
    Parameters
    ----------
    holo : numpy array
        input hologram
    mask: numpy array
        (smooth) mask to crop cross correlation in Patterson map
    settings: dict
        contains parameter for cropping
    
    Returns
    -------
    holo_filtered: numpy array
        reference-filtered "hologram"
    -------
    author: CK 2022
    '''
    
    diameter = settings['low_dia']
    
    #Transform to Patterson map
    tmp_array = reconstruct(holo)
    center = np.array(tmp_array.shape)/2
    
    #Crop Patterson map
    tmp_array = tmp_array[int(center[1]-diameter/2):int(center[1]+diameter/2+1),
                          int(center[0]-diameter/2):int(center[0]+diameter/2+1)]
    tmp_array = tmp_array*mask
    
    #Crop ROI of holograms
    tmp_array = FFT(tmp_array)
    holo_filtered = tmp_array.real

    return holo_filtered


#def seg_statistics(holo, Diameter, NrStd = 1, center = None, mask):
def seg_statistics(holo, mask, NrStd = 1):
    '''
    Creates mask that shows only value outside of a noise intervall defined by the statistics of the array
    
    Parameters
    ----------
    holo : array
        input hologram
    diameter: scalar
        diameter of highpass filter to calc noise level in outer areas
    NrStd: scalar, optional
        Multiplication factor of the standard deviation to count a pixel as noise. Default is 1.
    center: sequence of scalars, optional
        If given, the beamstop is masked at that position, otherwise the center of the image is taken. Default is None.
    mask : cupy array
        Predefined mask to calculate std and mean
    
    Returns
    -------
    statistics mask: cupy array
        bool mask of values larger than noise level
    -------
    author: CK 2022
    '''

    #Calc Statistics mask
    #if center is None:
    #    x0, y0 = [c/2 for c in holo.shape]
    #else:
    #    x0, y0 = [c for c in center]
    #
    #if mask is None:
    #    temp_mask = cp.zeros(holo.shape)
    #    yy, xx = circle(y0, x0, Diameter/2)
    #    temp_mask[yy, xx] = 1
    #else:
    temp_mask = mask
    
    temp = holo[temp_mask == 0]

    MEAN = np.mean(temp)
    STD  = np.std(temp)

    Statistics_mask = (np.abs(holo) >= MEAN + NrStd*STD)

    return Statistics_mask, MEAN, STD

def create_ring_mask(shape,center, radi):
    '''
    Creates mask that shows only value outside of a noise intervall defined by the statistics of the array
    
    Parameters
    ----------
    shape : int tuple
        shape of output arrays
    radi: list of int
        list of radi in px to create centered rings in q-space radi=[r1,r2,r3,...].
    
    Returns
    -------
    mask_circ: array
        2d array with labeled rings
    masks_ring: bool array
        3d array containing boolean masks for each ring
    -------
    author: CK 2023
    '''
    
    # Set up ring mask
    mask_circ = np.zeros(shape)

    # Create Ring mask
    for radius in radi:
        mask_circ = mask_circ + circle_mask(
            mask_circ.shape, center, radius, sigma="none"
        )

    mask_circ = np.abs(mask_circ - len(radi))
    mask_circ[mask_circ == len(radi)] = 0
    masks_ring = np.zeros((len(radi) - 1, shape[0], shape[1]), dtype=bool)
    for i in range(0, len(radi) - 1):
        masks_ring[i] = mask_circ == i + 1

    return mask_circ, masks_ring

def correlate_holograms(diff1, diff2, sum1, sum2, Statistics1, Statistics2):
    '''
    Function to determine the cross-correlation of two holograms.
    
    Parameters
    ----------
    diff1 : np array
        difference hologram of the first data image
    diff2 : np array
        difference hologram of the second data image
    sum1: np array
        sum hologram of the first data image
    sum2: np array 
        sum hologram of the first data image
    
    Returns
    -------
    c_val : scalar
        correlation value of the two holograms
    c_array: array
        pixelwise correlation array of the two holograms
    -------
    author: CK 2020-2021 / KG 2021
    '''    
    # replace all zeros in sum1/sum2 with another value to avoid infinities
    sum1[sum1 == 0] = 1e-8
    sum2[sum2 == 0] = 1e-8    
    
    # Combine Statistics Mask
    mask = np.logical_or(Statistics1,Statistics2).astype(float)
    
    # Calc flattened holos called scattering images
    S1 = diff1*mask/np.sqrt(sum1)
    S2 = diff2*mask/np.sqrt(sum2)
   
    # normalization Factor called scattering factor
    sf = np.sqrt(np.sum(S1 * S1)*np.sum(S2 * S2))
    
    # calculate the pixelwise correlation
    c_array = S1 * S2 / sf
    
    # average correlation
    c_val = np.sum(c_array)
    
    return (c_val, c_array)


def correlation_map(diff_holo_norm, statistics_mask):
    '''
    Function to determine the correlation of two holograms.
    
    Parameters
    ----------
    diff_holo_norm : d1xd2xd3 array (d1: nr holos, d2,d3: shape of single holo)
        array of all difference holograms (stack) normalized by corresponding topo holo
    statistics_mask : d1xd2xd3 array (d1: nr holos, d2,d3: shape of single holo)
        array for each image with bool mask of pixels with values larger than noise level (stack), must be of the same length as diff_holo_norm
    
    Returns
    -------
    corr_map : array
        correlation map where every image is correlatetd to each other image in the input
    -------
    author: CK 2022
    '''

    #predefine array
    corr_map = np.eye(diff_holo_norm.shape[0])

    #Varies first holo
    for ii in tqdm(range(diff_holo_norm.shape[0])):
        #Get holos and statistics of complete set for index ii
        holo_1 = diff_holo_norm[ii].astype('float32')
        statistics_1 = statistics_mask[ii].astype('float32')

        #Get holo mask for all other holos
        holo_2 = diff_holo_norm[ii+1:].astype('float32')

        #Get combined statistics mask
        mask = np.logical_or(statistics_1,statistics_mask[ii+1:])

        #Apply mask
        holo_1 = holo_1*mask
        holo_2 = holo_2*mask

        #normalization Factor
        sf = np.sqrt(np.sum(holo_1 * holo_1, axis = (1,2))*np.sum(holo_2 * holo_2, axis = (1,2)))

        #Correlation array
        corr_array = (holo_1*holo_2)/sf[:,None,None]
        corr_map[ii,ii+1:] = np.sum(corr_array,axis = (1,2))

    #Use symmetry to fill corr map
    corr_map = corr_map + np.rot90(np.fliplr(corr_map))
    corr_map = corr_map - np.eye(corr_map.shape[0])
        
    return corr_map


def correlation_map_fast(in_array):
    '''
    Function to determine the correlation of two holograms.
    
    Parameters
    ----------
    diff_holo_norm : d1xd2xd3 array (d1: nr holos, d2,d3: shape of single holo)
        array of scattering images (stack) normalized
    Returns
    -------
    corr_map : array
        correlation map where every image is correlatetd to each other image in the input
    -------
    author: CK 2022
    '''
    #If dimension is 3d
    if len(in_array.shape) == 3:
        in_array = in_array.reshape(in_array.shape[0], in_array.shape[1]*in_array.shape[2])

    #predefine array
    corr_map = np.zeros((in_array.shape[0],in_array.shape[0]))
    
    #Pure Multiplication
    corr_map_nonorm = np.dot(in_array, in_array.T) / in_array.shape[1] #Averaged value
    
    
    #Calc correlation function (Sutton)
    #Normalization 
    mean_counts = np.mean(in_array, axis=1)
    mean_counts[mean_counts <= 0] = 1 #Correction if average is 0 or below 0
    mean_counts = mean_counts.reshape(1, mean_counts.shape[0])
    norm = np.dot(mean_counts.T, mean_counts) #(nxn) Normalization array
    
    #Calc Corr
    corr_map_sutton = corr_map_nonorm/norm
    
    
    #Calc correlation function (Pearson)
    #Normalization
    cross_corr = np.diag(corr_map_nonorm)
    cross_corr = cross_corr.reshape(1,cross_corr.shape[0])
    norm = np.sqrt(np.dot(cross_corr.T, cross_corr)) #(nxn) Normalization array
    
    #Calc corr
    corr_map_pearson = corr_map_nonorm/norm

    
    return corr_map_nonorm, corr_map_pearson, corr_map_sutton


def reconstruct_correlation_map(frames,corr_array):
    '''
    Script Reconstruct the cluster's correlation map from the given
    cluster's 'frames' and the (large) correlation map of all
    frames.
    
    Parameters
    ----------
    frames: array
        relevant frames
    corr_array: array
        complete pair correlation map    
        
    Returns
    -------
    temp_core: array
        section of (large) correlation map defined by 'frames'
    -------
    author: CK 2021
    '''
    
    print(f'Reconstructing correlation map... (%d frames)'%len(frames))
    
    #Reshape frame array
    frames = np.reshape(frames,frames.shape[0])
    
    #Indexing of correlation array
    corr = corr_array[np.ix_(frames,frames)]
    
    #print('Reconstruction finished!')
    
    return corr


def create_linkage(cluster_idx,corr_array,metric='correlation',order = 1,plot = True):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
        
    Returns
    -------
    tlinkage: array
        clustering linkage array
    -------
    author: CK 2021
    '''
    #get colomap
    parula = parula_map()
    
    #Calc distance metric
    dist_metric = corr_array.copy()
    
    #Calculate higher orders of distance metrics
    if order > 0:
        for n in range(1,order+1):
            dist_metric = pairwise_distances(dist_metric, metric=metric,n_jobs = -1)
    
    #Calculate Linkage
    tlinkage = linkage(dist_metric,method='average',metric=metric)
    
    nr_cluster = 2
    temp_assignment = fcluster(tlinkage,nr_cluster,criterion='maxclust')
    
    #Output plots
    if plot is True:
        fig, _ = plt.subplots(figsize = (8,8))
        fig.suptitle(f'Cluster Index: {cluster_idx}')
    
        #Dist metric
        ax1 = plt.subplot(2,2,1)
        vmi, vma = np.percentile(dist_metric[dist_metric >= 1e-5],[1,99])
        ax1.imshow(dist_metric, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
        ax1.set_title('Distance metric')
        ax1.set_xlabel('Frame index k')
        ax1.set_ylabel('Frame index k')
    
        #Corr map
        ax2 = plt.subplot(2,2,2,sharex=ax1,sharey=ax1)
        vmi, vma = np.percentile(corr_array[corr_array <= 1-1e-5],[5,95])
        ax2.imshow(corr_array, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
        ax2.set_title('Correlation map')
        ax2.set_xlabel('Frame index k')
        ax2.set_ylabel('Frame index k')
        plt.gca().invert_yaxis()

        #Assignment plot
        ax3 = plt.subplot(2,2,3,sharex=ax1)
        ax3.plot(temp_assignment)
        ax3.set_title('Frame assignment')
        ax3.set_xlabel('Frame index k')
        ax3.set_ylabel('State')
        ax3.set_ylim((0.5,2.5))
        ax3.set_yticks([1,2])

        #Assignment plot
        ax4 = plt.subplot(2,2,4)
        dendrogram(tlinkage, p=100, truncate_mode = 'lastp')

        plt.tight_layout()
    
    return tlinkage, dist_metric

def cluster_hierarchical(tlinkage,parameter,clusteringOption='maxclust'):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    tlinkage: array
        clustering tlinkage array
    parameter: scalar
        parameter of clustering option, e.g., nr of clusters
    clusteringOption: string
        criterion used in forming flat clusters
        - 'inconsistent': cluster inconsistency threshold
        - 'maxcluster': number of total clusters
        
    Returns
    -------
    cluster_assignment: array
        assignment of frames to cluster
    -------
    author: CK 2021
    '''
    
    #Options
    if clusteringOption == 'maxclust':
        criterion_ = 'maxclust'
    elif clusteringOption == 'inconsistent':
        criterion_ = 'inconsistent'
    else:
        print('Error: clustering option not valid!')

    #Get cluster
    cluster_assignment = fcluster(tlinkage,parameter,criterion=criterion_)

    #Feedback
    nr = np.unique(cluster_assignment).shape[0]
    #print(f'{nr} clusters were constructed!')
    
    return cluster_assignment


def clustering_feedback(cluster_idx,nr,corr_array_large,corr_array_small,dist_metric_sq,tlinkage):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    nr: scalar
        index of subcluster
    corr_array_large: array
        initial pair correlation map
    corr_array_small: array
        pair correlation map of new subcluster
    dist_metric_sq: array
        distance metric of pair correlation map in square format
    tlinkage: array
        clustering linkage array
        
    Returns
    -------
    fig with plots
    -------
    author: CK 2021
    '''
    
    #get colomap
    parula = parula_map()
    
    #Output plots
    fig, _ = plt.subplots(figsize = (8,8))
    fig.suptitle(f'Cluster Index: {cluster_idx}-{nr}')
    
    #section of Initial Corr map
    ax1 = plt.subplot(2,2,1)
    vmi, vma = np.percentile(corr_array_large[corr_array_large != 1],[5,95])
    ax1.imshow(corr_array_large, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax1.set_title('Section initial correlation map')
    ax1.set_xlabel('Frame index k')
    ax1.set_ylabel('Frame index k')
    plt.gca().invert_yaxis()
    
    #section of Initial Corr map
    ax2 = plt.subplot(2,2,2)
    vmi, vma = np.percentile(corr_array_small[corr_array_small != 1],[5,95])
    ax2.imshow(corr_array_small, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax2.set_title('New correlation map')
    ax2.set_xlabel('Frame index k')
    ax2.set_ylabel('Frame index k')

    #Dist metric
    ax3 = plt.subplot(2,2,3,sharex=ax2,sharey=ax2)
    vmi, vma = np.percentile(dist_metric_sq[dist_metric_sq != 0],[1,99])
    ax3.imshow(dist_metric_sq, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax3.set_title('Distance metric')
    ax3.set_xlabel('Frame index k')
    ax3.set_ylabel('Frame index k')
    plt.gca().invert_yaxis()

    #Assignment plot
    ax4 = plt.subplot(2,2,4)
    dendrogram(tlinkage, p=150, truncate_mode = 'lastp')

    plt.tight_layout()
    return


def process_cluster(cluster, cluster_idx, corr_array, cluster_assignment, order = 1, metric = 'correlation', save=False, plot=True):
    '''
    processes a given cluster assignment and adds new subclusters to 'cluster'-list
    
    Parameters
    ----------
    cluster: list of dictionaries
        stores relevant data of clusters, e.g., assigned frames
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
    cluster_assignment: array
        assignment of frames to cluster
    save: bool
        save new subclusters in "cluster"-list and delete current cluster from list
    
    Returns
    -------
    cluster: list of dicts
        updated "cluster"-list
    -------
    author: CK 2022
    '''
    
    length = len(cluster)
    
    #Get initial frames in cluster
    frames = cluster[cluster_idx]['Cluster_Frames']
    frames = np.reshape(frames,frames.shape[0])
    
    #Get nr of new subclusters
    nr = np.unique(cluster_assignment)
    
    #Vary subclusters
    for ii in nr:
        print(f'Creating sub-cluster: {cluster_idx}-{ii}')
    
        #Get assignment
        tmp_assignment = np.argwhere(cluster_assignment == ii)
        tmp_assignment = np.reshape(tmp_assignment,tmp_assignment.shape[0])
        
        #Get subcluster correlation array
        tmp_corr_small = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        
        #Create mask which selects the section of the correlation that is assigned to sub-cluster ii
        tmp_mask = np.zeros([cluster_assignment.shape[0],cluster_assignment.shape[0]])
        tmp_mask[np.ix_(tmp_assignment,tmp_assignment)] = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        tmp_corr_large = tmp_mask
        
        #tmp_corr_large = np.zeros([cluster_assignment.shape[0],cluster_assignment.shape[0]])
        #tmp_corr_large[np.ix_(tmp_assignment,tmp_assignment)] = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        
        if plot is True:
            if len(tmp_assignment) > 1:
                #Calculate Linkage
                tlinkage, dist_metric = create_linkage(cluster_idx,corr_array,metric=metric,order = order,plot = False)
                #Plots
                clustering_feedback(cluster_idx,ii,tmp_corr_large,tmp_corr_small,dist_metric,tlinkage)
        #Save new cluster
        if save == True:
            print(f'Saving subcluster {cluster_idx}-{ii} as new cluster {length + ii}')
            cluster.append({"Cluster_Nr": length + ii,"Cluster_Frames": frames[np.ix_(tmp_assignment)]})
    
    #Del old cluster from 'cluster'-list
    if save == True:
        cluster[cluster_idx] = {}
                            
    return cluster


#### From Riccardo Battistelli 
"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction

2020
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

def create_hdf5(dict0,filename, extension=".hdf5"):
    
    f=createHDF5(dict0,filename, extension=extension)
    f.close()


def createHDF5(dict0,filename, extension=".hdf5",f=None):
    '''creates HDF5 data structures strating from a dictionary. supports nested dictionaries'''
#   print(dict0.keys())
    
#    try:
#        f = h5py.File(filename+ ".hdf5", "w")
#        print("ok")
#    except OSError:
#        print("could not read")
    
    if f==None:
         f = h5py.File(filename+ extension, "w")
    
    
    if type(dict0) == dict:
        
        for i in dict0.keys():
            
#            print("create group %s"%i)
#            print("---")
#            print(i,",",type(dict0[i]))

            if type(dict0[i]) == dict:
#                print('dict')
                grp=(f.create_group(i))
                createHDF5(dict0[i],filename,f=grp)
                
            elif type(dict0[i]) == np.ndarray:
                dset=(f.create_dataset(i, data=dict0[i]))
#                print("dataset created")
                
            elif (dict0[i] != None):
                dset=(f.create_dataset(i, data=dict0[i]))
#                print("dataset created")
#            print("---")
    return f

def read_hdf5(filename, extension=".hdf5", print_option=True):
    
    f = h5py.File(filename+extension, 'r')
    dict_output = readHDF5(f, print_option = print_option, extension=extension)
    
    return dict_output

def readHDF5(f, print_option=True, extension=".hdf5", dict_output={}):
    
    for i in f.keys():
        
    
        if type(f[i]) == h5py._hl.group.Group:
            if print_option==True:
                print("### ",i)
                print("---")
            dict_output[i]=readHDF5(f[i],print_option=print_option,dict_output={})
            if print_option==True:
                print("---")
        
        else:
            dict_output[i]=f[i][()]
            if print_option==True:
                print("•",i, "                  ", type(dict_output[i]))
        
        
    return dict_output