"""
Python Dictionary for FTH reconstructions

2016/2019/2020
@authors:   MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            FB: Felix Buettner (felix.buettner@helmholtz-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import configparser as cp
import matplotlib.pyplot as plt
import pandas as pd
import h5py

#reading matlap files
import scipy.io as sio
import scipy.constants as cst

from skimage.draw import disk as circle
from scipy import stats

###########################################################################################

#                               LOAD DATA                                                 #

###########################################################################################


def load_topo_RB2(pos, topo, auto_factor=True):
    '''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: arrays, images of positive and negative helicity
            auto_factor: optional, boolean, determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5 (defualt is False)
    OUTPUT: pos and neg offsetted and averaged
    --------
    author: RB 2021
    '''
    offset_pos = np.percentile(pos,1)
    pos = pos - offset_pos
    factor = np.sum(np.multiply(pos,topo))/np.sum(np.multiply(topo, topo))
    print('offset_pos,factor_pos = ',offset_pos,factor)
    return (pos - factor * topo, factor)

def load_both_RB(pos, neg, auto_factor=True):
    '''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: arrays, images of positive and negative helicity
            auto_factor: optional, boolean, determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5 (defualt is False)
    OUTPUT: pos,neg with factor and offset
    --------
    author: RB2021
    '''
    size = pos.shape
    if auto_factor:
        offset_pos = (np.mean(pos[:10,:10]) + np.mean(pos[-10:,:10]) + np.mean(pos[:10,-10:]) + np.mean(pos[-10:,-10:]))/4
        offset_neg = (np.mean(neg[:10,:10]) + np.mean(neg[-10:,:10]) + np.mean(neg[:10,-10:]) + np.mean(neg[-10:,-10:]))/4
        pos = pos - offset_pos
        neg = neg - offset_neg
        topo = pos+neg
        factor = np.sum(np.multiply(pos,topo))/np.sum(np.multiply(topo, topo))
    else:
        topo = pos + neg
        factor = 0.5
    print('Auto factor = ' + str(factor))

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]], factor * topo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (pos[:size[1], :], factor * topo[:size[1], :], factor)
    else:
        return (pos, factor/(1-factor)*neg, factor)

def load_both(pos, neg, auto_factor=False):
    '''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: arrays, images of positive and negative helicity
            auto_factor: optional, boolean, determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5 (defualt is False)
    OUTPUT: difference hologram and factor as a tuple
    --------
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

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]] - factor * topo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (pos[:size[1], :] - factor * topo[:size[1], :], factor)
    else:
        return (pos - factor * topo, factor)
    
def load_single(image, topo, helicity, auto_factor=False):
    '''
    Load image for a single helicity reconstruction
    INPUT:  image: array, data of the single helicity image
            topo: array, topography data
            helicity: boolean, True/False for pos/neg helicity image
            auto_factor: optional, boolean, determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 0.5 (defualt is False)
    OUTPUT: difference hologram and factor as a tuple
    --------
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
    if size[0]<size[1]:
        return (holo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (holo[:size[1], :], factor)
    else:
        return (holo, factor)
'''
    


def load_both(pos, neg, auto_factor=False):
    ''''''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: arrays, images of positive and negative helicity
            auto_factor: optional, boolean, determine the factor by which neg is normalized automatically, if FALSE: factor is set to 1  and intercept to 0(defualt is False)
    OUTPUT: (pos, neg, intercept, slope) as a tuple
    --------
    author: RB 2021
    ''''''
    size = pos.shape
    
    if auto_factor:
        
        x=(pos).flatten()
        y=(neg).flatten()
        res = stats.linregress(x,y)
        intercept, slope= res.intercept,res.slope
        print("neg=%0.3f + %0.3f*pos"%(intercept, slope))
        neg=(neg-intercept)/slope

    else:
        intercept, slope= 0,1

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]], neg[:, :size[0]], intercept, slope)
    elif size[0]>size[1]:
        return (pos[:size[1], :], neg[:size[1], :], intercept, slope)
    else:
        return (pos, neg, intercept, slope)

def load_single(image, topo, helicity, auto_factor=False):
    ''''''
    Load image for a single helicity reconstruction
    INPUT:  image: array, data of the single helicity image
            topo: array, topography data
            helicity: boolean, True/False for pos/neg helicity image
            auto_factor: optional, boolean, determine the factor by which neg is normalized automatically, if FALSE: factor is set to 1  and intercept to 0(defualt is False)
    OUTPUT: (pos, neg, intercept, slope) as a tuple
    --------
    author: RB 2021 2019
    ''''''
    #load the reference for topology
    topo = topo.astype(np.int64)
    #load the single image
    image = image.astype(np.int64)

    size = image.shape
    
    if auto_factor:
        
        if helicity:
            pos=image.copy()
            x=(pos).flatten()
            y=(topo-image).flatten()
            res = stats.linregress(x,y)
            intercept, slope= res.intercept,res.slope
            print("neg=%0.3f + %0.3f*pos"%(intercept, slope))
            neg=((topo-image)-intercept)/slope
            
        else:
            neg=image.copy()
            x=(topo-image).flatten()
            y=(neg).flatten()
            res = stats.linregress(y,x)
            intercept, slope= res.intercept,res.slope
            print("pos=%0.3f + %0.3f*pos"%(intercept, slope))
            pos=((topo-image)-intercept)/slope
        
    else:
        intercept, slope= 0,1


    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]] , neg[:, :size[0]] , intercept, slope)
    elif size[0]>size[1]:
        return (pos[:size[1], :] ,neg[:size[1], :] , intercept, slope)
    else:
        return (pos, neg , intercept, slope)
    
'''


def load_mat(folder, npos):
    '''
    load the reconstruction file from the matlab routine, needed only for the beamtimes, where we reconstructed with MATLAB (04.19, 05.19, 09.19)
    we now also have a complete python script for the reconstruction, so this function is no longer crucially needed
    INPUT:  folder: string, path to the matplab parameter file
            npos: int, number of the matlab file
    OUTPUT: center coordinates and beamstop diameter as a tuple
    --------
    author: KG 2019
    '''
    rec_params = sio.loadmat(folder + 'holo_%04d.mat'%npos)
    center = rec_params['middle'][0]
    beamstop = rec_params['bmask']
    bs_diam = np.max(np.append(np.sum(beamstop, axis=1), np.sum(beamstop, axis=0)))
    print('Loaded matlab file ' + folder + 'holo_%04d.mat'%npos)
    return (center, bs_diam)


###########################################################################################

#                               PLOTTING                                                  #

###########################################################################################

def plot(image, scale = (2,98), color = 'gray', colorbar = True):
    '''
    plot the image with the given scale, colormap and with a colorbar
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
    --------
    author: KG 2019
    '''
    mi, ma = np.percentile(image[ROI], scale)
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(image[ROI]), vmin = mi, vmax = ma, cmap = color)
    if colorbar:
        plt.colorbar()
    return

###########################################################################################

#                               COSMIC RAYS                                               #

###########################################################################################
 
def remove_cosmic_ray(holo, coordinates):
    """
    Replaces a single pixel by the mean of the 8 nearest neighbors.
    INPUT:  holo: array, hologram
            coordinates: array, coordinates of the pixel to be replaced in an array or list [x, y]
    OUTPUT: hologram with replaced pixel
    -------
    author: KG 2019
    """
    x = coordinates[0]
    y = coordinates[1]
    avg = 0
    for i in (x-1, x, x+1):
        for j in (y-1, y, y+1):
            if not np.logical_and(i == x, j == y):
                avg += holo[j, i]
    holo[y, x] = avg/8
    return holo

def remove_two(holo, x_coord, y_coord):
    """
    Replaces two neighboring pixels by the mean of the nearest neighbors.
    INPUT:  holo: array, hologram
            x_coord: int or array, x coordinates of the pixel to be replaced in an array or list [x1, x2] if there are two pixels in x direction or as a single number if the pixels have the same x coordinate
            y_coord: int or array, y coordinates of the pixel (see above)
    OUTPUT: hologram with replaced pixels
    -------
    author: KG 2019
    """
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)
    try:
        if x_coord.shape[0] == 2:
            holo = fth.remove_cosmic_ray(holo, [x_coord[0], y_coord])
            holo = fth.remove_cosmic_ray(holo, [x_coord[1], y_coord])
            holo = fth.remove_cosmic_ray(holo, [x_coord[0], y_coord])
    except:
        try:
            if y_coord.shape[0] == 2:
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[0]])
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[1]])
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[0]])
        except:
            print("No cosmic rays removed! Input two pixel!")
    return holo
    
    
def eliminateCosmicRays(image,
                        minDeviationInMultiplesOfSigma = 8, 
                        cellsize = 64, 
                        minAbsolutDeviation = 100
                        ):
    '''
    Definition of cosmic rays in several steps:

     1) devide the original image in a complete overlay of 
        (cellsize)x(cellsize) pixel subimages -> set1

     2) perform a second devision, where the cells are shifted by 
        (cellsize/2)x(cellsize/2) pixel -> set2,
        such that the middle of each cell of set2 is a corner of a cell 
        of set1 and vice versa. Now, every pixel (except for the one of
        the outer (cellsize/2) pixel shell)
        are in exactly one cell of set1 and one cell of set2

     3) for each cell of set1 and set2, calculate the average and
        standard deviation

     4) all pixels exceeding average +- minDev * sigma are potentially
        cosmic rays

     5) Define a 3rd set, which contains all intersections of cells of 
        set1 and set2 plus the outer frame of cellsize/2 pixels which are 
        exclusively in set1.
        Pixels in this cell will be replaced by the cell's avergae 
        provided that both parent
        cells rate this pixel as a cosmic ray.

     6) Repeat the procedure until no pixels are identified as cosmic
        rays.

     7) minAbsolutDeviation defines how much the hot pixels need to be
        above of below the average intensity in set1 or set2 in order to
        be counted as a cosmic ray. Should be set to two photon counts
        to avoid elimination of data in sections where only very few
        photons are found.
        
        
    Parameters
    ---------
    image : Numpy array of MxN pixels
        Hologram to be filtered
    
    minDeviationInMultiplesOfSigma : float
        Threshold, in units of standard deviation, to idenify a cosmic ray.
    
    cellsize : int
        Size of tiles in set1. Should be a divisor of each dimension of image.
    
    minAbsolutDeviation : float
        Absolute threshold to identify a cosmic ray.
    
    Returns
    ---------
    Filtered copy of image.
    
    -----
    author: MS/FB, 2016
    '''
    # Don't edit the provided image
    image = image.copy()
    n = cellsize
    minDev = minDeviationInMultiplesOfSigma
    minAbsDev = minAbsolutDeviation
  
    numberOfIdentifiedCosmicRays = 20
    totalNumberOfIdentifiedCosmicRays = 0
    numberOfIterations = 0
  
    nx,ny = image.shape[:2]
  
    while numberOfIdentifiedCosmicRays > 3:
        avSet1,stdSet1 = average_over_n_nearest_pixels_2D(
                            image, n, True)
        avSet2,stdSet2 = average_over_n_nearest_pixels_2D(
                            image[n/2:-n/2,n/2:-n/2], n, True)
        # Determine the upper and lower limits beyond which a pixel intensity
        # defines a hot pixel, individually for each pixel in image.
        ULSet1 = np.maximum(avSet1 + minDev * stdSet1,
                            avSet1 + minAbsDev)
        LLSet1 = np.minimum(avSet1 - minDev * stdSet1,
                            avSet1 - minAbsDev)
        ULSet2 = np.maximum(avSet2 + minDev * stdSet2,
                            avSet2 + minAbsDev)
        LLSet2 = np.minimum(avSet2 - minDev * stdSet2,
                            avSet2 - minAbsDev)
        # Make the UL and LL matrices as large as the original image.
        # Where set2 does not have own values, values from set1 are used.
        ULSet1_large = ULSet1.repeat(n,axis=0).repeat(n,axis=1)
        LLSet1_large = LLSet1.repeat(n,axis=0).repeat(n,axis=1)
        ULSet2_large = ULSet1_large.copy()
        LLSet2_large = LLSet1_large.copy()
        ULSet2_large[n/2:nx-n/2,n/2:ny-n/2] = ULSet2.repeat(n,axis=0).repeat(n,axis=1)
        LLSet2_large[n/2:nx-n/2,n/2:ny-n/2] = LLSet2.repeat(n,axis=0).repeat(n,axis=1)

        UL = np.maximum(ULSet1_large, ULSet2_large)
        LL = np.minimum(LLSet1_large, LLSet2_large)

        replace = np.logical_or(image<LL,image>UL)
        # Replace pixels by the average in set1
        replaceBy = avSet1.repeat(n,axis=0).repeat(n,axis=1)

        image[replace] = replaceBy[replace]
        numberOfIdentifiedCosmicRays = replace.sum()
        totalNumberOfIdentifiedCosmicRays += numberOfIdentifiedCosmicRays
        numberOfIterations += 1
        print('Replaced {} ({} in total) cosmic rays in {} iterations'.format(
              numberOfIdentifiedCosmicRays,
              totalNumberOfIdentifiedCosmicRays,
              numberOfIterations))
    return image
 


# %%
def average_over_n_nearest_pixels_2D(M,n,returnStdDev=False):
    '''
    Split the numpy array M in local groups of nxn pixels (along the x and
    y axes). Take the average of each group. Return the result.
  
    Parameters
    ----------
    M : numpy array of at least 2 dimensions
        Magnetization pattern
    n : float > 0
        If n is not an integer or if the shape of M cannot be divided by n,
        then the size of the local groups may vary by +-1.
    returnStdDev : bool
        If set to True, the standard deviation for each average will be returned
        as a second parameter.
  
    Returns
    -------
    An array where the x and y dimensions are by a factor n smaller than
    in the input array.
    
    -----
    author: MS/FB 2016
    '''
    # To test / visualize the following code, copy and play with this:
    ## Simulate a matrix with x and y dimensions and a 3-vector in
    ## each cell
    #t = np.array(range(420)).reshape((10,14,3))
    #nx = 2
    #ny = 7
    ## Set the x-component of each vector to 1000 to check that is
    ## does not get mixed with the y and z component in the procedure
    #t[...,0] = 1000
    ## Split t in 5x2 submatrices = 10/2 x 14/7
    #s = np.array(np.array_split(np.array(np.array_split(t,ny,axis=1)),nx,axis=1))
    ## t[0:5,0:2,...] is obtained from s[0,0]
    ## t[5:10,0:2,...] is obtained from s[1,0]
    ## Flatten the submatrices. Calculate the new shape depending on the number
    ## of dimensions of t
    #newShape = s.shape
    #if( len(s.shape) > 4 ):
    #  newShape = s.shape[0:2] + (-1,) + s.shape[4:]
    #else:
    #  newShape = s.shape[0:2] + (-1,)
    #sf = s.reshape(newShape)
    ## Compare tsf[1,0] and ts[1,0] to see the effect of the previous commands.
    ## They work for arbitrary shapes of t.
    ## Check that tsf[...,0] is still 1000 everywhere.
    ## Now averages (with stdDev) over the second axis are possible.
    if(n<1):
        n=1
    nx = max(1,int(np.ceil(M.shape[0]/float(n))))
    ny = max(1,int(np.ceil(M.shape[1]/float(n))))
    # pad M with NaN values to make the dimensions an integer multiple of nx and ny
    # NaN values will be ignored in the averaging
    shape=M.shape
    newshape = (int(nx*n),int(ny*n))+shape[2:]
    Mn = np.ones(newshape)*np.NaN
    Mn[(newshape[0]-shape[0])/2:shape[0]+(newshape[0]-shape[0])/2,
       (newshape[1]-shape[1])/2:shape[1]+(newshape[1]-shape[1])/2,
       ...] = M
    # Subdivide the array Mn in x and y direction.
    s = np.array(np.array_split(np.array(np.array_split(Mn,ny,axis=1)),nx,axis=1))
    newShape = s.shape
    if( len(s.shape) > 4 ):
        newShape = s.shape[0:2] + (-1,) + s.shape[4:]
    else:
        newShape = s.shape[0:2] + (-1,)
    sf = s.reshape(newShape)
    # Mask all NaN elements for the averaging
    sf = np.ma.MaskedArray(sf, mask=np.isnan(sf))
    # If requested, calculate the standard deviation.
    if( returnStdDev ):
        return np.average(sf,axis=2),np.std(sf,axis=2)
    return np.average(sf,axis=2)

###########################################################################################

#                               RECONSTRUCTION                                            #

###########################################################################################

def reconstruct(image):
    '''
    Reconstruct the image by fft
    -------
    author: MS 2016
    '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image)))


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
    '''return the rounded integer (if you cast a number as int, it will floor the number)'''
    return np.int(np.round(n))

def set_center(image, center):
    '''
    this centering routine shifts the image in a cyclical fashion
    INPUT:  image: array, difference hologram
            center: array, center coordinates [x, y]
    OUTPUT: centered hologram
    -------
    author: MS 2016, KG 2019
    '''
    xdim, ydim = image.shape
    xshift = integer(xdim / 2 - center[1])
    yshift = integer(ydim / 2 - center[0])
    image_shift = np.roll(image, yshift, axis=0)
    image_shift = np.roll(image_shift, xshift, axis=1)
    print('Shifted image by %i pixels in x and %i pixels in y.'%(xshift, yshift))
    return image_shift

def sub_pixel_centering(reco, dx, dy):
    '''
    Routine for subpixel centering
    INPUT:  reco :  array, the reconstructed image
            dx, dy: floats, amount to be shifted
    RETURNS: shifted hologram
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

def mask_beamstop(image, bs_size, sigma = 3, center = None):
    '''
    A smoothed circular region of the imput image is set to zero.
    INPUT:  image: array, the difference hologram
            bs_size: integer, diameter of the beamstop
            sigma: optional, float, the sigma of the applied gaussian filter (default is 3)
            center: optional, array, if the hologram is not centered, you can input the center coordinates for the beamstop mask. Default is None, so the center of the picture is taken.
    OUTPUT: hologram multiplied with the beamstop mask
    -------
    author: MS 2016, KG 2019
    '''

    #Save the center of the beamstop. If none is given, take the center of the image.
    if center is None:
        x0, y0 = [integer(c/2) for c in image.shape]
    else:
        x0, y0 = [integer(c) for c in center]

    #create the beamstop mask using scikit-image's circle function
    bs_mask = np.zeros(image.shape)
    yy, xx = circle((x0, y0), bs_size/2)
    bs_mask[yy, xx] = 1
    bs_mask = np.logical_not(bs_mask).astype(np.float64)
    #smooth the mask with a gaussion filter    
    bs_mask = gaussian_filter(bs_mask, sigma, mode='constant', cval=1)
    return image*bs_mask

def mask_beamstop_matlab(image, mask, sigma=8):
    '''
    If a binary mask the size of the image is given, use this function. Not used in the current reconstruction scripts...
    '''
    if np.logical_not(mask[0,0]): #if the outside is 0 and the beamstop is one
        mask = np.logical_not(mask).astype(np.float64)
    bs_mask = gaussian_filter(mask, sigma, mode='constant', cval=1)
    return image*bs_mask

###########################################################################################

#                                 PROPAGATION                                             #

###########################################################################################

def propagate(holo, prop_l, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, integer_wl_multiple=True):
    '''
    Parameters:
    ===========
    holo : array, hologram  to be propagated
    prop_l : float, propagation distance [m]
    experimental_setup : optional, dictionary, {CCD - sample distance [m] (default is 18e-2 [m]), photon energy [eV] (default is 779.5 [eV]), physical size of one pixel of the CCD [m] (default is 20e-6 [m])}
    integer_wl_mult : optional, boolean, if true, coerce propagation distance to nearest integermultiple of photon wavelength (default is True)
    
    Returns:
    ========
    holo : propagated hologram
    
    ========
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
    holo = np.exp(1j * phase) * holo

    #print ('Propagation distance: %.2fum' % (prop_l*1e6)) 
    return holo

def propagate_realspace(image, prop_l, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, integer_wl_multiple=True):
    '''
    Parameters:
    ===========
    image : array, real space image to be propagated
    prop_l : propagation distance [m]
    ccd_dist : CCD - sample distance [m]
    energy : photon energy [eV] 
    integer_wl_mult : if true, coerce propagation distance to nearest integermultiple of photon wavelength 
    
    Returns:
    ========
    image : propagated image
    
    ========
    author: KG 2020
    '''
    holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    holo = propagate(holo, prop_l, experimental_setup=experimental_setup, integer_wl_multiple = integer_wl_multiple) 
    return reconstruct(holo)

###########################################################################################

#                                   PHASE SHIFTS                                          #

###########################################################################################

def global_phase_shift(holo, phi):
    '''
    multiply the hologram with a global phase
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
    A : float
        ampltitude of the Gauss, please input a positive number because -A is taken as factor for the Gauss
    sigma: float
        sigma of the Gauss
    
    Returns
    -------
    data * HP : array
        given data multiplied with the Gauss high pass filter
    -------
    author: KG 2020
    '''
    x0, y0 = [s//2 for s in data.shape]
    x,y = np.mgrid[-x0:x0, -y0:y0]
    HP = 1 - amplitude * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return (data * HP, HP)


###########################################################################################

#                                  CONFIG FILES                                           #

###########################################################################################

def save_reco_dict_to_hdf(fname, reco_dict):
    '''Saves a flat dictionary to a new hdf group in given file.
    
    Parameters
    ----------
    fname : str
        hdf file name
    reco_dict : dict
        Flat dictionary
    
    Returns
    -------
    grp : str
        Name of the new data group.
    -------
    author: MS 2020
    '''
    with h5py.File(fname, mode='a') as f:
        i = 0
        while f'reco{i:02d}' in f:
            i += 1
        for k, v in reco_dict.items():
            f[f'reco{i:02d}/{k}'] = v
    return f'reco{i:02d}'
    

def save_config(image_numbers, center_coordinates, bs_size, prop_dist, phase_shift, roi_coordinates, conf_filename):
    '''
    save the reconstruction parameters in a config file with configparser, replaced in 2020 with saving as hdf file
    INPUT:
        image_numbers: 1D list, with either one or two values: [single_hel_number] or [pos, neg]
        center_coordinates: 1D array, with two values: [xcenter, ycenter]
        bs_size: float, the diameter of the beamstop
        prop_dist: float, the propagation length
        phase_shift: float, phase shift 
        roi_coordinates: 1D array, with four values: [xstart, xstop, ystart, ystop]
        conf_filename: string, the name and path under which the configfile should be saved
        
    -------
    author: KG 2019
    '''
    def list_to_str(mylist): #configparser can only read and write strings
        str_list = str()
        for i in mylist:
            str_list += str(i) + ' '
        return str_list

    #create the config data
    config = cp.ConfigParser()
    config['Image']={}
    config['Image']['Numbers'] = list_to_str(image_numbers)
    config['Center'] = {}
    config['Center']['Coordinates'] = list_to_str(center_coordinates)
    config['Beamstop'] = {}
    config['Beamstop']['Size'] = str(bs_size)
    config['Propagation'] = {}
    config['Propagation']['Distance'] = str(prop_dist)
    config['Phase Shift'] = {}
    config['Phase Shift']['phase'] = str(phase_shift)
    config['ROI'] = {}
    config['ROI']['Coordinates'] = list_to_str(roi_coordinates)

    print('Save Config file ' + conf_filename)
    #write the file
    with open(conf_filename, 'w') as configfile:
        config.write(configfile)
    return

def save_config_matlab(image_numbers, center_coordinates,  prop_dist, phase_shift, roi_coordinates, conf_filename):
    '''
    save the reconstruction parameters in a config file with configparser, not in use anymore
    PARAMETERS:
        image_numbers: 1D list, with either one or two values: [single_hel_number] or [pos, neg]
        center_coordinates: 1D array, with two values: [xcenter, ycenter]
        prop_dist: float, the propagation length
        phase_shift: float, the phase shift
        roi_coordinates: 1D array, with four values: [xstart, xstop, ystart, ystop]
        conf_filename: string, the name and path under which the configfile should be saved
    -------
    author: KG 2019
        '''
    def list_to_str(mylist): #configparser can only read and write strings
        str_list = str()
        for i in mylist:
            str_list += str(i) + ' '
        return str_list

    #create the config data
    config = cp.ConfigParser()
    config['Image']={}
    config['Image']['Numbers'] = list_to_str(image_numbers)
    config['Center'] = {}
    config['Center']['Coordinates'] = list_to_str(center_coordinates)
    config['Propagation'] = {}
    config['Propagation']['Distance'] = str(prop_dist)
    config['Phase Shift'] = {}
    config['Phase Shift']['phase'] = str(phase_shift)
    config['ROI'] = {}
    config['ROI']['Coordinates'] = list_to_str(roi_coordinates)

    print('Save Config file ' + conf_filename)
    #write the file
    with open(conf_filename, 'w') as configfile:
        config.write(configfile)
    return

def read_hdf(fname):
    '''
    reads the latest saved parameters in the hdf file
    INPUT:  fname: str, path and filename of the hdf file
    OUTPUT: image numbers, topography numbers, factor, center coordinates, beamstop diameter, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    i = 0
    while f'reco{i:02d}' in f:
        i += 1
    i -= 1
    
    image_numbers = f[f'reco{i:02d}/image numbers'][()]
    topo_numbers = f[f'reco{i:02d}/topo numbers'][()]
    #factor = f[f'reco{i:02d}/factor'][()]
    center = f[f'reco{i:02d}/center'][()]
    bs_diam = f[f'reco{i:02d}/beamstop diameter'][()]
    prop_dist = f[f'reco{i:02d}/Propagation distance'][()]
    phase = f[f'reco{i:02d}/phase'][()]
    dx = f[f'reco{i:02d}/dx'][()]
    dy = f[f'reco{i:02d}/dy'][()]
    roi = f[f'reco{i:02d}/ROI coordinates'][()]

    return (image_numbers, topo_numbers, center, bs_diam, prop_dist, phase, roi,dx,dy)


def read_config(conf_filename):
    '''read data from config file created with configparser, replaced by hdf files
    INPUT:  conf_filename: str, the name and path under which the configfile is saved
    OUTPUT: image numbers, center coordinates, beamstop diameter, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2019
    '''
    def str_to_list(mystr):#configparser can only read and write strings
        def append_string(a,b):
            if np.isnan(b):
                a.append(np.nan)
            else:
                a.append(int(b))
            return a

        mylist = []
        tmp = str()
        for i in mystr:
            if i == ' ':
                mylist = append_string(mylist, float(tmp))
                tmp=str()
            else:
                tmp += i
        mylist = append_string(mylist, float(tmp))
        return mylist

    print('Read Config file ' + conf_filename)
    #read the config file
    conf= cp.ConfigParser()
    conf.read(conf_filename)

    #save the parameters
    image_numbers = str_to_list(conf['Image']['Numbers'])
    center = str_to_list(conf['Center']['Coordinates'])
    bs_diam = np.float(conf['Beamstop']['Size'])
    prop_dist = np.float(conf['Propagation']['Distance'])
    phase = np.float(conf['Phase Shift']['phase'])
    roi = str_to_list(conf['ROI']['Coordinates'])

    return (image_numbers, center, bs_diam, prop_dist, phase, roi)

def read_config_matlab(conf_filename):
    '''
    read data from config file created with configparser, replaced by hdf files
    INPUT:  conf_filename: str, the name and path under which the configfile is saved
    OUTPUT: image numbers, center coordinates, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2019
    '''
    def str_to_list(mystr):#configparser can only read and write strings
        def append_string(a,b):
            if np.isnan(b):
                a.append(np.nan)
            else:
                a.append(int(b))
            return a

        mylist = []
        tmp = str()
        for i in mystr:
            if i == ' ':
                mylist = append_string(mylist, float(tmp))
                print(mylist)
                tmp=str()
            else:
                tmp += i
        mylist = append_string(mylist, float(tmp))
        return mylist

    print('Read Config file ' + conf_filename)
    #read the config file
    conf= cp.ConfigParser()
    conf.read(conf_filename)

    #save the parameters
    image_numbers = str_to_list(conf['Image']['Numbers'])
    center = str_to_list(conf['Center']['Coordinates'])
    prop_dist = np.float(conf['Propagation']['Distance'])
    phase = np.float(conf['Phase Shift']['phase'])
    roi = str_to_list(conf['ROI']['Coordinates'])


    return (image_numbers, center, prop_dist, phase, roi)
