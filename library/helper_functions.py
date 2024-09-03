"""
Python library for some general functions for paths, data processing, ...

2022-24
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import sys, os
import fnmatch

import numpy as np

import matplotlib.pyplot as plt
from wand.image import Image

# Scipy
import scipy.constants

#=====================
#Paths and directories
#=====================
def create_folder(folder):
    '''
    Creates input folder if necessary    
    '''
    
    if not(os.path.exists(folder)):
        print("Creating folder " + folder)
        os.makedirs(folder)
    #else:
    #    print('Folder: %s already exists!'%folder)
    return folder

def list_files(directory):
    '''
    List all files in directory
    '''
    return sorted(os.listdir(directory))


def list_files_excluding_pattern(directory, pattern):
    '''
    List all files in directory that do not have the pattern
    '''
    excluded_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not fnmatch.fnmatch(file, pattern):
                excluded_files.append(os.path.join(root, file))

    return sorted(excluded_files)

#======================
#Physics
#======================
#photon energy - wavelength converter
def photon_energy_wavelength(value, input_unit = 'eV'):
    '''
    Converts photon energy to wavelength and vice versa
    
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


#=========================
#Image processing
#=========================
#Pad 2d or 3d images
def padding(image):
    
    '''
    Padding of arrays to get square shape
    
    Parameter
    =========
    image : 2d or 3d array
        array with non square shape in last two dimensions
        
    Output
    ======
    image: 2d or 3d array
        padded array 
    ======
    author: ck 2023
    '''
    
    if image.shape[-1] != image.shape[-2]:
        print("Padding...")
        pad = image.shape[-2] - image.shape[-1]
        
        #2d case
        if image.ndim == 2:
            if pad < 0:
                image = np.pad(image, ((0, -pad), (0, 0)))
            elif pad > 0 :
                image = np.pad(image, ((0, 0), (0, pad)))
        #3d case
        elif image.ndim == 3:
            if pad < 0:
                image = np.pad(image, ((0, 0),(0, -pad), (0, 0)))
            elif pad > 0:
                image = np.pad(image, ((0, 0),(0, 0), (0, pad)))
        #4d case
        elif image.ndim == 4:
            if pad < 0:
                image = np.pad(image, ((0, 0),(0, 0),(0, -pad), (0, 0)))
            elif pad > 0:
                image = np.pad(image, ((0, 0),(0, 0),(0, 0), (0, pad)))
        else:
            raise TypeError
    #else: 
    #    print('No padding needed!')
    
    return image

# Remove infs and Nans from arrays
def fill_infs_nans(array,fill_value = 0):
    '''
    Remove infs and Nans from arrays
    
    Parameter
    =========
    array : numpy array
        array with non square shape in last two dimensions
    fill_value : scalar
        fill infs and nans with this value
        
    Output
    ======
    array : numpy array
        array with filled entries 
    nans_array : numpy array
        all of these elements
    ======
    author: ck 2023
    '''
    
    # Find infs and nans
    nans_array = np.logical_or(np.isnan(array), np.isinf(array))
    
    # Replace them
    array[nans_array] = fill_value
    
    return array, nans_array 


def binning(array, binning_factor):
    """
    Bins images: new_shape = old_shape/binning_factor

    Parameter
    =========
    array : numpy array ndim > 1
        last two dimension will be binned
    binning_factor : int
        new_shape = old_shape/binning_factor for last two dimensions

    Output
    ======
    new_array : numpy array
        binned array
    ======
    author: ck 2023/24

    """

    # Only if binning factor is relevant
    if binning_factor != 1:
        if array.ndim >= 2:
            # New output shape
            new_shape = np.array(array.shape)
            new_shape[-2:] = (new_shape[-2:] // binning_factor).astype(int)

            # Reshape by adding additional dimensions
            shape = new_shape.copy()
            shape = np.insert(shape, -1, array.shape[-1] // new_shape[-1])
            shape = np.append(shape, array.shape[-2] // new_shape[-2]).astype(int)

            new_array = array.reshape(shape).mean(-1).mean(-2)
            return new_array
        else:
            raise ValueError(
                f"Dimension mismatch: input array must have at least 2 dimension"
            )

    elif binning_factor == 1:
        return array

def make_square_shape(images):
    """Crops last two dimenions of n-d images to square shape"""
    crop = np.min(np.array([images.shape[-2], images.shape[-1]]))
    images = images[..., :crop, :crop]

    return images

def drop_inhomogenous_part(image_list):
    """
    Drops inhomogenous_parts of list of image stacks, i.e, drops length
    of image stacks to smallest share for creation of numpy array
    """
    # Find length of all image_list stacks
    length = np.array([im.shape[0] for im in image_list])
    max_stack_size = np.min(length)

    # Check if array is inhomogenous
    if np.all(length == max_stack_size) == False:
        print("Dropping inhomogenous part of array")
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:max_stack_size]

    return image_list

def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
    array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max((r.shape[0]) for r in M)
    arr_shape = (M[0].shape[-1], M[0].shape[-2])

    Z = np.zeros((len(M), maxlen, arr_shape[0], arr_shape[1]))
    for enu, row in enumerate(M):
        Z[enu, : len(row)] += row
    return Z


def reshape_arrays(arrays, dimensions):
    """
    Reshape all listed arrays according to dimensions
    """
    for i, array in enumerate(arrays):
        array = np.reshape(array, dimensions)
        arrays[i] = array
    return arrays

# =========================
# Other
# =========================

def create_gif(image_filenames, output_gif,fps = 1,loop = 0):
    '''
    creates gif from list of images or folder
    
    Parameter
    =========
    image_filenames : str or list of str
        path or list of paths (files and/or folder)
    output_gif : str
        output filename
    fps : scalar
        framerate
    loop : scalar
        nr of repetitions (0 => inf)
        
    Output
    ======
        gif in specified path
    ======
    author: ck 2024
    
    '''
    
    images = []

    #Check if input is list or str
    if isinstance(image_filenames,str):
        image_filenames = [image_filenames]

    with Image() as img:
        # Load all specified images
        for path in image_filenames:
            if os.path.isfile(path):
                #Load single image
                with Image(filename=path) as frame:
                    img.sequence.append(frame)
            elif os.path.isdir(path):
                # Load images from the input folder
                for filename in sorted(os.listdir(path)):
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        filepath = os.path.join(path, filename)
                        with Image(filename=path) as frame:
                            img.sequence.append(frame)
            else:
                print(f"{path} does not exist or is neither a file nor a directory.")


        # Convert fps to duration per frames
        for frame in img.sequence:
            frame.delay = int(100 / fps)  # Set the duration (adjust as needed)
            
        # Save the images as a GIF
        img.save(filename=output_gif)


def flatten_list(nested_list):
    """
    Convert nested list into single list
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


# =========================
# Plotting
# =========================

def quick_plot(data,**kwargs):
    """
    Quick plotting without much formatting
    """
    
    # Create figure
    fig, ax = plt.subplots(**kwargs)
    
    # Multiple plots arranged in list
    if isinstance(data,list):
        for i, single_data in enumerate(data):
            ax.plot(single_data,label = i)
            ax.legend()
    # Single plot
    else:
        ax.plot(data)
        
    return fig, ax


def hls_to_rgb(hls_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being HLS colours.
    Returns an array of same size, each row being RGB colours.
    Like `colorsys` python module, all values are between 0 and 1.

    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL

    from: https://gist.github.com/reinhrst/2d693a16c04861a8fbc5253938312410
    """

    y, x, z = hls_array.shape
    hls_array = np.reshape(hls_array, (y * x, z))

    ONE_THIRD = 1 / 3
    TWO_THIRD = 2 / 3
    ONE_SIXTH = 1 / 6

    def _v(m1, m2, h):
        h = h % 1.0
        return np.where(
            h < ONE_SIXTH,
            m1 + (m2 - m1) * h * 6,
            np.where(
                h < 0.5,
                m2,
                np.where(h < TWO_THIRD, m1 + (m2 - m1) * (TWO_THIRD - h) * 6, m1),
            ),
        )

    assert hls_array.ndim == 2
    assert hls_array.shape[1] == 3
    assert np.max(hls_array) <= 1
    assert np.min(hls_array) >= 0

    h, l, s = hls_array.T.reshape((3, -1, 1))
    m2 = np.where(l < 0.5, l * (1 + s), l + s - (l * s))
    m1 = 2 * l - m2

    r = np.where(s == 0, l, _v(m1, m2, h + ONE_THIRD))
    g = np.where(s == 0, l, _v(m1, m2, h))
    b = np.where(s == 0, l, _v(m1, m2, h - ONE_THIRD))

    rgb = np.reshape(np.concatenate((r, g, b), axis=1), (y, x, z))
    return rgb


def complex_to_color(array, abs_range=[0, 100]):
    # In HLS color system: hue (color), lightness, saturation
    # Angle should represent color
    hue = (np.angle(array) + np.pi) / (2 * np.pi)

    # Lightness
    lightness = np.abs(array)
    vmin, vmax = np.percentile(lightness, abs_range)
    lightness = (lightness - vmin) / (vmax - vmin)

    # Clip to color range
    lightness = np.clip(lightness, a_min=0, a_max=1)

    # Saturation
    saturation = 1 * np.ones(array.shape)

    return hls_to_rgb(np.stack((hue, lightness, saturation), axis=2))