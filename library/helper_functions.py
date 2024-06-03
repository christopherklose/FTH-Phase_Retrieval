"""
Python library for some general functions for paths, data processing, ...

2022/23
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

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
    '''
    Bins images: new_shape = old_shape/binning_factor
    
    Parameter
    =========
    array : numpy array
        input array
    binning_factor : int
        new_shape = old_shape/binning_factor
        
    Output
    ======
    new_array : numpy array
        binned array
    ======
    author: ck 2023
    
    '''
    new_shape = (np.array(array.shape) / binning_factor).astype(int)

    shape = (
        new_shape[0],
        array.shape[0] // new_shape[0],
        new_shape[1],
        array.shape[1] // new_shape[1],
    )
    
    new_array = array.reshape(shape).mean(-1).mean(1)
    return new_array

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
    if np.all(length == max_stack_size) is False:
        print("Dropping inhomogenous part of array")
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:max_stack_size]

    return image_list

#=========================
# Other
#=========================

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