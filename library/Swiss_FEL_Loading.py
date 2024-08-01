"""
Python library for SwissFEL experiments with 2d image detector

2024
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
            SG: Simon Gaebel (Simon.Gaebel@mbi-berlin.de)
"""


import sys, os
import time
from os.path import join
from os import path
from glob import glob
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sfdata import SFDataFiles
import helper_functions as helper

import numpy as np


# Commonly used hdf5 entries. SwissFEL specific
mnemonics = dict()
mnemonics["Photon-Energy-PER-PULSE-AVG"] = "SATFE10-PEPG046:PHOTON-ENERGY-PER-PULSE-AVG"
mnemonics["images"] = "SATES20-HOLO-CAM01:FPICTURE"
mnemonics["I0_intensity_Ar"] = "SATES21-GES1:A2_VALUES"
mnemonics["Diode"] = "SATES21-GES1:A4_VALUES"
mnemonics["photonenergy"] = "SATOP11-OSGM087:photonenergy"

def load_mnemonics():
    """Return mnemonics dictionary"""
    return mnemonics


def frame_list_to_acq_and_idx(frame_list, stack_length):
    """
    Convert a continous list of frame index that span multiple acq stacks into
    its corresponding acq numbers and frames, e.g. for stack_length = 100:
    frame_list = [99,100,101] --> acq_nrs = [1,2], frame_index_list = [[99],[0,1]]
    
    Parameter
    =========
    frame_list : list 
        list of successive frames
    stack_length: int
        length of frame stack
        
    Output
    ======
    acq_nrs : array
        stack indexes
    frame_index_list : nested list
        frames corresponding to a given stack
    ======
    author: ck 2024
    """
    
    # Calc quotient of frame_list and acq stack length to find relevant acq idx
    acq_idx = np.array(np.divmod(frame_list, stack_length))

    # Setup lists
    acq_nrs = []
    frame_index_list = []

    # Group frames that correspond to identical acq_stackd to minimize data access
    for i in np.unique(acq_idx[0]):
        acq_nrs.append(i + 1)  # First stack is acq0001
        frame_idx = np.where(acq_idx[0] == i)
        frame_index_list.append(acq_idx[1][frame_idx])

    return np.array(acq_nrs), frame_index_list


def list_data_filenames(run_nr,BASEFOLDER,  search_key="*"):
    """
    Returns a list of ALL data files that correspond to the
    given run number and contain the search key
    
    Parameter
    =========
    run_nr : int 
        identifier of experiment run
    BASEFOLDER : int
        general beamtime folder
    search_key : str
        searches files for additional key. Default: all files
        
    Output
    ======
    files : list
        list of searched filenames
    folder : str
        folder of run nr
    ======
    author: ck 2024
    
    """

    # Convert run number to string
    if type(run_nr) == int:
        run_nr = "*%04d*" % run_nr

    # Find folder that corresponds to run number
    folder = glob(join(BASEFOLDER, "raw", run_nr))[0]
    print("Found folder: %s" % folder)

    # Get sorted list of files in folder
    files = sorted(glob(join(folder, "data", search_key)))

    return files, folder


def list_acquisition_filenames(run_nr,BASEFOLDER, acq_nrs=[], ONLY_CAMERA=False):
    """
    Returns a list of data files for the given acquisition
    numbers
    
    Parameter
    =========
    run_nr : int 
        identifier of experiment run
    BASEFOLDER : int
        general beamtime folder
    acq_nrs : iterable list, array
        searches files for additional key. Default: all files
    ONLY_CAMERA : bool
        exports only camera acquisition files names if True or all related
        h5 files if False
        
    Output
    ======
    fnames_flattened : list
        list of filenames of the given acquisition numbers
    ======
    author: ck 2024
    
    """

    # Load only camera files?
    if ONLY_CAMERA:
        search_key = "*CAMERAS.h5"
    else:
        search_key = "*"

    # If for run_nr is only gives a int number
    if type(run_nr) == int:
        run_nr = "*%04d*" % int(run_nr)
    elif type(run_nr) == float:
        run_nr = "*%04d*" % int(run_nr)

    # If list is empty all files are loaded, only specific acquisition nrs
    # otherwise
    if len(acq_nrs) == 0:
        fnames, _ = list_data_filenames(run_nr,BASEFOLDER, search_key=search_key)
    elif len(acq_nrs) > 0:
        _, folder = list_data_filenames(run_nr,BASEFOLDER,  search_key=search_key)

        fnames = []
        for acq_nr in acq_nrs:
            acq_str = f"*{acq_nr:04d}{search_key}"

            # Get filename pattern
            fname_pattern = join(
                folder,
                "data",
                acq_str,
            )

            fnames.append(glob(fname_pattern))

    # Flatten potential nested list
    fnames_flattened = helper.flatten_list(fnames)

    return fnames_flattened


def load_run(fnames, mnemonics):
    """
    Load all relevant data that are specified in the mnemonics dict (bugged)
    """

    data = dict()
    N = len(fnames)
    
    with SFDataFiles(fnames[0]) as f:
        for key in mnemonics.keys():
            try:
                data[key] = f[mnemonics[key]].data
            except:
                pass
    if N > 1:
        for fname in tqdm(fnames[1:]):
            with SFDataFiles(fname) as f:
                for key in mnemonics.keys():
                    try:
                        data[key] = np.concatenate((data[key], f[mnemonics[key]].data))
                    except:
                        pass
    
    return data


def load_images(fnames, loadmode="avg", n_jobs=1, crop=0):
    """
    Loads images from list of filenames.
    
    Parameter
    =========
    fnames : list
        list of image acquisition filenames
    loadmode : str
        "avg": return average over all frames of a given filenames
        "frames": return all single frames
    n_jobs : int
        number of jobs, i.e., available cpu threads
    crop : int
        crops images symmtrically by "crop" number of pixels
        
    Output
    ======
    images : array
        numpy array of images
    ======
    author: ck 2024    
    """

    # Setup
    images = []
    N = len(fnames)

    # Cropping necessary?
    if (crop == 0) or  (crop == None):
        slice_crop = slice(None)
    else:
        slice_crop = slice(crop, -crop)

    # Define loader function based on required loadmode
    if loadmode == "frames":

        def loader(fname):
            with SFDataFiles(fnames[0]) as f:
                image_stack = f[mnemonics["images"]][:, slice_crop, slice_crop].data
            return image_stack

    elif loadmode == "avg":

        def loader(fname):
            with SFDataFiles(fname) as f:
                image = np.mean(
                    f[mnemonics["images"]][:, slice_crop, slice_crop].data, axis=0
                )
            return image

    print(f"Start loading images with {n_jobs} parallel processes.")
    t0 = time.time()
    images = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(loader)(fname) for fname in fnames
    )
    print(f"Elapsed time: {time.time()-t0} seconds.")

    return np.array(helper.drop_inhomogenous_part(images))


def load_specific_frames(fnames, indexes, crop=None):
    """
    Load only specific frames for a list of acquisition filenames
    
    Parameter
    =========
    fnames : list
        list of image acquisition filenames
    indexes : nested list
        relevant frames indexes of a given fname
    crop : int or None
        crops images symmtrically by "crop" number of pixels [crop:-crop]
        
    Output
    ======
    images : array
        numpy array of all relevant single frame images
    ======
    author: ck 2024  
    """

    # Setup
    images = []

    # Cropping necessary?
    if (crop == 0) or  (crop == None):
        slice_crop = slice(None)
    else:
        slice_crop = slice(crop, -crop)

    # Loop over different fnames and indices
    for i, fname in enumerate(fnames):
        # Load only relevant frames from file
        with SFDataFiles(fname) as f:
            image = f[mnemonics["images"]][indexes[i], slice_crop, slice_crop].data
        images.append(image)

    return np.vstack(images)


# Full image loading procedure
def load_processing_frames(fnames, loadmode="avg", crop=0, frame_index_list=[],nr_jobs = 1):
    """
    Loads images, calc average over all images,
    padding to square shape, Additional cropping (optional)
    
    Parameter
    =========
    fnames : list
        list of image acquisition filenames
    loadmode : str
        "avg": return average over all frames of a given filenames
        "frames": return all single frames
    frame_index_list : nested list
        relevant frames indexes of a given fname
    crop : int
        crops images symmtrically by "crop" number of pixels
     n_jobs : int
        number of jobs, i.e., available cpu threads
        
    Output
    ======
    image : array
        average over all images
    images : array
        numpy array of all relevant single frame images
    ======
    author: ck 2024  
    """
    # Basic loading of stacks into list
    if not frame_index_list:
        images = load_images(fnames, loadmode, n_jobs=nr_jobs, crop=crop)
    else:
        images = load_specific_frames(fnames, frame_index_list, crop=crop)

    # Bring to square shape
    images = helper.make_square_shape(images)

    # Calculate mean
    if images.ndim > 2:
        image = np.mean(images, axis=0)
    else:
        image = images.copy()

    return image, images
