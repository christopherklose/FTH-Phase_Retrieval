"""
Python library for MaxP04 endstation at PETRA III with 2d image detector

2024
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import sys, os
import time
from os.path import join
from os import path
from glob import glob
import h5py


##########################################################################

# Commonly used hdf5 entries. MaxP04 nexus file structure specific
mnemonics = dict()
mnemonics["images"] = "ccd"
mnemonics["magnet_mT"] = "magnett_read"
mnemonics["data"] = "/scan/data"
mnemonics["collection"] = "/scan/instrument/collection"
mnemonics["energy"] = ""


##########################################################################

def load_mnemonics():
    """Return mnemonics dictionary"""
    return mnemonics

def list_data_files(folder, search_key="*"):
    """
    Returns a list of ALL data files in a folder that contain the search key
    
    Parameter
    =========
    folder : str
        search folder
    search_key : str
        searches files for additional key. Default: all files
        
    Output
    ======
    files : list
        list of searched filenames
    ======
    author: ck 2024
    """

    # Convert run number to string
    if type(search_key) == int:
        search_key = str(search_key)
    
    # Get sorted list of files in folder
    files = sorted(glob(join(folder, search_key)))

    return files
    

def generate_filename(raw_folder, file_prefix, file_format, scan_nr):
    """
    Generates filename of the given scan id
    
    Parameter
    =========
    raw_folder : str
        folder with raw data
    file_prefix : str
        prefix of filename
    file_format : str
        file format (ending, e.g. ".nxs")
    scan_nr : int or str
        number identifier (id) of the given scan
        
    Output
    ======
    filename : str
        full generated filename
    ======
    author: ck 2024
    """

    # Convert scan number to string
    if type(scan_nr) == int:
        scan_nr = "%05d" % scan_nr

    # Combine all inputs
    filename = join(raw_folder,file_prefix+scan_nr+file_format)
    
    return filename
    

# Load any kind of data from measurements
def load_data(fname, keypath, keys = None):
    """
    Load any kind of data from 
    
    Parameter
    =========
    fname : str
        filename of data file
    keypath : str
        path of nexus file tree to relevant data field
    keys : str or list of strings
        keys to load from keypath
        
    Output
    ======
    data : dict
        data dictionary of keys
    ======
    author: ck 2024
    """
    
    with h5py.File(fname, "r") as f:
        # Create empty dictionary
        data = {}

        # Load all keys of path
        if keys == None:
            for key in f[keypath].keys():
                data[key] = f[keypath][key][()]
        # Load only keys from key list
        elif isinstance(keys, list):
            for key in keys:
                data[key] = f[keypath][key][()]
        # Load only single key
        else:
            data[keys] = f[keypath][keys][()]

    return data

# Load any kind of data from measurements
def load_key(fname, key):
    """
    Load any kind of data from 
    
    Parameter
    =========
    fname : str
        filename of data file
    key : str
        key path of nexus file tree to relevant data field
   
    Output
    ======
    data : dict
        data dictionaray on single key
    ======
    author: ck 2024
    """
    
    with h5py.File(fname, "r") as f:
        # Create empty dictionary
        data = {}

        # Load keys from path
        data[key] = f[key][()]
        
    return data

# Load image files
def load_images(fname):
    """
    Load only image data
    
    Parameter
    =========
    fname : str
        filename of data file
    im_id : int
        experiment data identifier number
        
    Output
    ======
    images : array
        image data
    ======
    author: ck 2024
    """

    # Load only relevant image data
    data = load_data(fname, mnemonics["data"], keys = [mnemonics["images"]])

    return data[mnemonics["images"]].squeeze()
