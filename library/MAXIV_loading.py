"""
Python library for MAXI chamber from MBI

2024
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import sys, os
import time
from os.path import join
from os import path
from glob import glob
import h5py
import numpy as np 


##########################################################################

# Commonly used hdf5 entries. MAXI nexus file structure specific
mnemonics = dict()

# tree
mnemonics["measurement"] = "measurement"
mnemonics["pre_scan_snapshot"] = "measurement/pre_scan_snapshot"
mnemonics["pre_scan"] = "snapshots/pre_scan"
mnemonics["post_scan"] = "snapshots/post_scan"

# Camera related
mnemonics["ccd"] = "instrument/picam/data"
mnemonics["exposure_time"] = "instrument/picam/exposure"
mnemonics["pixel_format"] = "instrument/picam/frame_shape"

# instruments
mnemonics["diode"] = "measurement/aem_eb01_01_ch1"

# snapshots
mnemonics["energy"] = "measurement/pre_scan_snapshot/beamline_energy"
mnemonics["det_dist"] = "snapshots/post_scan/detectorz"

mnemonics["pre_energy"] = "snapshots/pre_scan/beamline_energy"


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
    

# Load any kind of data from measurements
def load_data(fname, keypath, keys = None):
    """
    Load data of all specified keys from keypath
    
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
        # Get entry
        entry = str(list(f.keys())[0])

        # Create empty dictionary
        data = {}

        # Load all keys of path
        if keys == None:
            for key in list(f[entry][keypath].keys()):
                try:
                    data[key] = f[entry][keypath][key][()].squeeze()
                except:
                    pass
        # Load only keys from key list
        elif isinstance(keys, list):
            for key in keys:
                try:
                    data[key] = f[entry][keypath][key][()].squeeze()
                except:
                    pass
        # Load only single key
        else:
            data[keys] = f[entry][keypath][keys][()].squeeze()

        return data

# Load any kind of data from measurements
def load_key(fname, key):
    """
    Load any kind of data specified by key (path)
    
    Parameter
    =========
    fname : str
        filename of data file
    key : str
        key path of nexus file tree to relevant data field
   
    Output
    ======
    data : array
        data of the given key
    ======
    author: ck 2024
    """
    
    with h5py.File(fname, "r") as f:
        # Get entry
        entry = str(list(f.keys())[0])

        # Load keys from path
        data = f[entry][key][()].squeeze()
        
    return data
