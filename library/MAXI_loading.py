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


##########################################################################

# Commonly used hdf5 entries. MaxP04 nexus file structure specific
mnemonics = dict()
mnemonics["measurement"] = "measurement"
mnemonics["ccd"] = "measurement/ccd2"
mnemonics["pre_scan_snapshot"] = "measurement/pre_scan_snapshot"
mnemonics["energy"] = "measurement/pre_scan_snapshot/energy"
mnemonics["helicity"] = "measurement/pre_scan_snapshot/helicity"
mnemonics["magOOP"] = "measurement/pre_scan_snapshot/magOOP"
mnemonics["magIP"] = "measurement/pre_scan_snapshot/magIP"
mnemonics["cmos"] = "measurement/cmos"

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
    data : dict
        data dictionaray on single key
    ======
    author: ck 2024
    """
    
    with h5py.File(fname, "r") as f:
        # Get entry
        entry = str(list(f.keys())[0])
        
        # Create empty dictionary
        data = {}

        # Load keys from path
        data[key] = f[entry][key][()].squeeze()
        
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
    data = load_data(fname, mnemonics["measurement"], keys = [mnemonics["images"]])

    return data[mnemonics["images"]].squeeze()
