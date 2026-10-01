"""
Python library for loading HDF5/NeXus data of different facilities
(MAXI chamber from MBI, PETRA III MaxP04, MAX IV SoftiMAX).

All facilities share the same loading functions, only the hdf5 entries
(mnemonics) differ. Select them with load_mnemonics(facility).
SwissFEL data uses a different file structure, see Swiss_FEL_Loading.py

2024-26
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

# Commonly used hdf5 entries. Facility and nexus file structure specific
MNEMONICS = dict()

# MAX IV, SoftiMAX
MNEMONICS["MAXIV"] = {
    # tree
    "measurement": "measurement",
    "pre_scan_snapshot": "measurement/pre_scan_snapshot",
    "pre_scan": "snapshots/pre_scan",
    "post_scan": "snapshots/post_scan",
    
    # Camera related
    "ccd": "instrument/picam/data",
    "exposure_time": "instrument/picam/exposure",
    "pixel_format": "instrument/picam/frame_shape",
    
    # instruments
    "diode": "measurement/aem_eb01_01_ch1",
    
    # snapshots
    "energy": "measurement/pre_scan_snapshot/beamline_energy",
    "det_dist": "snapshots/post_scan/detectorz",
    "pre_energy": "snapshots/pre_scan/beamline_energy",
}

# MAXI chamber from MBI
MNEMONICS["MAXI"] = {
    "measurement": "measurement",
    "ccd": "measurement/ccd2",
    "images": "ccd2",  # key inside mnemonics["measurement"]
    "pre_scan_snapshot": "measurement/pre_scan_snapshot",
    "energy": "measurement/pre_scan_snapshot/energy",
    "helicity": "measurement/pre_scan_snapshot/helicity",
    "magOOP": "measurement/pre_scan_snapshot/magOOP",
    "magIP": "measurement/pre_scan_snapshot/magIP",
    "cmos": "measurement/cmossoftimax",
    "sample_rotation": "measurement/pre_scan_snapshot/srotz",
    "diode_software": "measurement/adc2sw",
    "diode": "measurement/adc2",
    "cmos_images": "/entry_0000/MAXI/sCMOS/data",
    "mono": "measurement/mono",
}

# PETRA III, MaxP04
MNEMONICS["PETRA"] = {
    "images": "ccd",
    "magnet_mT": "/scan/data/m_caena",
    "magnet_A": "/scan/data/m_magnetA",
    "data": "/scan/data",
    "collection": "/scan/instrument/collection",
    "energy": "/scan/instrument/mono/energy",
    "marana": "measurement/m_marana",
    "measurement": "/scan/instrument/collection",
    "helicity": "measurement/pre_scan_snapshot/und_shift",
    "nx_marana": "/entry/instrument/detector/data",
    "framerate": "/entry/instrument/detector/framerate",
    "temperature": "/scan/data/cryoin4",
    "diode": "/scan/data/adc_beck_femto_diodemax",
    "magnet": "/scan/instrument/collection/m_caena",
}

##########################################################################


def load_mnemonics(facility):
    """
    Return mnemonics dictionary of the given facility

    Parameter
    =========
    facility : str
        "MAXI", "PETRA" or "MAXIV"

    Output
    ======
    mnemonics : dict
        copy of the facility mnemonics, i.e., beamtime specific changes in
        the notebook do not change the library defaults
    ======
    author: ck 2026
    """

    if facility not in MNEMONICS:
        raise ValueError(
            f"Unknown facility '{facility}'. Allowed: {sorted(MNEMONICS)}"
        )

    return dict(MNEMONICS[facility])


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
    elif isinstance(scan_nr, np.generic):
        scan_nr = "%05d" % scan_nr

    # Combine all inputs
    filename = join(raw_folder, file_prefix + scan_nr + file_format)

    return filename


# Load any kind of data from measurements
def load_data(fname, keypath, keys=None):
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
