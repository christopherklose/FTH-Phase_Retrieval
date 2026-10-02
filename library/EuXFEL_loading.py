import sys, os
import time
from os.path import join
from os import path
from glob import glob
import h5py
import numpy as np 
import toolbox_scs as tb
import xarray as xr
from tqdm.auto import tqdm


##########################################################################

# Commonly used hdf5 entries. MaxP04 nexus file structure specific
mnemonics = dict()
mnemonics["images"] = "MTE3"
mnemonics["MTE3"] = "MTE3"
mnemonics["FFT_MCPraw"] = "FFT_MCPpeaks"
mnemonics["FFT_REFLraw"] = "FFT_MCPpeaks"
mnemonics["FFT_PD2raw"] = "FFT_PD2peaks"

mnemonics["AppleX"] = "AppleX"
mnemonics["transmission"] = "transmission"
mnemonics["transmission_col2"] = "transmission_col2"
mnemonics["PP800_HWP_POWER"] = "PP800_HWP_POWER"
mnemonics["PP800_SynchDelayLine"]="PP800_SynchDelayLine"
mnemonics["SCS_XGM"] = "SCS_SA3"
mnemonics["SCS_SA3"] = "SCS_SA3"
mnemonics["magnet"] = "magnet"
mnemonics["magnet_current"]="magnet_current"
mnemonics["energy"] = "nrj"


mnemonics["Delay"] = 'PP800_SynchDelayLine'
mnemonics["t0"] =  'PP800_T0_mm'

mnemonics["magnet_mT"] = "/scan/data/m_caena"
mnemonics["magnet_A"] = "/scan/data/m_magnetA"
mnemonics["data"] = "/scan/data"
mnemonics["collection"] = "/scan/instrument/collection"
mnemonics["nrj"] = "nrj"
mnemonics["marana"] = "measurement/m_marana"
mnemonics["measurement"] = "/scan/instrument/collection"
mnemonics["helicity"] = "measurement/pre_scan_snapshot/und_shift"
mnemonics["nx_marana"] = "/entry/instrument/detector/data"
mnemonics["framerate"] = "/entry/instrument/detector/framerate"
mnemonics["temperature"] = "/scan/data/cryoin4"

##########################################################################


def load_mnemonics():
    """Return mnemonics dictionary"""
    return mnemonics




# Load any kind of data from measurements
def load_data(proposal, run_nr, keys = ["MTE3"]):
    """
    Load data of all specified keys from keypath
    
    Parameter
    =========
    proposal : str
        proposal of beamtime
    run_nr : int
        index of data to load
    keys : str or list of strings
        keys to load 
        
    Output
    ======
    data : dict
        data dictionary of keys
    ======
    author: ck 2024, sw 2026
    """
    data = {}

    
    run, run_data = tb.load(proposal, run_nr, keys)
    
    
    if type(keys) is list:
        
        for key in keys:
            if key== "FFT_PD2raw":
                list_dat = []
                for dat in tqdm(run["SCS_FFT_DIAG/ADC/PD2:output", "data.rawData"].xarray()):
                    list_dat.append(np.average(dat))
                mnemonics_key = mnemonics[key]
                data[key] = np.array(list_dat)
            else:
                mnemonics_key = mnemonics[key]
                data[key]=run_data[mnemonics_key].values
             
    elif type(keys) is str:
        if keys== "FFT_PD2raw":
                list_dat = []
                for dat in tqdm(run["SCS_FFT_DIAG/ADC/PD2:output", "data.rawData"].xarray()):
                    list_dat.append(np.average(dat))
                data[keys] = np.array(list_dat)
        else:
            mnemonics_key = mnemonics[keys]
            data[keys]=run_data[mnemonics_key].values
        
    
    return data




# Load any kind of data from measurements
def load_key(proposal, run_nr, key):
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
    
    data = {}

    
    run, run_data = tb.load(proposal, run_nr, key)
    data[key]=run_data[key].values
        
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


