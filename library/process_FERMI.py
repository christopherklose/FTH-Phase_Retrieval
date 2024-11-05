"""
Created on Tue, Jun 30 2020
Updated Nov 2021

FERMI specific helper functions

@author:
    Michael Schneider, MBI Berlin
    Kathinka Gerlinger, MBI Berlin
"""


import numpy as np
import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, split, getmtime
import xarray as xr

# commonly used hdf5 entries
mnemonics = dict(
    xgm_UH='photon_diagnostics/FEL01/I0_monitor/iom_uh_a',
    xgm_SH='photon_diagnostics/FEL01/I0_monitor/iom_sh_a',
    filter_UH='photon_diagnostics/UH_Filter/Instrument',
    filter_seed='photon_diagnostics/Seed_Filter/Instrument',
    filter_EH_in='photon_diagnostics/Filters/EH_filter_IN',
    filter_UH_in='photon_diagnostics/Filters/UH_filter_IN',
    wavelength='photon_source/FEL02/wavelength',
    diode_I0='PAM/FQPDIa',
    diode_sum='PAM/FQPDSum',
    polarization='photon_source/FEL01/polarization_status',
    harmonic='photon_source/FEL01/harmonic_number',
    gatt_pressure='photon_diagnostics/Gas_Attenuator/Pressure',
    gatt='DPI/Attenuator',
    image='image/ccd1',
    image_full_rate='image/tucam_full_rate',
    image_decimate='image/tucam_decimate',
    delay='DPI/DelayLine',
    delay_ACDC='photon_diagnostics/Delay_Line/Delay',
    alignz='DPI/AlignZ',
    IR='Laser/Energy1',
    ccdz='DPI/CCDZ',
    ccd_rot='DPI/CCDTheta',
    samplez='DPI/SampleZ',
    valve_pos2='FEL/ValvePOS2',
    valve_dpi3='FEL/ValveDPI3',
    comment='ExperimentalComments',
)

def create_folder(folder):
    """
    Creates input folder if it does not exist yet
    """

    if not (os.path.exists(folder)):
        print("Creating folder " + folder)
        os.makedirs(folder)
    return folder

### glob files
def findfile(basefolder, membrane, bunchid):
    '''
    Finds all hdf5 files with the given bunchids in the folder.
    
    Parameters
    ==========
    basefolder : str
        basefolder of the sample
       
    membrane : str
        membrane name, folder where to find the files
       
    bunchid : list of int or str
        list of the bunchids
        
    Returns
    =======
    fname : list of str
        list of the complete paths of all hdf5 files of the given bunchids    
    '''
    pattern = '%s/rawdata/*%s.h5' % (membrane, bunchid)
    pattern = os.path.join(basefolder, pattern)    
    fname = glob.glob(pattern)
    if len(fname) == 0:
        print('no file found for BID %s on membrane %s!' % (bunchid, membrane))
        print(pattern)
        return
    if len(fname) > 1:
        print('found %d files for bunchid %s on membrane %s.'
              % (len(fname), bunchid, membrane))
        return fname
    return fname[-1]


def get_exp_dataframe(folder, recursive=True, keys={}):
    '''
    Loads basic experimental data from a list of hdf5 files. The images will NOT be loaded!
    
    Parameters
    ==========
    folder : str
        folder of the hdf5 files
       
    recursive : bool, optional (default True)
        wether to search for hdf5 files recursively in all subdirectories or not
       
    keys : dict, optional (default {})
        extra keys to load from the hdf5 files
        
    Returns
    =======
    exp : pd.DataFrame
        Returns a pandas Dataframe with the exp meta infos
    '''
    pattern = '**/*h5' if recursive else '*h5'
    flist = glob.glob(os.path.join(folder, pattern), recursive=recursive)
    if len(flist) == 0:
        raise FileNotFoundError('No files found using pattern ', pattern)
    exp = []
    exp_keys = mnemonics.copy()
    exp_keys.update(keys)
    skip_keys = ['image','image_full_rate', 'time']  # don't load these
    for h5file in flist:
        try:
            with h5py.File(h5file, mode='r') as h5:
                info = dict(filename=h5file)
                for k, v in exp_keys.items():
                    if 'time' in exp_keys:
                        info.update({'time': getmtime(h5file)})
                    if k not in skip_keys:
                        try:
                            val = h5[v][()]
                            val = val.decode() if isinstance(val, bytes) else val
                            info.update({k: val})
                        except KeyError:
                            pass
                            # info.update({k: np.nan})
                exp.append(info)
        except:
            raise
    exp = pd.DataFrame(exp)
    exp.filename = exp.filename.apply(str)
    return exp


def loadh5(filename, extra_keys={}, ccd=True, raise_on_error=False, roi=None):
    """
    Loads CCD image and bunch energy E_tot(µJ).

    Parameters
    ==========
    filename : str
        hdf file path/name

    extra_keys : list or dict, optional
        list of mnmemonics or hdf data paths

    ccd : boolean, optional (default True)
        whether to return the CCD image

    raise_on_error : bool, optional (default False)
        If True, skip over missing entries in hdf file.

    roi : slice_expression, optional
        If given, used to slice image data. Return full image array if None (default)

    Returns
    =======
    image : np.array
        the CCD image (only if ccd=True)

    meta : dict
        non-image data

    Notes
    =====
    FERMI's GMD is upstream of the last filter (seed filter). The shot-energy
    needs to be corrected for the filter transmission at the experiment's
    wavelength.
    """
    meta = {}
    if isinstance(extra_keys, list):
        extra_keys = {k: k for k in extra_keys}
    with h5py.File(filename, "r") as h5file:
        if ccd:
            roi = () if roi is None else roi
            image_keys = ["image", "image_full_rate", "image_decimate"]
            for key in image_keys:
                if mnemonics[key] in h5file:
                    image = h5file[mnemonics[key]][roi].astype(np.float16)
                    #print(f"Using image entry {key}.")
                    break
            else:
                errormsg = f"No image entry found {image_keys}"
                print(errormsg)
                if raise_on_error:
                    raise

        for k, v in extra_keys.items():
            try:
                h5path = mnemonics[v] if v in mnemonics else v
                meta.update({k: h5file[h5path][()]})
            except KeyError as exc:
                if raise_on_error:
                    raise
                else:
                    meta.update({k: np.nan})
                    print(f"Error reading {key}: {exc} - skipping.")
    if ccd:
        return image, meta
    else:
        return meta


def load_image_diodenorm(fname, dark, normkey = 'PAM/FQPDSum', full_rate=False):
    '''
    Loads CCD image and normalizes it to the given key.
    
    Parameters
    ==========
    fname : str
        hdf file path/name
       
    dark : array
        dark file
       
    normkey : str, optional (default 'PAM/FQPDSum')
        hdf5 key for which to normalize
        
    Returns
    =======
    im : np.array
        the normalized CCD image with dark substracted
    '''
    im, meta = loadh5(fname, extra_keys=[normkey], full_rate=full_rate)
    im = (im - dark)/np.sum(meta[normkey])
    return im
    
    
def zero_slope(data, chunksize = 500, max_slope = 4e-7):
    '''
    determines wether a data chunk has a slope lower than the given maximum.
    
    Parameters
    ==========
    data : array
        data array
       
    chunksize : int, optional (default 500)
        window over which the slope of data is determined
       
    max_slope : float, optional (default 4e-7)
        maximal slope to be still declared flat
        
    Returns
    =======
    is_plateau : np.array
        indices of data where the slope is below max_slope
    '''
    midindex = chunksize // 2
    is_plateau = np.zeros((data.shape[0]))
    for index in range(midindex, len(data) - midindex):
        chunk = data[index - midindex : index + midindex]
        # subtract the endpoints of the chunk
        # if not sufficient, maybe use a linear fit
        dx = chunksize
        dy = abs(chunk[0] - chunk[-1])
        if (-max_slope < dy / dx < max_slope):
            is_plateau[index] = 1
    return is_plateau

def get_magnet(fname, offset = .2, plot = True, chunksize = 500, max_slope = 4e-7):
    '''
    loads the magnet information
    
    Parameters
    ==========
    fname : str
        path to the magnet file you want to load
       
    offset : float, optional (default 0.2)
        offset to add to the mean magnet value
       
    plot : bool, optional (default True)
        wether to plot the magnet values
        
    chunksize : int, optional (default 500)
        parameter for zero_slope(); window over which the slope of data is determined
       
    max_slope : float, optional (default 4e-7)
        parameter for zero_slope(); maximal slope to be still declared flat
        
    Returns
    =======
    magnet : np.array
        all loaded magnet values
    mag_upper: float
        maximal magnet value
    mag_lower: float
        minimal magnet value
    '''
    scan = split(split(split(fname)[0])[0])[-1]
    sample = split(split(split(split(fname)[0])[0])[0])[-1]
    with h5py.File(fname, 'r') as f:
        magnet = f['Lecroy/Wave3'][()]
    mag_mean = np.mean(magnet, axis = 0) + offset
    mag_lower = np.mean(mag_mean[np.where(zero_slope(mag_mean, 500) == 1)])
    mag_upper = np.max(mag_mean)
    if plot:
        sname = split(fname)[-1]
        fig, ax = plt.subplots()
        ax.plot(mag_mean, label = 'mean')
        ax.plot(zero_slope(mag_mean, chunksize, max_slope), label = 'magnet change')
        ax.grid()
        ax.set_xlabel('time (a.u.)')
        ax.set_ylabel('magnet current (A)')
        ax.legend()
        ax.set_title(f'{sample}/{scan}' + ': max = %.1f, min = %.1f'%(mag_upper, mag_lower)) #include max and min of magnet current in title
        #fig.savefig(f'../images/Magnet/{sname}.png')
    return (magnet, mag_lower, mag_upper)
    
def filtering(folder, endings, recursive = True):
    '''
    filters a file list for given endings
    
    Parameters
    ==========
    folder : str
        folder to the files
       
    endings : str
        endings for which to filter the list
       
    recursive : bool, optional (default True)
        wether to search for hdf5 files recursively in all subdirectories or not
        
    Returns
    =======
    filtered : list
        list of all files in folder with the given endings (if recursive is True also in all subdirectories)
    '''
    flist = glob.glob(folder + '**/*h5', recursive=recursive)
    filtered = list(filter(lambda f: any([f.endswith(t) for t in endings]), flist))
    return filtered

def integrate_from_list(flist, ai, norm, mask=None, dark=None, labels=None, oneHz=False):
    az_int = []
    for f in flist:
        im, meta = loadh5(f, extra_keys={'diode_sum': 'PAM/FQPDSum',})
        im = im - dark if dark is not None else im
        mask = np.ones_like(im) if mask is None else mask
        norm_I0 = np.sum(meta['diode_sum']) if not oneHz else np.sum(meta['diode_sum'][::50])
        az_int.append(ai(im * mask) / norm / norm_I0)
    coords = {'theta': ai.theta}
    if labels is not None:
        coords.update({'labels': labels})
    da = xr.DataArray(az_int, dims=['labels', 'theta'], coords=coords)
    return da

def make_mask(image, center, rmin, beamstop=None, plot=True):
    '''
    creates a circular mask
    
    Parameters
    ==========
    image : array
        data image array
       
    center : list of int
        center position for the circular mask
       
    rmin : int
        radius of the circular mask
        
    beamstop: bool array of same dimension as image, optional (default None)
        position of data moints to be masked additionally
    
    plot: bool, optional (default True)
        wether to plot the image with the mask
        
    Returns
    =======
    mask : array
        boolean array that is 0 for all pixels that are masked
    '''
    mask = np.ones_like(image)
    if beamstop is not None:
        mask[beamstop[0][0]:beamstop[0][1], :] = 0
        mask[:, beamstop[1][0]:beamstop[1][1]] = 0
    ix, iy = np.ogrid[:image.shape[0], :image.shape[1]]
    dist = np.hypot(ix - center[0], iy - center[1])
    mask[dist < rmin] = 0
    if plot:
        #vmin, vmax = np.percentile(image, [2, 99])
        vmin, vmax = np.percentile(image, [20, 80])
        fig, ax = plt.subplots()
        m = ax.imshow(image * mask, vmin=vmin, vmax=vmax)
        for r in [600, 800, 1000]:
            ax.add_artist(plt.Circle((center[1], center[0]), r, fill=None, ec='red'))
        plt.colorbar(m)
    return mask
