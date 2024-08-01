"""
Python Dictionary for FTH reconstructions

2016/2019/2020/2021/2024
@authors:   MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            FB: Felix Buettner (felix.buettner@helmholtz-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import xarray as xr
import extra_data as ed
import toolbox_scs as tb
import ipywidgets
import matplotlib.pyplot as plt

tb.mnemonics['mte'] = {
    'source': 'SCS_CDIDET_MTE3/CAM/CAMERA:daqOutput',
    'key': 'data.image.pixels',
    'dim': ['x', 'y']
}

tb.mnemonics['mte_temp_read'] = {
    'source': 'SCS_CDIDET_MTE3/CAM/CAMERA',
    'key': 'SensorTemperature.reading.value',
    'dim': []
}

tb.mnemonics['samZ'] = {
    'source': 'SCS_CDIFFT_MOV/MOTOR/SAM_Z',
    'key': 'actualPosition.value',
    'dim': []
}

def list_sources(run):
    '''Print a list of all control, instrument and detector sources'''
    print('CONTROL SOURCES')
    [print('\t- ', s) for s in run.control_sources]
    print('INSTRUMENT SOURCES')
    [print('\t- ', s) for s in run.instrument_sources]
    print('DETECTOR SOURCES')
    [print('\t- ', s) for s in run.detector_sources]


def load_rundata(proposal, run_nr, ccd=True, extra_keys=[], limit=None):
    '''Load commonly used data
    
    Paramters
    ---------
    proposal : int
        UPEX proposal number
    run_nr : int
        run number
    ccd : bool (default True)
        Whether to try and load CCD images. Will be returned as a separate
        DataArray to allow trainId alignment.
    extra_keys : list(str)
        Additional scs_toolbox.mnemonics to load
    limit : int, or None (default None)
        If not None, load at most <limit> trains
        
    Returns
    -------
    data : xr.Dataset
        The run data.
    images : xr.DataArray
        Only if load_mte=True
    '''
    keys_to_load = [
        'npulses_sase3', 'SCS_XGM', 'SCS_photonFlux', 'bunchpattern',
        'nrj', 'scannerX', 'scannerY', 'ESLIT', 'SCS_SA3', 'magnet'
    ]
    keys_to_load += extra_keys
    w = max([len(s) for s in keys_to_load])
    
    ds = xr.Dataset()
    run = tb.load(proposal, run_nr)
    if limit is not None:
        run = run.select_trains(ed.by_index[:limit])
    ds.attrs['run'] = run
    ds.attrs['run_nr'] = run_nr
    
    for k in keys_to_load:
        try:
            data = tb.get_array(run, k)
#             print(f'{k:>{w}s}: {data.nbytes / 1024**2:.3f} MB')
            ds[k] = data
        except Exception as ex:
            print(f'{k}: {ex}\n')
    
    if ccd:
        images = tb.get_array(run, 'mte')
        print(f'{"images":>{w}s}: {images.nbytes / 1024**2:.3f} MB')
        return ds, images
    return ds


def align_ccd_trains(ds, images, train_shift):
    '''Align image trainIds with other DAQ data and limit to common trains.
    
    Parameters
    ----------
    ds : xr.Dataset
        run data excluding CCD images
    images : xr.DataArray
        CCD images
    train_shift : int
        trainId offset between fast instruments and CCD due to read-out time
    -------
    authors: MS 2021
    '''
    images = images.assign_coords(trainId=(images.trainId + train_shift).astype(int))
    ds = ds.reindex_like(images)
    ds['mte'] = images
    return ds.dropna('trainId')

def axis_lim_to_roi(ax, xarray=False):
    '''Return a slice expression that corresponds to matplotlib axis limits.
    
    xarray: bool
        If True, return a dictionary with slice expressions for 'x' and 'y'
        dimensions instead of a numpy slice
    '''
    x0, x1 = sorted([int(x) for x in ax.get_xlim()])
    y0, y1 = sorted([int(y) for y in ax.get_ylim()])
    if xarray:
        return dict(x=slice(y0, y1), y=slice(x0, x1))
    return np.s_[y0:y1, x0:x1]


class ImageView(object):
    '''Interactive viewer for stacked image data.'''
    def __init__(self, images):
        self.images = images
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.images[0])
        ipywidgets.interact(
            self.update,
            index=range(self.images.shape[0]),
            vmin=ipywidgets.IntText(0),
            vmax=ipywidgets.IntText(2**16)
        )
    
    def update(self, vmin=0, vmax=2**16, index=0):
        try:
            self.ax.set_title(int(self.images.trainId[index]))
        except AttributeError:
            pass
        self.im.set_data(self.images[index])
        self.im.set_clim(vmin, vmax)