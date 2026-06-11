"""
Library with matplotlib widget for gui functions

@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
2022-2026
"""

import os

import numpy as np
import h5py

import scipy as scp
from scipy.ndimage import gaussian_filter, fourier_shift, rotate
from scipy.ndimage import shift as scipy_shift
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider
from scipy.constants import c, h, e
import scipy.constants as cst

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Ellipse
import ipywidgets
import ipywidgets as widgets

import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

import skimage.morphology
from dipy.segment.mask import median_otsu

from mask_lib import create_arc_supportmask, arc_mask


#########################################
# Helper functions
#########################################

#----------- Reconstructions -------------

def reconstruct(image):
    '''
    Reconstruct the image by inverse fft
    -------
    author: CK 2022
    '''
    return scp.fft.ifftshift(scp.fft.ifft2(scp.fft.fftshift(image),workers=os.cpu_count()))


def FFT(image):
    '''
    Fourier transform
    -------
    author: CK 2022
    '''
    return scp.fft.fftshift(scp.fft.fft2(scp.fft.ifftshift(image),workers=os.cpu_count()))


def propagate(holo, prop_l, experimental_setup, integer_wl_multiple=True):
    '''
    Propagate the hologram
    
    Parameters
    ----------
    holo : array
        input hologram
    prop_l: scalar
        distance of propagation in metre
    experimental_setup: dict
        experimental setup parameters in the following form: {'ccd_dist': [in metre], 'energy': [in eV], 'px_size': [in metre]}
    integer_wl_multiple: bool, optional
        Use a propagation, that is an integer multiple of the x-ray wave length, default is True.
    
    Returns
    -------
    prop_holo: array
        propagated hologram
    -------
    author: MS 2016
    '''
    wl = cst.h * cst.c / (experimental_setup['energy'] * cst.e)
    if integer_wl_multiple:
        prop_l = np.round(prop_l / wl) * wl

    l1, l2 = holo.shape
    q0, p0 = [s / 2 for s in holo.shape] # centre of the hologram
    q, p = np.mgrid[0:l1, 0:l2]  #grid over CCD pixel coordinates   
    pq_grid = (q - q0) ** 2 + (p - p0) ** 2 #grid over CCD pixel coordinates, (0,0) is the centre position
    dist_wl = 2 * prop_l * np.pi / wl
    phase = (dist_wl * np.sqrt(1 - (experimental_setup['px_size']/ experimental_setup['ccd_dist']) ** 2 * pq_grid))
    return np.exp(1j * phase) * holo


#----------- Masking ----------------
def circle_mask(shape,center,radius,sigma=None):

    '''
    Draws circle mask with option to apply gaussian filter for smoothing
    
    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter
        
    Output
    ======
    mask: array
        binary mask, or smoothed binary mask        
    ======
    author: ck 2022
    '''
    
    #setup array
    x = np.linspace(0,shape[1]-1,shape[1])
    y = np.linspace(0,shape[0]-1,shape[0])
    X,Y = np.meshgrid(x,y)

    # define circle
    mask = np.sqrt(((X-center[1])**2+(Y-center[0])**2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if np.logical_and(sigma != None,sigma != 0):
        mask = gaussian_filter(mask,sigma)
           
    return mask


#--------- Other utility -----------------------
def shift_image(image,shift,interpolation = True,out_dtype = 'numpy'):
    '''
    Shifts image with sub-pixel precission in Fourier space
    
    
    Parameters
    ----------
    image: array
        Moving image, will be shifted by shift vector
        
    shift: vector
        x and y translation in px
    
    Returns
    -------
    image_shifted: cupy/numpy array
        Shifted image
    -------
    author: CK 2023
    '''
    
    #Shift Image
    if interpolation is True:
        shifted_image = scipy_shift(image,shift,mode = 'reflect')
    else:
        shifted_image = fourier_shift(scp.fft.fft2(image), shift)
        shifted_image = scp.fft.ifft2(shifted_image)
    
    return shifted_image



###########################################
#             Interactive
###########################################


def cimshow(im, title = None, **kwargs):
    """Simple 2d image plot with adjustable contrast.
    
    Returns matplotlib figure and axis created.
    """
    im = np.array(im).astype("float")
    fig, ax = plt.subplots(figsize=(7,7))

    if title is not None:
        title0 = title[0] if len(im.shape) == 3 else title
        ax.set_title(title0)
    
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99.9, .0001, 99.9999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        mm.set_clim(contrast)
    
    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)
        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            mm.set_data(im[nr])

            if title is not None:
                ax.set_title(title[nr])
    
    return fig, ax



class InteractiveCenter:
    """Plot image with controls for contrast and beamstop alignment tools."""
    
    def __init__(self, im, c0=None, c1=None, rBS=15, **kwargs):
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]
        
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        self.c0 = c0
        self.c1 = c1
        self.rBS = rBS
        
        self.circles = []
        for i in range(5):
            color = 'g' if i == 1 else 'r'
            circle = plt.Circle([c0, c1], 10 * (i + 1), ec=color, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

        self.widgets = { "w_c0" :ipywidgets.IntText(value=c0,step = 0.5, description="c0 (vert)"),
        "w_c1" : ipywidgets.IntText(value=c1,step = 0.5, description="c1 (hor)"),
        "w_rBS" : ipywidgets.IntText(value=rBS, description="rBS")}
        
        ipywidgets.interact(self.update, c0=self.widgets["w_c0"], c1=self.widgets["w_c1"], r=self.widgets["w_rBS"])
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
    
    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))
            
    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c1, c0 = (event.xdata, event.ydata)
            self.widgets["w_c0"].value = int(c0)
            self.widgets["w_c1"].value = int(c1)
            self.update(int(c0),int(c1),self.rBS)


def axis_to_roi(axis, labels=None):
    """
    Generate numpy slice expression from bounds of matplotlib figure axis.
    
    If labels is not None, return a roi dictionary for xarray.
    """
    x0, x1 = sorted(axis.get_xlim())
    y0, y1 = sorted(axis.get_ylim())
    if labels is None:
        roi = np.s_[
            int(round(y0)):int(round(y1)),
            int(round(x0)):int(round(x1))
        ]
    else:
        roi = {
            labels[0]: slice(int(round(y0)), int(round(y1))),
            labels[1]: slice(int(round(x0)), int(round(x1)))
        }
    return roi
    

class AzimuthalIntegrationCenter:
    """Plot image with controls for contrast and center alignment tools."""

    def __init__(self, im, ai, c0=None, c1=None, mask=None,**kwargs):
        # User Feedback/Instructions
        print("Left: 1d azimuthal Integration I(q)")
        print("Right: 2d azimuthal Integration I(q,chi)")
        print("Use arrow buttons on keyboard to adjust center position after selecting a slider.") 
        print("Try to transform all rings of the Airy pattern into a straight line in the 2d I(q,chi)-plot. Maximize fringe contrast in 1d I(q) plot for fine-tuning.")
        
        # Get center
        self.im = np.array(im)
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        #Variables
        self.c0 = c0
        self.c1 = c1
        self.radial_range = kwargs["radial_range"]
        self.im_data_range = kwargs["im_data_range"]
        self.pixel_size1 = ai.detector.get_pixel1()
        self.pixel_size2 = ai.detector.get_pixel2()
        self.qlines = kwargs["qlines"]
        self.ai = ai
        self.mask = mask

        # Calc azimuthal integration
        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            method = "BBox"
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 4))    

        # 1d Ai
        self.ax[0].plot(self.q_t, self.mI_t)
        self.ax[0].set_xlim(self.radial_range)
        self.ax[0].set_xlabel("q in 1/nm")
        self.ax[0].set_ylabel("Mean Integrated Intensity")
        self.ax[0].grid()
        
        # 2d Ai
        mi, ma = np.percentile(self.im,[1,99])
        self.timshow = self.ax[1].imshow(self.I_t, vmin=mi, vmax=ma)
        self.ax[1].set_ylabel("Angle")
        self.ax[1].set_xlabel("q in px")
        self.ax[1].grid()

        # qlines
        for qt in self.qlines:
            self.ax[1].axvline(qt, ymin=0, ymax=360, c="red")

        w_c0 = ipywidgets.FloatSlider(value=c0,min=im.shape[-2]/2-np.round(im.shape[-2]/6),max=im.shape[-2]/2+np.round(im.shape[-2]/6),step=.25, description="y-center",layout=ipywidgets.Layout(width="500px"))
        w_c1 = ipywidgets.FloatSlider(value=c1,min=im.shape[-1]/2-np.round(im.shape[-1]/6),max=im.shape[-1]/2+np.round(im.shape[-1]/6),step=.25, description="x-center",layout=ipywidgets.Layout(width="500px"))

        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1)

    def update(self, c0, c1, **kwargs):
        self.c0 = c0
        self.c1 = c1

        self.ai.poni1 = (
            self.c0 * self.pixel_size1)  # y (vertical)
        self.ai.poni2 = (
            self.c1 * self.pixel_size2)  # x (horizontal)

        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            method = "BBox"
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        # 1d Ai
        self.ax[0].clear()
        self.ax[0].plot(self.q_t, self.mI_t)
        self.ax[0].set_xlabel("q in 1/nm")
        self.ax[0].set_ylabel("Mean Integrated Intensity")
        self.ax[0].grid()

        # 2d Ai
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.timshow.set_data(self.I_t)
        self.timshow.set_clim([mi, ma])


class InteractiveBeamstop:
    """Plot image with controls for contrast and draw a beamstop. Use to find best radi and smoothing values."""
    def __init__(self, im, c0=None, c1=None, rBS=60,stdBS=4, **kwargs):        
        #Parameter coordinates
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        self.center = [c0,c1]
        
        #Beamstop parameter
        self.rBS = rBS
        self.stdBS = stdBS
        
        # Create beamstop mask
        im = np.array(im)
        self.im = im
        self.mask_bs = 1 - circle_mask(
            im.shape, self.center, self.rBS, sigma = self.stdBS
        )
        self.image = np.array(im*self.mask_bs)
        
        #Plotting
        fig, ax = plt.subplots()
        self.mm = ax.imshow(self.image)
        cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99, .1, 99.9])
        sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
        )
        cim = ipywidgets.interact(self.update_plt, contrast = sl_contrast)
        
        #Change beamstop parameter
        w_rBS = ipywidgets.IntText(value=self.rBS, description="radius")
        w_std = ipywidgets.IntText(value=self.stdBS, description="smoothing")
        w_c0 = ipywidgets.IntText(value=self.center[0], description="c0 (vert)")
        w_c1 = ipywidgets.IntText(value=self.center[1], description="c1 (horz)")
        ipywidgets.interact(self.update_bs, r=w_rBS,std = w_std, c0 = w_c0, c1 = w_c1)
    
    #Update plot
    def update_plt(self,contrast):
        self.mm.set_clim(contrast)
    
    #Update bs
    def update_bs(self, r,std, c0, c1):
        self.center = [c0,c1]
        self.rBS = r
        self.stdBS = std
        self.mask_bs = 1 - circle_mask(
            self.mask_bs.shape, self.center, r, sigma = std
        )
        self.image = self.im*self.mask_bs
        self.mm.set_data(self.image)


class draw_polygon_mask:
    """Interactive drawing of polygon masks"""

    def __init__(self, image,**kwargs):
        self.image = image
        self.image_plot = image
        self.full_mask = np.zeros(image.shape)
        self.coordinates = []
        self.masks = []
        self._create_widgets()
        self.kwargs = kwargs
        self.draw_gui()

    def _create_widgets(self):
        self.button_add = ipywidgets.Button(
            description="Add mask",
            button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_add.on_click(self.add_mask)
        
        
        self.button_del = ipywidgets.Button(
            description="Delete mask",
            #button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_del.on_click(self.del_mask)

    def draw_gui(self):
        """Create plot and control widgets"""

        # Plotting
        fig, self.ax = plt.subplots(figsize= (8,8))
        self.mm = self.ax.imshow(self.image_plot,**self.kwargs)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.01, 99.99])

        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )
        cim = ipywidgets.interact(self.update_plt, contrast=sl_contrast)

        # How to use
        print("Click on the figure to create a polygon corner.")
        print("Click `Add mask` to store coordinates and apply mask.")
        print("Press the 'esc' key to reset the polygon for new drawing.")
        print("")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")
        print("Button `Delete mask` deletes the masks recursively.")
        

        self.reset_polygon_selector()
        self.output = ipywidgets.Output()
        display(self.button_add,self.button_del, self.output)

    # Update plot
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def reset_polygon_selector(self):
        self.selector = PolygonSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="r", linestyle="-", linewidth=2, alpha=0.9),
        )

    def create_polygon_mask(self, shape, coordinates):
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(coordinates)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        self.masks.append(mask)
        self.coordinates.append(coordinates)
        
    def combine_masks(self):
        if len(self.masks) == 0:
            self.full_mask = np.zeros(self.image.shape)
        if len(self.masks) == 1:
            self.full_mask = self.masks[0]
        elif len(self.masks) > 1:
            self.full_mask = np.sum(np.array(self.masks).astype(int), axis=0)

        self.full_mask[self.full_mask > 1] = 1

    def add_mask(self, change):
        self.create_polygon_mask(self.image.shape, self.selector.verts)
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)
        
    def del_mask(self,change):
        self.coordinates.pop()
        self.masks.pop()
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)
        
    def round_nested_list(self, nested_list, precision):
        """
        Round all values in a nested list to a specified precision.

        Args:
        nested_list (list): The nested list containing numerical values and/or tuples.
        precision (int): Number of decimal places to round to (default is 2).

        Returns:
        list: A new nested list with all values rounded to the specified precision.
        """
        
        if isinstance(nested_list, list):
            return [self.round_nested_list(item, precision) for item in nested_list]
        elif isinstance(nested_list, tuple):
            return tuple(round(value, precision) for value in nested_list)
        else:
            return round(nested_list, precision)
        
    def get_vertice_coordinates(self):
        return self.round_nested_list(self.coordinates,1)

class InteractiveAutoBeamstop:
    """Plot image with controls for contrast and beamstop alignment tools."""

    def __init__(self, image, thres, radius, expand, method="intensity", **kwargs):
        self.image = image
        self.mask_bs = np.zeros(image.shape)
        self.thres = thres
        self.radius = radius
        self.expand = expand
        self.method = method

        self.create_widgets()
        self.automated_beamstop()
        self.draw_gui()

    def create_widgets(self):
        self.widgets = {
            "thres": ipywidgets.FloatSlider(
                min=0,
                max=5000,
                value=self.thres,
                step=10,
                description="Filter Threshold",
            ),
            "radius": ipywidgets.FloatSlider(
                min=0,
                max=np.max(np.array(self.image.shape) / 2),
                value=self.radius,
                step=10,
                description="Radius",
            ),
            "expand": ipywidgets.FloatSlider(
                min=0, max=20, value=self.expand, step=.5, description="Expansion"
            ),
        }

    def draw_gui(self):
        fig, self.ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.1, 99, 0.1, 99.9])
        self.m0 = self.ax[0].imshow(self.image)
        self.m1 = self.ax[1].imshow(self.image * self.mask_bs)
        self.m2 = self.ax[2].imshow(self.image * (1 - self.mask_bs))

        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )

        ipywidgets.interact(self.update_plt, contrast=sl_contrast)
        ipywidgets.interact(self.update_thres, thres=self.widgets["thres"])
        ipywidgets.interact(self.update_radius, radius=self.widgets["radius"])
        ipywidgets.interact(self.update_expand, expand=self.widgets["expand"])

    # Update imshow plot colormap
    def update_plt(self, contrast):
        self.m0.set_clim(contrast)
        self.m1.set_clim(contrast)
        self.m2.set_clim(contrast)

    def update_thres(self, thres):
        self.thres = thres
        self.automated_beamstop()
        self.update_mask()

    def update_radius(self, radius):
        self.radius = radius
        self.automated_beamstop()
        self.update_mask()

    def update_expand(self, expand):
        self.expand = expand
        self.automated_beamstop()
        self.update_mask()

    def mask_postprocessing(self, hologram_mask, radius, expand):
        # Draw beamstop only up to given radius
        hologram_mask = hologram_mask * circle_mask(
            self.image.shape,
            [self.image.shape[0] / 2, self.image.shape[1] / 2 - 3],
            radius,
            sigma=None,
        )
        hologram_mask = hologram_mask.astype(bool)

        # Morphological operations to filter reference modulations as these also lead to strong intensity gradients
        # close the "dots" of the ref modulations
        footprint = skimage.morphology.disk(2)
        hologram_mask = skimage.morphology.erosion(hologram_mask, footprint)

        # Filter remainings of ref modulations
        hologram_mask = skimage.morphology.remove_small_objects(
            hologram_mask, min_size=500
        )
        hologram_mask = 1 - skimage.morphology.remove_small_objects(
            (1 - hologram_mask).astype(bool), min_size=500
        )

        # Expand Mask
        footprint = skimage.morphology.disk(expand)
        hologram_mask = skimage.morphology.dilation(hologram_mask, footprint)

        # Fill up small holes in the mask
        hologram_mask = scp.ndimage.binary_fill_holes(
            hologram_mask, structure=np.ones((5, 5))
        )
        return hologram_mask

    def mask_postprocessing_otsu(self, hologram_mask, radius, expand):
        # Morphological operations to filter reference modulations
        # close the "dots" of the ref modulations
        footprint = skimage.morphology.disk(1)
        hologram_mask = skimage.morphology.erosion(
            (1 - hologram_mask).astype(bool), footprint
        )

        # Filter remainings of ref modulations
        hologram_mask = skimage.morphology.remove_small_objects(
            hologram_mask.astype(bool), min_size=2000
        )
        hologram_mask = 1 - skimage.morphology.remove_small_objects(
            (1 - hologram_mask).astype(bool), min_size=2000
        )

        # Expand mask to desired size
        footprint = skimage.morphology.disk(expand)
        hologram_mask = skimage.morphology.dilation(hologram_mask, footprint)

        # Draw beamstop only up to given radius
        hologram_mask = hologram_mask * circle_mask(
            hologram_mask.shape,
            np.array(hologram_mask.shape)/2,
            radius,
            sigma = None,
       )

        return hologram_mask

    def automated_beamstop(self):
        # Take from class
        hologram = self.image
        thres = self.thres
        radius = self.radius
        expand = self.expand

        # Different methods for filtering of beamstop
        if self.method == "intensity":
            # Some image preprocessing

            # Thresholding of gradient
            hologram_mask = hologram < thres

            # Postprocessing
            hologram_mask = self.mask_postprocessing(hologram_mask, radius, expand)
        elif self.method == "gradient":
            # Some image preprocessing
            hologram = np.mean(np.abs(np.gradient(hologram)), axis=0)

            # Thresholding of gradient
            hologram_mask = hologram < thres

            # Postprocessing
            hologram_mask = self.mask_postprocessing(hologram_mask, radius, expand)
        elif self.method == "otsu":
            # Some image preprocessing
            hologram[hologram<0]=0
            hologram = hologram +1 
            hologram = np.log10(hologram)

            # Prepare raw mask using otsu threshold method
            _, hologram_mask = median_otsu(hologram, median_radius=1, numpass=1)

            # Postprocessing
            hologram_mask = self.mask_postprocessing_otsu(hologram_mask, radius, expand)

        # Add mask to class
        self.mask_bs = hologram_mask.copy()

    def update_mask(self):
        self.m1.set_data(self.image * self.mask_bs)
        self.m2.set_data(self.image * (1 - self.mask_bs))

            

class InteractiveCircleCoordinates:
    """
    Creates overlay with circles on an image. Slider allow changing
    between circles, adjust circle position and radi. Usefull for
    creating support mask for holographically aided phase retrieval

    Return list of tuples with mask parameters (center, radius)
    """
    masks = []

    def __init__(self, image, num_masks,coordinates=None):
        print("Use circle index slider to change between circles. The active circle is highlighted in red."
        )
        print("Right click to move circle to mouse position!")
        self.image = image
        self.num_masks = num_masks
        self.init_masks(coordinates)
        self.draw_gui()

    def init_masks(self,coordinates):
        if coordinates is None:
            coordinates = []
            for n in range(self.num_masks):
                coordinates.append([self.image.shape[0]/2,self.image.shape[1]/2,10])
                
        self.masks = [
            plt.Circle((coordinates[n][1],coordinates[n][0]), coordinates[n][2], fill=False, ec="r") for n in range(self.num_masks)]
    
    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.mm = self.ax.imshow(self.image,vmin=vmin,vmax=vmax,cmap='gray')
        for mask in self.masks:
            self.ax.add_artist(mask)
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "mask_index": widgets.IntSlider(min=0, max=self.num_masks - 1, value=0),
            "radius": widgets.FloatSlider(
                min=0, max=400, value=10, step=0.5, description="radius",layout=ipywidgets.Layout(width="350px"),
            ),
            "c0": widgets.FloatSlider(
                min=0, max=2048, value=1024, step=0.5, description="x",layout=ipywidgets.Layout(width="400px"),
            ),
            "c1": widgets.FloatSlider(
                min=0, max=2048, value=1024, step=0.5, description="y",layout=ipywidgets.Layout(width="400px"),
            ),
        }

        ipywidgets.interact(self.update_plt, contrast=self.widgets["contrast"])
        widgets.interact(self.update_controls, index=self.widgets["mask_index"])
        widgets.interact(
            self.update_circle,
            radius=self.widgets["radius"],
            c0=self.widgets["c0"],
            c1=self.widgets["c1"],
        )
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
        
    # Update imshow plot colormap
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def update_controls(self, index):
        """Update control widget values with selected circle parameters."""
        circle = self.masks[index]
        r, (c0, c1) = circle.radius, circle.center

        self.widgets["radius"].value = r
        self.widgets["c0"].value = c0
        self.widgets["c1"].value = c1
            
        for c in self.masks:
            c.set_edgecolor("g")
        self.masks[index].set_edgecolor("r")

    def update_circle(self, radius, c0, c1):
        """Set center and size of active circle."""
        index = self.widgets["mask_index"].value
        self.masks[index].set_radius(radius)
        self.masks[index].set_center([c0, c1])
        
        print("Aperture Coordinates:")
        print(self.get_params())

    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        index = self.widgets["mask_index"].value
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            self.masks[index].set_center([c0, c1])
            self.widgets["c0"].value = c0
            self.widgets["c1"].value = c1

    def get_params(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return [(np.round(c.center[1],1),np.round(c.center[0],1), np.round(c.radius,1)) for c in self.masks]


class InteractiveEllipseCoordinates:
    def __init__(self, image, num_masks,coordinates=None):
        """
        Creates overlay with ellipses on an image. Sliders allow changing
        between ellipses, adjust ellipse positions, ellipse sizes and rotation angle. Usefull for
        creating support mask for holographically aided phase retrieval
    
        Return list of tuples with mask parameters (center, height, width, angle)
        """

        print("Use circle index slider to change between circles. The active circle is highlighted in red."
        )
        print("Right click to move circle to mouse position!")
        
        masks = []
        self.image = image
        self.num_masks = num_masks
        self.init_masks(coordinates)
        self.draw_gui()

    def init_masks(self,coordinates):
        if coordinates is None:
            coordinates = []
            for n in range(self.num_masks):
                coordinates.append([(self.image.shape[0]/2,self.image.shape[1]/2),10,10,0])

        self.masks = [Ellipse(coordinates[n][0],coordinates[n][2],coordinates[n][1],angle=coordinates[n][3],fill=False, ec="r") for n in range(self.num_masks)]

    
    def draw_gui(self):
        """Create plot and control widgets."""

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.mm = self.ax.imshow(self.image,vmin=vmin,vmax=vmax,cmap='gray')

        # Add masks to figure
        for mask in self.masks:
            self.ax.add_patch(mask)
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "mask_index": widgets.IntSlider(min=0, max=self.num_masks - 1, value=0),
            "height": widgets.FloatSlider(
                min=0, max=600, value=10, step=0.5, description="height",layout=ipywidgets.Layout(width="350px"),
            ),
            "width": widgets.FloatSlider(
                min=0, max=600, value=10, step=0.5, description="width",layout=ipywidgets.Layout(width="350px"),
            ),
            "angle": widgets.FloatSlider(
                min=0, max=180, value=0, step=0.5, description="angle",layout=ipywidgets.Layout(width="400px"),
            ),
            "c0": widgets.FloatSlider(
                min=0, max=2500, value=1024, step=0.5, description="x",layout=ipywidgets.Layout(width="400px"),
            ),
            "c1": widgets.FloatSlider(
                min=0, max=2500, value=1024, step=0.5, description="y",layout=ipywidgets.Layout(width="400px"),
            ),
        }

        # contrast slider
        ipywidgets.interact(self.update_plt, contrast=self.widgets["contrast"])

        #update widgets
        widgets.interact(self.update_controls, index=self.widgets["mask_index"])

        #update Ellipse
        widgets.interact(
            self.update_Ellipse,
            height = self.widgets["height"],
            width = self.widgets["width"],
            angle = self.widgets["angle"],
            c0=self.widgets["c0"],
            c1=self.widgets["c1"],
        )
        
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
        
    # Update imshow plot colormap
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def update_controls(self, index):
        """Update control widget values with selected circle parameters."""
        ellipse = self.masks[index]
        (c0, c1), height, width, angle = ellipse.center, ellipse.height, ellipse.width, ellipse.angle

        self.widgets["height"].value = height
        self.widgets["width"].value = width

        self.widgets["angle"].value = angle
        
        self.widgets["c0"].value = c0
        self.widgets["c1"].value = c1
            
        for c in self.masks:
            c.set_edgecolor("g")
        self.masks[index].set_edgecolor("r")

    def update_Ellipse(self, c0, c1, height, width, angle):
        """Set center and size of active circle."""
        index = self.widgets["mask_index"].value

        self.masks[index].set_center((c0, c1))
        self.masks[index].set_height(height)
        self.masks[index].set_width(width)
        self.masks[index].set_angle(angle)
        
        print("Aperture Coordinates:")
        print(self.get_params())

    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        index = self.widgets["mask_index"].value
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            self.masks[index].set_center([c0, c1])
            self.widgets["c0"].value = c0
            self.widgets["c1"].value = c1

    def get_params(self):
        """Return list of tuples with mask parameters (center, height, width, angle)"""
        return [((np.round(c.center[0],1),np.round(c.center[1],1)), c.height, c.width, np.round(c.angle,1)) for c in self.masks]


class InteractiveArcCoordinates:
    def __init__(self, image, num_masks, coordinates=None):
        self.image = image
        self.num_masks = num_masks
        self._updating = False
        self.init_masks(coordinates)
        self.draw_gui()

    def coordinates_to_dict(self, coordinate):
        return {
            "centery": coordinate[0][0],
            "centerx": coordinate[0][1],
            "r0": coordinate[1][0],
            "r1": coordinate[1][1],
            "phi0": coordinate[2][0],
            "phi1": coordinate[2][1],
        }

    def init_masks(self, coordinates):
        if coordinates is None:
            coordinates = []
            for n in range(self.num_masks):
                coordinates.append([
                    (self.image.shape[0] / 2, self.image.shape[1] / 2),
                    [50, 100],
                    [0, np.pi]
                ])

        self.mask_objects = [self.coordinates_to_dict(coordinate) for coordinate in coordinates]
        self.masks = [arc_mask(self.image.shape, *coordinates[n]) for n in range(self.num_masks)]
        self.project_masks()

    def project_masks(self):
        active_index = self.widgets["mask_index"].value if hasattr(self, "widgets") else 0
        h, w = self.image.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=float)

        for i, mask in enumerate(self.masks):
            m = mask.astype(float)
            if i == active_index:
                overlay[..., 0] = np.maximum(overlay[..., 0], m)
                overlay[..., 3] = np.maximum(overlay[..., 3], m * 0.5)
            else:
                overlay[..., 2] = np.maximum(overlay[..., 2], m)
                overlay[..., 3] = np.maximum(overlay[..., 3], m * 0.3)

        self.mask = overlay

    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.mm = self.ax.imshow(self.image, vmin=vmin, vmax=vmax, cmap='gray')
        self.m2 = self.ax.imshow(self.mask)

        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
                value=(vmin, vmax),
                min=cmin,
                max=cmax,
                step=(vmax - vmin) / 500,
                layout=ipywidgets.Layout(width="500px"),
            ),
            "mask_index": widgets.IntSlider(min=0, max=self.num_masks - 1, value=0),
            "centery": widgets.FloatSlider(
                min=0, max=self.image.shape[0], value=self.image.shape[0] / 2,
                step=0.5, description="centery", layout=ipywidgets.Layout(width="350px"),
            ),
            "centerx": widgets.FloatSlider(
                min=0, max=self.image.shape[1], value=self.image.shape[1] / 2,
                step=0.5, description="centerx", layout=ipywidgets.Layout(width="350px"),
            ),
            "inner_radius": widgets.FloatSlider(
                min=0, max=np.array(self.image.shape).max() / 2, value=50,
                step=0.5, description="inner_radius", layout=ipywidgets.Layout(width="450px"),
            ),
            "outer_radius": widgets.FloatSlider(
                min=0, max=np.array(self.image.shape).max() / 2, value=100,
                step=0.5, description="outer_radius", layout=ipywidgets.Layout(width="450px"),
            ),
            "start_angle": widgets.FloatSlider(
                min=0, max=2 * np.pi, value=0,
                step=0.0125, description="start_angle",readout_format=".4f", layout=ipywidgets.Layout(width="450px"),
            ),
            "end_angle": widgets.FloatSlider(
                min=0, max=2 * np.pi, value=np.pi,
                step=0.0125, description="end_angle", readout_format=".4f",layout=ipywidgets.Layout(width="450px"),
            ),
        }

        ipywidgets.interact(self.update_clim, contrast=self.widgets["contrast"])
        widgets.interact(self.update_controls, index=self.widgets["mask_index"])
        widgets.interact(
            self.update_arc,
            centery=self.widgets["centery"],
            centerx=self.widgets["centerx"],
            inner_radius=self.widgets["inner_radius"],
            outer_radius=self.widgets["outer_radius"],
            start_angle=self.widgets["start_angle"],
            end_angle=self.widgets["end_angle"],
        )

    def update_clim(self, contrast):
        self.mm.set_clim(contrast)

    def update_controls(self, index):
        """Update control widget values with selected circle parameters."""
        self._updating = True
        arc = self.mask_objects[index]
        self.widgets["centery"].value = arc["centery"]
        self.widgets["centerx"].value = arc["centerx"]
        self.widgets["inner_radius"].value = arc["r0"]
        self.widgets["outer_radius"].value = arc["r1"]
        self.widgets["start_angle"].value = arc["phi0"]
        self.widgets["end_angle"].value = arc["phi1"]
        self._updating = False
        self.project_masks()
        self.update_image()

    def update_arc(self, centery, centerx, inner_radius, outer_radius, start_angle, end_angle):
        if self._updating:
            return
        index = self.widgets["mask_index"].value
        arc_dict = {
            "centery": centery,
            "centerx": centerx,
            "r0": inner_radius,
            "r1": outer_radius,
            "phi0": start_angle,
            "phi1": end_angle,
        }
        self.mask_objects[index].update(arc_dict)
        self.masks[index] = arc_mask(self.image.shape, *self.dict_to_params(arc_dict))
        self.project_masks()
        self.update_image()

        print("Aperture Coordinates:")
        print(self.get_params())

    def update_image(self):
        self.m2.set_data(self.mask)

    def dict_to_params(self, arc_dict):
        return [
            [arc_dict["centery"], arc_dict["centerx"]],
            [arc_dict["r0"], arc_dict["r1"]],
            [np.round(arc_dict["phi0"],3), np.round(arc_dict["phi1"],3)],
        ]

    def get_params(self):
        """Return list of tuples with mask parameters (center, height, width, angle)"""
        return [self.dict_to_params(arc_dict) for arc_dict in self.mask_objects]

    def return_mask(self):
        return np.clip(np.sum(self.masks,axis=0),0,1)

class Shift_Scale_Mask:
    """Plot image with controls for contrast, x/y shift and scaling."""
    
    def __init__(self, image, mask, shift = [0,0], scale = 0, **kwargs):
        self.image = image
        self.shape = self.image.shape
        self.mask_original = mask
        self.mask = mask
        self.mask_shifted = mask
        self.shift = shift
        self.scale = scale
        self.kwargs = kwargs
        self.draw_gui()

        
    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.m0 = self.ax[0].imshow(self.image,**self.kwargs)
        self.m1 = self.ax[1].imshow(self.image,**self.kwargs)
        self.ax[0].set_title("Image*Mask")
        self.ax[1].set_title("Image*(1-Mask)")
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "shift_ver": widgets.FloatSlider(
                min=-self.shape[1]/4, max=self.shape[1]/4, value=self.shift[0], step=1, description="shift_ver",layout=ipywidgets.Layout(width="350px")),
            "shift_hor": widgets.FloatSlider(
                min=-self.shape[1]/4, max=self.shape[1]/4, value=self.shift[1], step=1, description="shift_hor",layout=ipywidgets.Layout(width="350px")),
            "scale": widgets.IntSlider(min=-20, max=20, value=self.scale,description="scale"),
                    }

        ipywidgets.interact(self.update_plt_contrast, contrast=self.widgets["contrast"])
        widgets.interact(
            self.update_mask,
            shift_ver=self.widgets["shift_ver"],
            shift_hor=self.widgets["shift_hor"],
            scale=self.widgets["scale"],
        )
        
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
            
    def update_plt_contrast(self, contrast):
        self.m0.set_clim(contrast)
        self.m1.set_clim(contrast)
    
    def update_plt_images(self):
        self.m0.set_data(self.image*self.mask)
        self.m1.set_data(self.image*(1-self.mask))
    
    def shift_mask(self, shift_ver,shift_hor):
            self.shift = [shift_ver,shift_hor]
            self.mask = np.round(shift_image(self.mask_original,self.shift))
        
    def scale_mask(self,scale):
        self.scale = -scale
        if scale > 0:
            footprint = skimage.morphology.disk(scale)
            self.mask = skimage.morphology.dilation(self.mask, footprint)
        elif scale < 0:
            footprint = skimage.morphology.disk(np.abs(scale))
            self.mask = skimage.morphology.erosion(self.mask, footprint)
            
    def update_mask(self,shift_ver,shift_hor,scale):
        self.shift_mask(shift_ver,shift_hor)
        if scale !=0:
            self.scale_mask(scale)
            
        self.update_plt_images()
            
    def onclick_handler(self, event):
        """Set the center of the mask to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            shift = [self.mask.shape[0]/2-c0,self.mask.shape[0]/2-c1]
            self.update_mask(shift[0],shift[1],self.scale)
        
    def get_mask(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return self.mask, self.shift, self.scale


class Shift_Rotate:
    """Plot image with controls for contrast, x/y shift and scaling."""
    
    def __init__(self, image, shift = [0,0], angle = 0, ticks = None):
        self.image = image
        self.shape = self.image.shape
        self.image_original = image
        self.shift = shift
        self.angle = angle
        self.ticks = ticks
        
        self.draw_gui()        
        
    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.01, 99.99])
        self.m0 = self.ax.imshow(self.image,vmin=vmin,vmax=vmax)
        self.ax.set_title("Image")
            
        if self.ticks is not None:
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            self.ax.set_xticks(self.ticks[1])
            self.ax.set_yticks(self.ticks[0])
            plt.grid()
            
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "shift_ver": widgets.FloatSlider(
                min=-self.shape[1]/2, max=self.shape[1]/2, value=self.shift[0], step=0.5, description="shift_ver",layout=ipywidgets.Layout(width="350px")),
            "shift_hor": widgets.FloatSlider(
                min=-self.shape[1]/2, max=self.shape[1]/2, value=self.shift[1], step=0.5, description="shift_hor",layout=ipywidgets.Layout(width="350px")),
            "angle": widgets.FloatSlider(min=-180, max=180, value=self.angle,step=0.25,description="angle"),
                    }

        ipywidgets.interact(self.update_plt_contrast, contrast=self.widgets["contrast"])
        widgets.interact(
            self.update_image,
            shift_ver=self.widgets["shift_ver"],
            shift_hor=self.widgets["shift_hor"],
            angle=self.widgets["angle"],
        )
        
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
            
    def update_plt_contrast(self, contrast):
        self.m0.set_clim(contrast)
    
    def update_plt_images(self):
        self.m0.set_data(self.image)
        
    def update_image(self,shift_ver,shift_hor,angle):
        self.rotate_image(angle)
        self.shift_image(shift_ver,shift_hor)
            
        self.update_plt_images()
    
    def update_controls(self,shift):
        self.widgets["shift_ver"].value = shift[0]
        self.widgets["shift_hor"].value = shift[1]
    
    def shift_image(self, shift_ver,shift_hor):
            self.shift = [shift_ver,shift_hor]
            self.image = np.round(shift_image(self.image,self.shift))
        
    def rotate_image(self,angle):
        self.angle = angle
        
        if self.angle != 0:
            self.image = rotate(self.image_original,self.angle,reshape=False)
        elif self.angle == 0:
            self.image = self.image_original.copy()
            
    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            shift = [-1*(self.shape[0]/2-c1),-1*(self.shape[1]/2-c0)]
            self.update_controls(shift)
            self.update_image(shift[0],shift[1],self.angle)
        
    def get_parameter(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return self.image, self.shift, self.angle


def focusCDI(pos,neg, roi, mask=1,phase=0, prop_dist=0,dx=0, dy=0, scale=(0,100), experimental_setup={'ccd_dist':18e-2, 'energy':779.5, 'px_size':20e-6}, operation="-", max_prop_dist=10):
    '''
    Applies a sub-pixel centering, propagation distance and global phase shift.
    Also plots real,image,abs,angle images while you do it
    INPUT:  pos,neg: array, the shifted and masked holograms
            mask: optional array, =1 in the region you want to consider, =0 elsewhere. Limits of the colormaps are going to be chosen in this region
            roi: array, coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
            phase: optional, float, starting value for the phase slider (default is 0)
            prop_dist: optional, float, starting value for the propagation slider (default is 0)
            scale: optional, tuple of floats, values for the scaling using percentiles (default is (0, 100))
            experimental_setup: dictionary containing:
             - ccd_dist: optional, float, distance between CCD and sample in meter (default is 18e-2 (m))
             - energy: optional, float, energy of the x-rays in eV (default is 779.5 (eV))
             - px_size: optional, float, physical size of the CCD pixel in m (default is 20e-6 (m))
            operation: the operation you'll do on those holograms (-,/,+,-/+, load_both)
            max_prop_dist: maximum value for propagated distances
    OUPUT:  sliders for the propagation, phase, subpixel shift distances in x and y
            When you are finished, you can save the positions of the sliders.
    -------
    author: RB 2020
    '''
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(2,2, figsize=(6,6))
    def p(x, y, fx, fy):
        image_p = FFT(propagate(pos, x*1e-6, experimental_setup)* np.exp(1j*y))
        image_n = FFT(propagate(neg, x*1e-6, experimental_setup)* np.exp(1j*y))
        
        image_n = shift_image(image_n, [fx, fy])
        maskroi=(mask)[roi]
        
        if operation== "-":
            image= (image_p-image_n)
        elif operation== "+":
            image= (image_p+image_n)
        elif operation=="/":
            image= (image_p/image_n)* np.exp(1j*y)
        elif operation=="log":
            image= np.log(np.abs(image_p/image_n))* np.exp(1j*np.angle(image_p/image_n))
        elif operation=="-/+":
            image= (image_p-image_n)/(image_p+image_n) * np.exp(1j*y)

        image=np.nan_to_num(image, nan=0, posinf=0, neginf=0)[roi]
        simage_mask=image[maskroi==1]
    
        mi,ma=np.percentile(np.abs(simage_mask), scale)
        ax1 = axs[0,0].imshow(np.abs(image),cmap = 'gray', vmin=mi, vmax=ma)
        axs[0,0].set_title("Abs")
        
        mi,ma=np.percentile(np.angle(simage_mask), scale)
        ax2 = axs[0,1].imshow(np.angle(image), cmap='gray', vmin=mi, vmax=ma)
        axs[0,1].set_title("Phase")
        
        mi,ma=np.percentile(np.real(simage_mask), scale)
        ax3 = axs[1,0].imshow(np.real(image), cmap='gray', vmin=mi, vmax=ma)
        axs[1,0].set_title("Real Part")
        
        mi,ma=np.percentile(np.imag(simage_mask), scale)
        ax4 = axs[1,1].imshow(np.imag(image), cmap='gray', vmin=mi, vmax=ma)
        axs[1,1].set_title("Imaginary Part")
        #fig.tight_layout()
        return
    
    layout = widgets.Layout(width='75%')
    style = {'description_width': 'initial'}
    slider_prop = widgets.FloatSlider(min=-max_prop_dist, max=max_prop_dist, step=0.01, value=prop_dist, layout=layout,
                                      description='propagation[um]', style=style)
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout,
                                       description='phase shift', style=style)
    slider_dx = widgets.FloatSlider(min = -6, max = 6, step = 0.01, value = dx, layout = layout,
                                       description = 'x shift', style = style)
    slider_dy = widgets.FloatSlider(min = -6, max = 6, step = 0.01, value = dy, layout = layout,
                                       description = 'y shift', style = style)

    widgets.interact(p, x=slider_prop, y=slider_phase, fx = slider_dx, fy = slider_dy)

    return (slider_prop, slider_phase, slider_dx, slider_dy)


def propagate_phase(holo, ROI, phase=0, prop_dist=0, scale=(0,100), experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}):
    '''
    starts the quest for the right propagation distance and global phase shift.
    Input:  centered and masked hologram (difference, sum, single helicity, ...)
            coordinates of the ROI in the order np.array([y1, y2, x1, x2]) as np.s_
    Returns the two slider's position which can be retrieved
    -------
    author: CK 2026
    '''
    ph_flip = False
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,3,figsize=(9,4))
    def p(x,y):
        image = reconstruct(propagate(holo, x*1e-6, experimental_setup = experimental_setup)*np.exp(1j*y))
        mir, mar = np.percentile(np.real(image[ROI]), scale)
        mii, mai = np.percentile(np.imag(image[ROI]), scale)
        mia, maa = np.percentile(np.abs(image[ROI]), scale)

        ax1 = axs[0].imshow(np.real(image[ROI]), cmap='gray', vmin = mir, vmax = mar)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[ROI]), cmap='gray', vmin = mii, vmax = mai)
        axs[1].set_title("Imaginary Part")
        ax2 = axs[2].imshow(np.abs(image[ROI]), cmap='gray', vmin = mii, vmax = mai)
        axs[2].set_title("Absolute Value")
        
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return
    
    layout = widgets.Layout(width='750px')
    style = {'description_width': 'initial'}
    slider_prop = widgets.FloatSlider(min=-10, max=10, step=0.01, value=prop_dist, layout=layout, description='propagation[um]', style=style)
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout, description='phase shift', style=style)
    
    widgets.interact(p, x=slider_prop, y=slider_phase)
    
    return (slider_prop, slider_phase)