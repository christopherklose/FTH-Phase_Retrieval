import numpy as np
import h5py

import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider
from scipy.constants import c, h, e

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import ipywidgets

import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

import skimage.morphology
from dipy.segment.mask import median_otsu

from fth import reconstruct, shift_image, propagate, shift_phase


#Draw circle mask
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
    if sigma != None:
        mask = gaussian_filter(mask,sigma)
           
    return mask

def cimshow(im, **kwargs):
    """Simple 2d image plot with adjustable contrast.
    
    Returns matplotlib figure and axis created.
    """
    im = np.array(im)
    fig, ax = plt.subplots()
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99.9, .001, 99.999])
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

        w_c0 = ipywidgets.IntText(value=c0,step = 0.5, description="c0")
        w_c1 = ipywidgets.IntText(value=c1,step = 0.5, description="c1")
        w_rBS = ipywidgets.IntText(value=rBS, description="rBS")
        
        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1, r=w_rBS)
    
    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))

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



class InteractiveOptimizer:
    """
    Interactively adjust FTH parameters: center, propagation and phase shift.
    
    TODO: parameters...
    """
    
    params = {"phase": 0, "center": (0, 0), "propdist": 0, "pixelsize": 13.5e-6,
              "energy": 779, "detectordist": 0.2}
    widgets = {}
    
    def __init__(self, holo, roi, params={}):
        self.params.update(params)
        self.holo = holo  #.astype(np.single)
        self.holo_centered = holo.copy()
        self.holo_prop = holo.copy()
        self.roi = roi
        
        self.make_ui()
    
    def make_ui(self):
        self.fig, (self.axr, self.axi) = plt.subplots(
            ncols=2, figsize=(7, 3.5), sharex=True, sharey=True,
            constrained_layout=True,
        )
        
        self.reco = reconstruct(self.holo)[self.roi]
        vmin, vmax = np.percentile(self.reco.real, [.01, 99.9])
        vlim = 2 * np.abs(self.reco.real).max()

        opt = dict(vmin=vmin, vmax=vmax, cmap="gray_r")
        self.mm_real = self.axr.imshow(self.reco.real, **opt)
        self.mm_imag = self.axi.imshow(self.reco.imag, **opt)
    
        self.widgets["clim"] = FloatRangeSlider(
            value=(vmin, vmax), min=-vlim, max=vlim,
        )
        self.widgets["phase"] = FloatSlider(
            value=self.params["phase"], min=-np.pi, max=np.pi,
        )
        self.widgets["c0"] = FloatSlider(
            value=self.params["center"][0], min=-5, max=5, step=.01
        )
        self.widgets["c1"] = FloatSlider(
            value=self.params["center"][1], min=-5, max=5, step=.01
        )
        self.widgets["propdist"] = FloatSlider(
            value=self.params["propdist"], min=-10, max=10, step=.1
        )
        self.widgets["energy"] = ipywidgets.BoundedFloatText(
            value=self.params["energy"], min=1, max=10000,
        )
        self.widgets["detectordist"] = ipywidgets.BoundedFloatText(
            value=self.params["detectordist"], min=.01
        )
        self.widgets["pixelsize"] = ipywidgets.BoundedFloatText(
            value=self.params["pixelsize"], min=1e-7,
        )
        
        interact(self.update_clim, clim=self.widgets["clim"])
        interact(self.update_phase, phase=self.widgets["phase"])
        interact(
            self.update_center,
            c0=self.widgets["c0"],
            c1=self.widgets["c1"]
        )
        interact(
            self.update_propagation,
            dist=self.widgets["propdist"],
            det=self.widgets["detectordist"],
            pxs=self.widgets["pixelsize"],
            energy=self.widgets["energy"],
        )
    
    def update_clim(self, clim):
        self.mm_real.set_clim(clim)
        self.mm_imag.set_clim(clim)
    
    def update_phase(self, phase):
        self.params["phase"] = phase
        reco_shifted = shift_phase(self.reco, phase)
        self.mm_real.set_data(reco_shifted.real)
        self.mm_imag.set_data(reco_shifted.imag)
    
    def update_center(self, c0, c1):
        self.params["center"] = (c0, c1)
        self.holo_centered = shift_image(self.holo, [c0, c1])
        self.reco = reconstruct(self.holo_centered)[self.roi]
        self.update_phase(self.params["phase"])
    
    def update_propagation(self, dist, det, pxs, energy):
        dist *= 1e-6
        self.params.update({
            "propdist": dist,
            "detectordist": det,
            "pixelsize": pxs,
            "energy": energy
        })
        self.holo_prop = propagate(self.holo_centered, dist, det, pxs, energy)
        self.reco = reconstruct(self.holo_prop)[self.roi]
        self.update_phase(self.params["phase"])
    
    def get_full_reco(self):
        return shift_phase(reconstruct(self.holo_prop), self.params["phase"])


def intensity_scale(im1, im2, mask=None):
    mask = mask if mask is not None else 1
    diff = (im1 - im2) * mask
    fig, ax = plt.subplots()
    hist, bins, patches = ax.hist(mask.flatten(), np.linspace(-100, 100, 201))
    ax.set_yscale("log")
    ax.axvline(0, c='r', lw=.5)
    ax.grid(True)

    @ipywidgets.interact(f=(.2, 2.0, .001))
    def update(f):
        diff = mask * (im1 - f * im2)
        hist, _ = np.histogram(diff, bins)
        for p, v in zip(patches, hist):
            p.set_height(v)
    return fig, ax
    
    
    
class AzimuthalIntegrationCenter:
    """Plot image with controls for contrast and center alignment tools."""

    def __init__(self, im, ai, c0=None, c1=None, **kwargs):
        # User Feedback/Instructions
        print("Left: 1d azimuthal Integration I(q)")
        print("Right: 2d azimuthal Integration I(q,chi)")
        print("Use sliders to adjust center position. Try to transform all rings of the Airy pattern into a straight line in the I(q,chi)-plot. Maximize fringe contrast in I(q) plot for fine-tuning.")
        
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

        # Calc azimuthal integration
        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            #method="cython"
        )
        self.mI_t = np.mean(self.I_t, axis=0)

        # Plot
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        # 1d Ai
        self.ax[0].plot(self.q_t, self.mI_t)
        self.ax[0].set_xlim(self.radial_range)
        self.ax[0].set_xlabel("q in 1/nm")
        self.ax[0].set_ylabel("Mean Integrated Intensity")
        self.ax[0].grid()
        # 2d Ai
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.timshow = self.ax[1].imshow(self.I_t, vmin=mi, vmax=ma)
        self.ax[1].set_ylabel("Angle")
        self.ax[1].set_xlabel("q in px")
        self.ax[1].grid()

        # qlines
        for qt in self.qlines:
            self.ax[1].axvline(qt, ymin=0, ymax=360, c="red")

        w_c0 = ipywidgets.FloatSlider(value=c0,min=im.shape[-2]/2-np.round(im.shape[-2]/6),max=im.shape[-2]/2+np.round(im.shape[-2]/6),step=.5, description="y-center",layout=ipywidgets.Layout(width="500px"))
        w_c1 = ipywidgets.FloatSlider(value=c1,min=im.shape[-1]/2-np.round(im.shape[-1]/6),max=im.shape[-1]/2+np.round(im.shape[-1]/6),step=.5, description="x-center",layout=ipywidgets.Layout(width="500px"))

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
            #method="cython"
        )
        self.mI_t = np.mean(self.I_t, axis=0)

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
        
        
class InteractiveCircleCoordinates:
    """Draw a given number of circles. Radi and position of each circel can be adjusted with sliders. Useful for holographically-aided, iterative phase retrieval."""

    def __init__(self, im, nr_circ, support_coordinates="None", **kwargs):
        # Display image
        im = np.array(im)
        self.fig, self.ax = cimshow(im, cmap="gray", **kwargs)
        self.mm = self.ax.get_images()[0]

        # Create list of aperture coordinates
        if support_coordinates == "None":
            self.c_yxr = []
            for i in range(nr_circ):
                self.c_yxr.append([im.shape[-2] // 2, im.shape[-1] // 2, 15])
        else:
            self.c_yxr = support_coordinates

        # Draw circles
        self.circles = []
        for i in range(nr_circ):
            color = "r" if i == 0 else "g"
            circle = plt.Circle(
                [self.c_yxr[i][1], self.c_yxr[i][0]],
                self.c_yxr[i][2],
                ec=color,
                fill=False,
            )
            self.circles.append(circle)
            self.ax.add_artist(circle)

        # Create slider to select aperture
        w_idx = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=nr_circ - 1,
            step=1,
            description="circle index",
            layout=ipywidgets.Layout(width="250px"),
        )

        # Create circle widget sliders
        label = ["yc", "xc", "r0"]
        width = ["600px", "600px", "500px"]
        mi = [im.shape[-2] // 2 - im.shape[-2] // 2, im.shape[-2] // 2 - im.shape[-2] // 2, 0]
        ma = [im.shape[-2] // 2 + im.shape[-2] // 2, im.shape[-2] // 2 + im.shape[-2] // 2, im.shape[-2] // 10]
        w = [
            ipywidgets.FloatSlider(
                value=self.c_yxr[0][k],
                min=mi[k],
                max=ma[k],
                step=0.5,
                description=label[k],
                layout=ipywidgets.Layout(width=width[k]),
            )
            for k in range(3)
        ]
        self.w = w

        # Interactive user interface
        iidx = ipywidgets.interact(self.update_index, idx_circ=w_idx)
        icirc = ipywidgets.interact(
            self.update,
            c0=self.w[0],
            c1=self.w[1],
            r=self.w[2],
        )

    # Update initial widget values to the ones from list when changing between circles
    def update_index(self, idx_circ):
        self.idx = idx_circ
        cy = self.c_yxr[self.idx][0]
        cx = self.c_yxr[self.idx][1]
        cr = self.c_yxr[self.idx][2]

        # Keep these separated as this prevents overwriting of the individual widget values
        self.w[0].value = cy
        self.w[1].value = cx
        self.w[2].value = cr
        
        #Change color, active circle is red
        for i, c in enumerate(self.circles):
            color = "r" if i == self.idx else "g"
            c.set_edgecolor(color)

    # Update circle values
    def update(self, c0, c1, r):
        # Update coordinate dictionary
        self.c_yxr[self.idx] = [c0, c1, r]

        # Update drawn circles
        c = self.circles[self.idx]
        c.set_center([c1, c0])
        c.set_radius(r)

        print("Aperture Coordinates:")
        print(self.c_yxr)
        
        
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
        w_rBS = ipywidgets.IntText(value=self.rBS, description="rBS")
        w_std = ipywidgets.IntText(value=self.stdBS, description="stdBS")
        ipywidgets.interact(self.update_bs, r=w_rBS,std = w_std)
    
    #Update plot
    def update_plt(self,contrast):
        self.mm.set_clim(contrast)
    
    #Update bs
    def update_bs(self, r,std):
        self.rBS = r
        self.stdBS = std
        self.mask_bs = 1 - circle_mask(
            self.mask_bs.shape, self.center, r, sigma = std
        )
        self.image = self.im*self.mask_bs
        self.mm.set_data(self.image)
        
        
class draw_polygon_mask:
    """Interactive drawing of polygon masks"""

    def __init__(self, image):
        self.image = image
        self.image_plot = image
        self.full_mask = np.zeros(image.shape)
        self.coordinates = []
        self.masks = []
        self._create_widgets()
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
        fig, self.ax = plt.subplots()
        self.mm = self.ax.imshow(self.image_plot)
        # self.overlay = self.ax.imshow(self.full_mask, alpha=0.2)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.1, 99, 0.1, 99.9])

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
                min=0, max=20, value=self.expand, step=1, description="Expansion"
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
            hologram_mask, min_size=2000
        )
        hologram_mask = 1 - skimage.morphology.remove_small_objects(
            (1 - hologram_mask).astype(bool), min_size=2000
        )

        # Expand Mask
        footprint = skimage.morphology.disk(expand)
        hologram_mask = skimage.morphology.dilation(hologram_mask, footprint)

        # Fill up small holes in the mask
        hologram_mask = sp.ndimage.binary_fill_holes(
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
            [hologram_mask.shape[0] / 2, hologram_mask.shape[1] / 2 - 3],
            radius,
            sigma=None,
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