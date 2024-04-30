import numpy as np
from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter

# Is there a GPU?
try:
    # Cupy
    import cupy as cp
    import cupyx as cpx

    GPU = True

    print("GPU available")

    # Self-written library
    import CCI_core_cupy as cci
except:
    GPU = False
    import CCI_core as cci

    print("GPU unavailable")
    
    
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
    

def create_single_polygon_mask(shape, coordinates):
    """
    Creates a polygon mask from coordinates of corner points

    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    coordinates: nested list
        coordinates of polygon corner points [(yc_1,xc_1),(yc_2,xc_2),...]


    Output
    ======
    mask: array
        binary mask where filled polygon is "1"
    ======
    author: ck 2023
    """

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(coordinates)
    mask = path.contains_points(points)
    mask = mask.reshape(shape)
    return mask


def create_polygon_mask(shape, coordinates):
    """
    Creates multiple polygon masks from set of coordinates of corner points

    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    coordinates: nested list
        coordinates of polygon corner points for multiple polygons
        [[(yc_1,xc_1),(yc_2,xc_2),...],[(yc_1,xc_1),(yc_2,xc_2),...]]

    Output
    ======
    mask: array
        binary mask where filled polygons are "1"
    ======
    author: ck 2023
    """

    if len(coordinates) == 1:
        mask = create_single_polygon_mask(shape, coordinates[0])

    # Loop over coordinates
    elif len(coordinates) > 1:
        mask = np.zeros(shape)
        for coord in coordinates:
            mask = mask + create_single_polygon_mask(shape, coord)
            mask[mask > 1] = 1

    return mask


def load_poly_masks(shape, mask_dict, polygon_name_list):
    """
    Loads set of polygon masks based on stored coordinates

    Parameter
    =========
    shape : tuple
        shape of output mask
    polygon_name_list : list
        keys of different mask coordinates to load

    Output
    ======
    mask: array
        binary mask where filled polygons are "1"
    ======
    author: ck 2023
    """

    if len(polygon_name_list) > 0:
        mask = []

        # Loop over relevant mask keys
        for polygon_name in polygon_name_list:
            coord = mask_dict[polygon_name]
            mask.append(create_polygon_mask(shape, coord).astype(float))

        # Combine all individual mask layers
        mask = np.array(mask)
        mask = np.sum(mask, axis=0)
        mask[mask > 1] = 1
    else:
        mask = np.zeros(shape)

    return mask


def auto_shift_mask(
    mask,
    image,
    shift_range_y=(-10, 10),
    shift_range_x=(-10, 10),
    step_size=1,
    crop=None,
    method="minimize",
):
    """
    Automatically shifts a binary (beamstop) mask to the optimal position

    Parameter
    =========
    mask : array
        binary mask, e.g., beamstop mask
    image : array
        image partially covered by mask
    shift_range_y, _x : tupel
        Min and max limit for search area in y- and x-direction
    step_size : scalar
        step size of scan area
    crop : int
        additional cropping of arrays to speed up calculations
        Crops according to [crop:-crop,crop:-crop]
    method : string
        method used for optimization. 'minimize' or 'maximize' metric <mask,image>

    Output
    ======
    optimized shift: tupel
        shift vector for optimized posistion
    mask_shifted: array
        mask shifted according to best shift vector
    overlap : array
        computed evaluation metric
    ======
    author: ck 2023
    """

    # Basic looping
    # Create set of arrays for shifting
    yshift = np.arange(shift_range_y[0], shift_range_y[1], step_size)
    xshift = np.arange(shift_range_x[0], shift_range_x[1], step_size)

    # Setup loss function to evaluate overlap
    overlap = np.zeros((yshift.shape[0], xshift.shape[0]))

    # Optinal cropping to reduce computation time
    if crop is not None:
        tmask, timage = mask[crop:-crop, crop:-crop], image[crop:-crop, crop:-crop]
    else:
        tmask, timage = mask.copy(), timage.copy()

    # Loop over all combinations of shifts
    for i, y in enumerate(yshift):
        for j, x in enumerate(xshift):
            # Shift mask to new position
            mask_shift = cci.shift_image(tmask, [y, x])

            # Calculate overlap
            overlap[i, j] = np.sum(mask_shift * timage)

    # Get best shift
    if method == "minimize":
        idx = np.unravel_index(
            np.argmin(
                overlap,
            ),
            overlap.shape,
        )
    elif method == "maximize":
        idx = np.unravel_index(
            np.argmax(
                overlap,
            ),
            overlap.shape,
        )

    # Output
    optimized_shift = (yshift[idx[0]], yshift[idx[1]])
    print(
        "Best mask overlap for shift: [%.2f,%.2f]"
        % (optimized_shift[0], optimized_shift[1])
    )

    # Shift
    mask_shifted = cci.shift_image(mask, optimized_shift)

    # Binarize mask (necessary for sub-px shift)
    mask_shifted[mask_shifted > 0.5] = 1
    mask_shifted[mask_shifted < 0.5] = 0

    return optimized_shift, mask_shifted, overlap


def create_circle_supportmask(support_coordinates, shape):
    """
    Create cdi support mask from a combination of multiple circular apertures

    Parameter
    =========
    support_coordinates: nested list
        Contains center coordinates and radius of each aperture [[yc_1,xc_1,r_1],[yc_2,xc_2,r_2],...]
    shape : int tuple
        shape/dimension of output array

    Output
    ======
    supportmask: array
        composed binary mask where circular apertures are "1"
    ======
    author: ck 2023

    """

    # Create support mask
    supportmask = np.zeros(shape)
    for i in range(len(support_coordinates)):
        supportmask += circle_mask(
            supportmask.shape,
            [support_coordinates[i][0], support_coordinates[i][1]],
            support_coordinates[i][2],
        )

    return supportmask


def create_ellipse_supportmask(support_coordinates, shape):
    """
    Create cdi support mask as a combination of multiple ellipses

    Parameter
    =========
    support_coordinates: nested list
        Contains center coordinates, height, width and rotation angle of each aperture
        [[(yc_1,xc_1),height_1,width_1,angle_1],[(yc_2,xc_2),height_2,width_2,angle_2],...]
    shape : int tuple
        shape/dimension of output array

    Output
    ======
    supportmask: array
        composed binary mask where ellipses apertures are "1"
    ======
    author: ck/sg 2024

    """

    # Create support mask
    supportmask = np.zeros(shape)
    for i in range(len(support_coordinates)):
        center, height, width, angle = (
            support_coordinates[i][0],
            support_coordinates[i][1],
            support_coordinates[i][2],
            support_coordinates[i][3],
        )
        yy, xx = ellipse(center[1], center[0], height / 2, width / 2, rotation=-angle)
        supportmask[yy, xx] = 1

    return supportmask