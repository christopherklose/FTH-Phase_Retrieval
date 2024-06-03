def frame_list_to_acq_and_idx(frame_list, stack_length):
    """
    Convert a continous list of frame index that span multiple acq stacks into
    its corresponding acq numbers and frames, e.g. for stack_length = 100:
    frame_list = [99,100,101] --> acq_nrs = [1,2], frame_index_list = [[99],[0,1]]
    """
    # Calc quotient of frame_list and acq stack length to find relevant acq idx
    acq_idx = np.array(np.divmod(frame_list, stack_length))

    # Setup lists
    acq_nrs = []
    frame_index_list = []

    # Group frames that correspond to identical acq_stackd to minimize data access
    for i in np.unique(acq_idx[0]):
        acq_nrs.append(i + 1)  # First stack is acq0001
        frame_idx = np.where(acq_idx[0] == i)
        frame_index_list.append(acq_idx[1][frame_idx])

    return np.array(acq_nrs), frame_index_list


def list_data_filenames(run_nr,BASEFOLDER,  search_key="*"):
    """
    Returns a list of ALL data files that correspond to the
    given run number and contain the search key
    """

    # Convert run number to string
    if type(run_nr) == int:
        run_nr = "*%04d*" % run_nr

    # Find folder that corresponds to run number
    folder = glob(join(BASEFOLDER, "raw", run_nr))[0]
    print("Found folder: %s" % folder)

    # Get sorted list of files in folder
    files = sorted(glob(join(folder, "data", search_key)))

    return files, folder


def list_acquisition_filenames(run_nr,BASEFOLDER, acq_nrs=[], ONLY_CAMERA=False):
    """
    Returns a list of data files for the given acquisition
    numbers
    """

    # Load only camera files?
    if ONLY_CAMERA:
        search_key = "*CAMERAS.h5"
    else:
        search_key = "*"

    # If for run_nr is only gives a int number
    if type(run_nr) == int:
        run_nr = "*%04d*" % run_nr

    # If list is empty all files are loaded, only specific acquisition nrs
    # otherwise
    if len(acq_nrs) == 0:
        fnames, _ = list_data_filenames(run_nr,BASEFOLDER, search_key=search_key)
    elif len(acq_nrs) > 0:
        _, folder = list_data_filenames(run_nr,BASEFOLDER,  search_key=search_key)

        fnames = []
        for acq_nr in acq_nrs:
            acq_str = f"*{acq_nr:04d}{search_key}"

            # Get filename pattern
            fname_pattern = join(
                folder,
                "data",
                acq_str,
            )

            fnames.append(glob(fname_pattern))

    # Flatten potential nested list
    fnames_flattened = flatten_list(fnames)

    return fnames_flattened


def load_run(fnames, mnemonics):
    """
    Load all relevant data that are specified in the mnemonics dict (bugged)
    """

    data = dict()
    N = len(fnames)

    with SFDataFiles(fnames[0]) as f:
        for key in mnemonics.keys():
            try:
                data[key] = f[mnemonics[key]].data
            except:
                pass
    if N > 1:
        for fname in tqdm(fnames[1:]):
            with SFDataFiles(fname) as f:
                for key in mnemonics.keys():
                    try:
                        data[key] = np.concatenate((data[key], f[mnemonics[key]].data))
                    except:
                        pass
    return data


def load_images(fnames, loadmode="avg", n_jobs=1, crop=0):
    """
    Loads images from list of filenames.
    """

    # Setup
    images = []
    N = len(fnames)

    # Cropping necessary?
    if crop == 0:
        slice_crop = slice(0, -1)
    else:
        slice_crop = slice(crop, -crop)

    # Define loader function based on required loadmode
    if loadmode == "frames":

        def loader(fname):
            with SFDataFiles(fnames[0]) as f:
                image_stack = f[mnemonics["images"]][:, slice_crop, slice_crop].data
            return image_stack

    elif loadmode == "avg":

        def loader(fname):
            with SFDataFiles(fname) as f:
                image = np.mean(
                    f[mnemonics["images"]][:, slice_crop, slice_crop].data, axis=0
                )
            return image

    print(f"Start loading images with {n_jobs} parallel processes.")
    t0 = time.time()
    images = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(loader)(fname) for fname in fnames
    )
    print(f"Elapsed time: {time.time()-t0} seconds.")

    return np.array(drop_inhomogenous_part(images))


def load_specific_frames(fnames, indexes, crop=0):
    """
    Load only specific frames for a list of filenames
    """

    # Setup
    images = []

    # Cropping necessary?
    if crop == 0:
        slice_crop = slice(0, -1)
    else:
        slice_crop = slice(crop, -crop)

    # Loop over different fnames and indices
    for i, fname in enumerate(fnames):
        # Load only relevant frames from file
        with SFDataFiles(fname) as f:
            image = f[mnemonics["images"]][indexes[i], slice_crop, slice_crop].data
        images.append(image)

    return np.vstack(images)


# Full image loading procedure
def load_processing_frames(fnames, loadmode="avg", crop=0, frame_index_list=[]):
    """
    Loads images, calc average over all images,
    padding to square shape, Additional cropping (optional)
    """
    # Basic loading of stacks into list
    if not frame_index_list:
        images = load_images(fnames, loadmode, n_jobs=NR_JOBS, crop=crop)
    else:
        images = load_specific_frames(fnames, frame_index_list, crop=crop)

    # Bring to square shape
    images = make_square_shape(images)
    gc.collect()

    # Calculate mean
    if images.ndim > 2:
        image = np.mean(images, axis=0)
    else:
        image = images.copy()

    return image, images