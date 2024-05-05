BASEFOLDER = f"/das/home/{USER}/p{BEAMTIMEID}"


def get_im_id(run_nr, acq_nrs):
    fnames = []
    for acq_nr in acq_nrs:
        acq_str = "acq%04d.*.h5" % (acq_nr)

        # Get filename
        fname = join(
            "raw",
            run_nr,
            "data",
            acq_str,
        )
        fnames.append(fname)

    return fnames


def load_run(fnames):

    data = dict()
    N = len(fnames)

    with SFDataFiles(fnames[0]) as f:
        for key in mnemonics.keys():
            data[key] = f[mnemonics[key]].data
    if N > 1:
        for fname in fnames[1:]:
            with SFDataFiles(fname) as f:
                for key in mnemonics.keys():
                    # print(key, data[key], f[mnemonics[key]].data)
                    data[key] = np.concatenate((data[key], f[mnemonics[key]].data))
    return data


# Full image loading procedure
def load_processing(im_ids, crop=None):
    """
    Loads images, averaging of two individual images (scans in tango consist of two images),
    padding to square shape, Additional cropping (optional)
    """

    # Load image data
    data = load_run([join(BASEFOLDER, id) for id in im_ids])
    images = data["images"]

    ## Zeropad to get square shape
    # images = sup.padding(images)

    # Calculate mean
    if images.ndim > 2:
        image = np.mean(images, axis=0)
    else:
        image = images.copy()

    # Optional cropping
    if crop is not None:
        images = images[:, :crop, :crop]
        image = image[:crop, :crop]

    return data, image, images