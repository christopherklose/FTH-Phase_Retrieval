"""
Python library for Phase retrieval in Python using functions. Functions taken and adapted from code base of:

Riccardo Battistelli, Daniel Metternich, Michael Schneider, Lisa-Marie Kern, Kai Litzius, Josefin Fuchs, Christopher Klose, Kathinka Gerlinger, Kai Bagschik, Christian M. Günther, Dieter Engel, Claus Ropers, Stefan Eisebitt, Bastian Pfau, Felix Büttner, and Sergey Zayko, "Coherent x-ray magnetic imaging with 5 nm resolution," Optica 11, 234-237 (2024)


2020 - Original Code
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)

2026 - Refactoring
@authors: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import logging

log = logging.getLogger(__name__)

import os
import time

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt

from scipy import stats

#############################################################
#       GPU handling
# ############################################################

try:
    import cupy as cp

    GPU = cp.is_available()
    log.info("CUDA GPU available.")
except ImportError:
    log.warning(
        "Could not import cupy module (is it installed?). "
        "Proceeding with CPU support only."
    )
    GPU = False
except Exception as ex:
    log.warning(
        f"Error determining GPU availability: {ex}. "
        "Proceeding with CPU support only."
    )
    GPU = False

# Use cupyx or scipy fft functions
if GPU:
    import cupy as xp
    from cupyx.scipy.fft import fft2, ifft2
else:
    import numpy as xp
    import scipy.fft as fft
    from scipy.fft import fft2, ifft2

    # Change number of workers fot fft
    def fft2(array, **kwargs):
        return fft.fft2(array, workers=os.cpu_count(), **kwargs)

    def ifft2(array, **kwargs):
        return fft.ifft2(array, workers=os.cpu_count(), **kwargs)


def to_numpy(array, xp):
    """
    Convert xp array to NumPy safely.
    """
    if xp.__name__ == "cupy":
        return array.get()
    return array


#############################################################
#       PHASE RETRIEVAL Algorithm
# ############################################################


def default_phase_retrieval_recipe():
    """
    Returns dict that contains default phase retrieval setup parameter:
        - algorithm_list_full_coherence : list[str] (length multiple of 3)
        - number_iterations_full_coherence : list[int] (length multiple of 3)
        - algorithm_list_partial_coherence : list[str] (length multiple of 3)
        - number_iterations_partial_coherence : list[int] (length multiple of 3)
        - use_partial_coherence_algorithm : bool
        - hologram_intensity_cutoff_vmin : float
        - Startimage : np.ndarray or None
        - Startgamma : np.ndarray or None
        - partial_coherence_nr_iterations_per_RL_cycle : int
        - partial_coherence_frequency_of_RL_cycles : int

    ------
    author: CK 2026
    """

    recipe = {
        "algorithm_list_full_coherence": ["HAPRE", "ER", "ER"],
        "number_iterations_full_coherence": [700, 50, 50],
        "algorithm_list_partial_coherence": ["HAPRE", "ER", "ER"],
        "number_iterations_partial_coherence": [700, 50, 50],
        "use_partial_coherence_algorithm": True,
        "hologram_intensity_cutoff_vmin": 1,
        "Startimage": None,
        "Startgamma": None,
        "partial_coherence_nr_iterations_per_RL_cycle": 50,
        "partial_coherence_frequency_of_RL_cycles": 20,
    }

    return recipe


def phase_retrieval_algorithm(
    pos: ArrayLike,
    neg: ArrayLike,
    mask_pixel: ArrayLike,
    supportmask: ArrayLike,
    phase_retrieval_recipe=None,
):
    """
    Iterative phase retrieval (full coherence + optional partial coherence refinement)
    for positive/negative helicity holograms.

    Parameters
    ----------
    pos, neg : array_like
        2D hologram intensity images (same shape). Values may contain NaNs.
    mask_pixel : array_like
        2D mask (same shape as pos/neg). Convention assumed:
        - mask_pixel == 0 : valid pixels
        - mask_pixel != 0 : masked pixels
    supportmask : array_like
        2D support mask (same shape as pos/neg). Used to generate a default start image.
    phase_retrieval_recipe : dict, optional
        Overrides for algorithm and iteration settings. Supported keys include:
        - algorithm_list_full_coherence : list[str] (length multiple of 3)
        - number_iterations_full_coherence : list[int] (length multiple of 3)
        - algorithm_list_partial_coherence : list[str] (length multiple of 3)
        - number_iterations_partial_coherence : list[int] (length multiple of 3)
        - use_partial_coherence_algorithm : bool
        - hologram_intensity_cutoff_vmin : float
        - Startimage : np.ndarray or None
        - Startgamma : np.ndarray or None
        - partial_coherence_nr_iterations_per_RL_cycle : int
        - partial_coherence_frequency_of_RL_cycles : int

    Returns
    -------
    retrieved_p: ArrayLike
        Phase retrieved positive helicity hologram with full coherence assumption
    retrieved_n: ArrayLike
        Phase retrieved negative helicity hologram with full coherence assumption
    retrieved_p_pc: ArrayLike
        Phase retrieved positive helicity hologram with partial coherence assumption
    retrieved_n_pc: ArrayLike
        Phase retrieved negative helicity hologram with partial coherence assumption
    bsmask_p: ArrayLike
        mask of invalid values for positive helicity
    bsmask_n: ArrayLike
        mask of invalid values for negative helicity
    gamma_p: ArrayLike
        mutual coherence function positive helicity
    gamma_n: ArrayLike
        mutual coherence function negative helicity
    -------
    author: CK 2026
    """

    # ----------------------------
    # Recipe: defaults + overrides
    # ----------------------------
    recipe = default_phase_retrieval_recipe()
    if phase_retrieval_recipe:
        recipe.update(phase_retrieval_recipe)

    _verify_valid_algorithm_list(
        recipe["algorithm_list_full_coherence"],
        recipe["number_iterations_full_coherence"],
        name="Full-coherence",
    )

    _verify_valid_algorithm_list(
        recipe["algorithm_list_partial_coherence"],
        recipe["number_iterations_partial_coherence"],
        name="Partial-coherence",
    )

    # ----------------------------
    # Prepare inputs
    # ----------------------------
    pos_input = pos.copy()
    neg_input = neg.copy()

    # Baseline subtract (ignore zeros)
    vmin = recipe["hologram_intensity_cutoff_vmin"]
    if vmin > 0:
        vals = pos_input[pos_input != 0]
        if vals.size:
            mi, _ = np.nanpercentile(vals, [vmin, 99.9])
            pos_input = pos_input - mi

        vals = neg_input[neg_input != 0]
        if vals.size:
            mi, _ = np.nanpercentile(vals, [vmin, 99.9])
            neg_input = neg_input - mi

    # fill nan, then clip to >= 0
    pos_input = np.where(np.isnan(pos_input), 0, pos_input)
    neg_input = np.where(np.isnan(neg_input), 0, neg_input)
    pos_input = np.clip(pos_input, 0, None)
    neg_input = np.clip(neg_input, 0, None)

    # Beamstop masks: inherit mask_pixel plus intensity<=0
    bsmask_p = mask_pixel.copy()
    bsmask_p[pos_input <= 0] = 1
    bsmask_n = mask_pixel.copy()
    bsmask_n[neg_input <= 0] = 1

    # Precompute amplitudes used repeatedly
    pos_amp = np.sqrt(pos_input)
    neg_amp = np.sqrt(neg_input)

    # Normalization factor for initializing negative helicity phase
    pos_sum = float(np.sum(pos_input))
    neg_sum = float(np.sum(neg_input))
    norm_factor = np.sqrt(neg_sum / pos_sum) if pos_sum > 0 else 1.0

    # ----------------------------
    # Initialize Startimage/Startgamma
    # ----------------------------
    if recipe["Startimage"] is None:
        Startimage = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(supportmask)))
    else:
        Startimage = recipe["Startimage"].copy()

    if recipe["Startgamma"] is None:
        Startgamma = np.ones(pos_input.shape, dtype=float) * 1e-6 * 2
        Startgamma[pos_input.shape[0] // 2, pos_input.shape[1] // 2] = 0.7
    else:
        Startgamma = recipe["Startgamma"].copy()

    # Roughly normalize Startimage to match data scale using unmasked pixels
    valid_pix = (mask_pixel == 0) & (pos_input > 0)
    if np.any(valid_pix):
        x = np.sqrt(pos_input[valid_pix]).ravel()
        y = np.abs(Startimage[valid_pix]).ravel()
        if x.size >= 2:
            res = stats.linregress(x, y)
            Startimage = Startimage - res.intercept
            if abs(res.slope) > 1e-12:
                Startimage = Startimage / res.slope

    # ----------------------------
    # Execute Phase Retrieval
    # ----------------------------
    start_time = time.time()

    retrieved_p = retrieved_n = None
    retrieved_p_pc = retrieved_n_pc = None
    gamma_p = gamma_n = None

    # Initialize error lists
    error_p_it_1, error_p_it_2, error_n_it_3 = [], [], []
    error_p_pc_it_1, error_p_pc_it_2, error_n_pc_it_3 = [], [], []

    for step in range(0, len(recipe["number_iterations_full_coherence"]), 3):
        print("############ -   CDI Full Coherence")

        # Positive helicity - beta_mode="arctan"
        retrieved_p, Error_diff_p, Error_supp = PhaseRtrv_GPU(
            diffract=pos_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step],
            beta_mode="arctan",
            plot_every=349,
            Phase=Startimage,
            seed=False,
            real_object=False,
            bsmask=bsmask_p,
            average_img=30,
            Fourier_last=True,
        )

        # Positive helicity - beta_mode="const"
        retrieved_p, Error_diff_p2, Error_supp = PhaseRtrv_GPU(
            diffract=pos_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step + 1],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step + 1],
            beta_mode="const",
            plot_every=24,
            Phase=retrieved_p,
            seed=False,
            real_object=False,
            bsmask=bsmask_p,
            average_img=30,
            Fourier_last=True,
        )

        # Negative helicity - beta_mode="const"
        retrieved_n, Error_diff_n2, Error_supp = PhaseRtrv_GPU(
            diffract=neg_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step + 2],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step + 2],
            beta_mode="const",
            plot_every=24,
            Phase=retrieved_p * norm_factor,
            seed=False,
            real_object=False,
            bsmask=bsmask_n,
            average_img=30,
            Fourier_last=True,
        )

        # Append errors to lists
        error_p_it_1.extend(Error_diff_p)
        error_p_it_2.extend(Error_diff_p2)
        error_n_it_3.extend(Error_diff_n2)

        print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
        Startimage = retrieved_p.copy()

        if recipe["use_partial_coherence_algorithm"]:
            print("############   -   CDI Partial Coherence")

            # Replace beamstop region with current reconstruction intensity
            pos_pc_input = (np.abs(retrieved_p) ** 2) * bsmask_p + pos_input * (
                1 - bsmask_p
            )
            neg_pc_input = (np.abs(retrieved_n) ** 2) * bsmask_n + neg_input * (
                1 - bsmask_n
            )

            # Partial coherence: positive (arctan)
            retrieved_p_pc, Error_diff_p_pc, Error_supp, gamma_p = PhaseRtrv_with_RL(
                diffract=np.sqrt(pos_pc_input),
                mask=supportmask,
                mode=recipe["algorithm_list_partial_coherence"][step],
                beta_zero=0.5,
                Nit=recipe["number_iterations_partial_coherence"][step],
                beta_mode="arctan",
                gamma=Startgamma,
                RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                plot_every=349,
                Phase=Startimage,
                seed=False,
                real_object=False,
                bsmask=np.zeros_like(bsmask_p),
                average_img=30,
                Fourier_last=True,
            )

            # Partial coherence: positive (const)
            retrieved_p_pc, Error_diff_p_pc2, Error_supp, gamma_p = PhaseRtrv_with_RL(
                diffract=np.sqrt(pos_pc_input),
                mask=supportmask,
                mode=recipe["algorithm_list_partial_coherence"][step + 1],
                beta_zero=0.5,
                Nit=recipe["number_iterations_partial_coherence"][step + 1],
                beta_mode="const",
                gamma=gamma_p,
                RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                plot_every=24,
                Phase=retrieved_p_pc,
                seed=False,
                real_object=False,
                bsmask=np.zeros_like(bsmask_p),
                average_img=30,
                Fourier_last=True,
            )

            # Partial coherence: negative (const) (use gamma_p as in original)
            retrieved_n_pc, Error_diff_n_pc2, Error_supp, gamma_n = PhaseRtrv_with_RL(
                diffract=np.sqrt(neg_pc_input),
                mask=supportmask,
                mode=recipe["algorithm_list_partial_coherence"][step + 2],
                beta_zero=0.5,
                Nit=recipe["number_iterations_partial_coherence"][step + 2],
                beta_mode="const",
                gamma=gamma_p,
                RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                plot_every=24,
                Phase=retrieved_p_pc * norm_factor,
                seed=False,
                real_object=False,
                bsmask=np.zeros_like(bsmask_n),
                average_img=30,
                Fourier_last=True,
            )

            # Append errors to lists
            error_p_pc_it_1.extend(Error_diff_p_pc)
            error_p_pc_it_2.extend(Error_diff_p_pc2)
            error_n_pc_it_3.extend(Error_diff_n_pc2)

            print("--- %s seconds ---" % np.round((time.time() - start_time), 2))

            Startimage = retrieved_p_pc.copy()
            Startgamma = gamma_p.copy()
        else:
            retrieved_p_pc = retrieved_p.copy()
            retrieved_n_pc = retrieved_n.copy()
            gamma_p = Startgamma.copy()
            gamma_n = Startgamma.copy()

    # Create dictionary for error data
    error = {
        "error_p_it_1": np.asarray(error_p_it_1),
        "error_p_it_2": np.asarray(error_p_it_2),
        "error_n_it_3": np.asarray(error_n_it_3),
    }

    if recipe["use_partial_coherence_algorithm"]:
        error.update(
            {
                "error_p_pc_it_1": np.asarray(error_p_pc_it_1),
                "error_p_pc_it_2": np.asarray(error_p_pc_it_2),
                "error_n_pc_it_3": np.asarray(error_n_pc_it_3),
            }
        )

    print("Phase Retrieval Done!")

    return (
        retrieved_p,
        retrieved_n,
        retrieved_p_pc,
        retrieved_n_pc,
        bsmask_p,
        bsmask_n,
        gamma_p,
        gamma_n,
        error,
    )


def single_helicity_phase_retrieval_algorithm(
    pos: ArrayLike,
    mask_pixel: ArrayLike,
    supportmask: ArrayLike,
    phase_retrieval_recipe=None,
):
    """
    Iterative phase retrieval (full coherence + optional partial coherence refinement)
    for single helicity (called pos) holograms.

    Parameters
    ----------
    pos: array_like
        2D hologram intensity images. Values may contain NaNs.
    mask_pixel : array_like
        2D mask (same shape as pos). Convention assumed:
        - mask_pixel == 0 : valid pixels
        - mask_pixel != 0 : masked pixels
    supportmask : array_like
        2D support mask (same shape as pos). Used to generate a default start image.
    phase_retrieval_recipe : dict, optional
        Overrides for algorithm and iteration settings. Supported keys include:
        - algorithm_list_full_coherence : list[str] (length multiple of 3)
        - number_iterations_full_coherence : list[int] (length multiple of 3)
        - algorithm_list_partial_coherence : list[str] (length multiple of 3)
        - number_iterations_partial_coherence : list[int] (length multiple of 3)
        - use_partial_coherence_algorithm : bool
        - hologram_intensity_cutoff_vmin : float
        - Startimage : np.ndarray or None
        - Startgamma : np.ndarray or None
        - partial_coherence_nr_iterations_per_RL_cycle : int
        - partial_coherence_frequency_of_RL_cycles : int

    Returns
    -------
    retrieved_p: ArrayLike
        Phase retrieved positive helicity hologram with full coherence assumption
    retrieved_p_pc: ArrayLike
        Phase retrieved positive helicity hologram with partial coherence assumption
    bsmask_p: ArrayLike
        mask of invalid values for positive helicity
    gamma_p: ArrayLike
        mutual coherence function positive helicity
    -------
    author: CK 2026
    """

    # ----------------------------
    # Recipe: defaults + overrides
    # ----------------------------
    recipe = default_phase_retrieval_recipe()
    if phase_retrieval_recipe:
        recipe.update(phase_retrieval_recipe)

    _verify_valid_algorithm_list(
        recipe["algorithm_list_full_coherence"],
        recipe["number_iterations_full_coherence"],
        name="Full-coherence",
    )

    _verify_valid_algorithm_list(
        recipe["algorithm_list_partial_coherence"],
        recipe["number_iterations_partial_coherence"],
        name="Partial-coherence",
    )

    # ----------------------------
    # Prepare inputs
    # ----------------------------
    pos_input = pos.copy()

    # Baseline subtract (ignore zeros)
    vmin = recipe["hologram_intensity_cutoff_vmin"]
    if vmin > 0:
        vals = pos_input[pos_input != 0]
        if vals.size:
            mi, _ = np.nanpercentile(vals, [vmin, 99.9])
            pos_input = pos_input - mi

    # fill nan, then clip to >= 0
    pos_input = np.where(np.isnan(pos_input), 0, pos_input)
    pos_input = np.clip(pos_input, 0, None)

    # Beamstop masks: inherit mask_pixel plus intensity<=0
    bsmask_p = mask_pixel.copy()
    bsmask_p[pos_input <= 0] = 1

    # Precompute amplitudes used repeatedly
    pos_amp = np.sqrt(pos_input)

    # ----------------------------
    # Initialize Startimage/Startgamma
    # ----------------------------
    if recipe["Startimage"] is None:
        Startimage = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(supportmask)))
    else:
        Startimage = recipe["Startimage"].copy()

    if recipe["Startgamma"] is None:
        Startgamma = np.ones(pos_input.shape, dtype=float) * 1e-6 * 2
        Startgamma[pos_input.shape[0] // 2, pos_input.shape[1] // 2] = 0.7
    else:
        Startgamma = recipe["Startgamma"].copy()

    # Roughly normalize Startimage to match data scale using unmasked pixels
    valid_pix = (mask_pixel == 0) & (pos_input > 0)
    if np.any(valid_pix):
        x = np.sqrt(pos_input[valid_pix]).ravel()
        y = np.abs(Startimage[valid_pix]).ravel()
        if x.size >= 2:
            res = stats.linregress(x, y)
            Startimage = Startimage - res.intercept
            if abs(res.slope) > 1e-12:
                Startimage = Startimage / res.slope

    # ----------------------------
    # Execute Phase Retrieval
    # ----------------------------
    start_time = time.time()

    retrieved_p = None
    retrieved_p_pc = None
    gamma_p = None

    for step in range(0, len(recipe["number_iterations_full_coherence"]), 3):
        print("############ -   CDI Full Coherence")

        # Positive helicity - beta_mode="arctan"
        retrieved_p, Error_diff_p, Error_supp = PhaseRtrv_GPU(
            diffract=pos_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step],
            beta_mode="arctan",
            plot_every=349,
            Phase=Startimage,
            seed=False,
            real_object=False,
            bsmask=bsmask_p,
            average_img=30,
            Fourier_last=True,
        )

        # Positive helicity - beta_mode="const"
        retrieved_p, Error_diff_p2, Error_supp = PhaseRtrv_GPU(
            diffract=pos_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step + 1],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step + 1],
            beta_mode="const",
            plot_every=24,
            Phase=retrieved_p,
            seed=False,
            real_object=False,
            bsmask=bsmask_p,
            average_img=30,
            Fourier_last=True,
        )

        print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
        Startimage = retrieved_p.copy()

        if recipe["use_partial_coherence_algorithm"]:
            print("############   -   CDI Partial Coherence")

            # Replace beamstop region with current reconstruction intensity
            pos_pc_input = (np.abs(retrieved_p) ** 2) * bsmask_p + pos_input * (
                1 - bsmask_p
            )

            # Partial coherence: positive (arctan)
            retrieved_p_pc, Error_diff_p_pc, Error_supp, gamma_p = PhaseRtrv_with_RL(
                diffract=np.sqrt(pos_pc_input),
                mask=supportmask,
                mode=recipe["algorithm_list_partial_coherence"][step],
                beta_zero=0.5,
                Nit=recipe["number_iterations_partial_coherence"][step],
                beta_mode="arctan",
                gamma=Startgamma,
                RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                plot_every=349,
                Phase=Startimage,
                seed=False,
                real_object=False,
                bsmask=np.zeros_like(bsmask_p),
                average_img=30,
                Fourier_last=True,
            )

            # Partial coherence: positive (const)
            retrieved_p_pc, Error_diff_p_pc2, Error_supp, gamma_p = PhaseRtrv_with_RL(
                diffract=np.sqrt(pos_pc_input),
                mask=supportmask,
                mode=recipe["algorithm_list_partial_coherence"][step + 1],
                beta_zero=0.5,
                Nit=recipe["number_iterations_partial_coherence"][step + 1],
                beta_mode="const",
                gamma=gamma_p,
                RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                plot_every=24,
                Phase=retrieved_p_pc,
                seed=False,
                real_object=False,
                bsmask=np.zeros_like(bsmask_p),
                average_img=30,
                Fourier_last=True,
            )

            print("--- %s seconds ---" % np.round((time.time() - start_time), 2))

            Startimage = retrieved_p_pc.copy()
            Startgamma = gamma_p.copy()
        else:
            retrieved_p_pc = retrieved_p.copy()
            gamma_p = Startgamma.copy()

    print("Phase Retrieval Done!")

    return (
        retrieved_p,
        retrieved_p_pc,
        bsmask_p,
        gamma_p,
    )


def phase_retrieval_algorithm_on_second_helicity_only(
    new_helicity: ArrayLike,
    topo: ArrayLike,
    retrieved_topo: ArrayLike,
    retrieved_topo_pc: ArrayLike,
    gamma_topo: ArrayLike,
    mask_pixel: ArrayLike,
    supportmask: ArrayLike,
    phase_retrieval_recipe=None,
):
    """
    Iterative phase retrieval which will be applied to the second helicity only,
    taking the already retrieved phases from a previous routine

    Parameters
    ----------
    topo, new_helicity : array_like
        2D hologram intensity images (same shape). Values may contain NaNs.
    mask_pixel : array_like
        2D mask (same shape as topo/new_helicity). Convention assumed:
        - mask_pixel == 0 : valid pixels
        - mask_pixel != 0 : masked pixels
    supportmask : array_like
        2D support mask (same shape as topo/new_helicity). Used to generate a default start image.
    phase_retrieval_recipe : dict, optional
        Overrides for algorithm and iteration settings. Supported keys include:
        - algorithm_list_full_coherence : list[str] (length multiple of 3)
        - number_iterations_full_coherence : list[int] (length multiple of 3)
        - algorithm_list_partial_coherence : list[str] (length multiple of 3)
        - number_iterations_partial_coherence : list[int] (length multiple of 3)
        - use_partial_coherence_algorithm : bool
        - hologram_intensity_cutoff_vmin : float
        - Startimage : np.ndarray or None
        - Startgamma : np.ndarray or None
        - partial_coherence_nr_iterations_per_RL_cycle : int
        - partial_coherence_frequency_of_RL_cycles : int

    Returns
    -------
    retrieved_new: ArrayLike
        Phase retrieved new helicity hologram with full coherence assumption
    retrieved_new_pc: ArrayLike
        Phase retrieved new helicity hologram with partial coherence assumption
    bsmask_new: ArrayLike
        mask of invalid values for positive helicity
    gamma_new: ArrayLike
        mutual coherence function positive helicity
    -------
    author: CK 2026
    """

    # ----------------------------
    # Recipe: defaults + overrides
    # ----------------------------
    recipe = default_phase_retrieval_recipe()
    if phase_retrieval_recipe:
        recipe.update(phase_retrieval_recipe)

    _verify_valid_algorithm_list(
        recipe["algorithm_list_full_coherence"],
        recipe["number_iterations_full_coherence"],
        name="Full-coherence",
    )

    _verify_valid_algorithm_list(
        recipe["algorithm_list_partial_coherence"],
        recipe["number_iterations_partial_coherence"],
        name="Partial-coherence",
    )

    # ----------------------------
    # Prepare inputs
    # ----------------------------
    topo_input = topo.copy()
    new_helicity_input = new_helicity.copy()

    # Baseline subtract (ignore zeros)
    vmin = recipe["hologram_intensity_cutoff_vmin"]
    if vmin > 0:
        vals = topo_input[topo_input != 0]
        if vals.size:
            mi, _ = np.nanpercentile(vals, [vmin, 99.9])
            topo_input = topo_input - mi

        vals = new_helicity_input[new_helicity_input != 0]
        if vals.size:
            mi, _ = np.nanpercentile(vals, [vmin, 99.9])
            new_helicity_input = new_helicity_input - mi

    # fill nan, then clip to >= 0
    topo_input = np.where(np.isnan(topo_input), 0, topo_input)
    new_helicity_input = np.where(np.isnan(new_helicity_input), 0, new_helicity_input)
    topo_input = np.clip(topo_input, 0, None)
    new_helicity_input = np.clip(new_helicity_input, 0, None)

    # Beamstop masks: inherit mask_pixel plus intensity<=0
    bsmask_topo = mask_pixel.copy()
    bsmask_topo[topo_input <= 0] = 1
    bsmask_new = mask_pixel.copy()
    bsmask_new[new_helicity_input <= 0] = 1

    # Precompute amplitudes
    new_helicity_amp = np.sqrt(new_helicity_input)

    # Normalization factor for initializing new helicity phase
    topo_sum = float(np.sum(topo_input))
    new_helicity_sum = float(np.sum(new_helicity_input))
    norm_factor = np.sqrt(new_helicity_sum / topo_sum) if topo_sum > 0 else 1.0

    # ----------------------------
    # Initialize Startimage/Startgamma
    # ----------------------------
    if recipe["Startimage"] is None:
        Startimage = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(supportmask)))
    else:
        Startimage = recipe["Startimage"].copy()

    if recipe["Startgamma"] is None:
        Startgamma = np.ones(topo_input.shape, dtype=float) * 1e-6 * 2
        Startgamma[topo_input.shape[0] // 2, topo_input.shape[1] // 2] = 0.7
    else:
        Startgamma = recipe["Startgamma"].copy()

    # Roughly normalize Startimage to match data scale using unmasked pixels
    valid_pix = (mask_pixel == 0) & (topo_input > 0)
    if np.any(valid_pix):
        x = np.sqrt(topo_input[valid_pix]).ravel()
        y = np.abs(Startimage[valid_pix]).ravel()
        if x.size >= 2:
            res = stats.linregress(x, y)
            Startimage = Startimage - res.intercept
            if abs(res.slope) > 1e-12:
                Startimage = Startimage / res.slope

    # ----------------------------
    # Execute Phase Retrieval
    # ----------------------------
    start_time = time.time()

    retrieved_new = retrieved_new_pc = None
    gamma_new = None

    for step in range(0, len(recipe["number_iterations_full_coherence"]), 3):
        print("############ -   CDI Full Coherence")

        # New helicity - beta_mode="const"
        retrieved_new, Error_diff_new2, Error_supp = PhaseRtrv_GPU(
            diffract=new_helicity_amp,
            mask=supportmask,
            mode=recipe["algorithm_list_full_coherence"][step + 2],
            beta_zero=0.5,
            Nit=recipe["number_iterations_full_coherence"][step + 2],
            beta_mode="const",
            plot_every=24,
            Phase=retrieved_topo * norm_factor,
            seed=False,
            real_object=False,
            bsmask=bsmask_new,
            average_img=30,
            Fourier_last=True,
        )

        print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
        Startimage = retrieved_topo.copy()

        if recipe["use_partial_coherence_algorithm"]:
            print("############   -   CDI Partial Coherence")

            # Replace beamstop region with current reconstruction intensity
            new_helicity_pc_input = (
                np.abs(retrieved_new) ** 2
            ) * bsmask_new + new_helicity_input * (1 - bsmask_new)

            # Partial coherence: new helicity (const) (use gamma_topo as in original)
            retrieved_new_pc, Error_diff_new_pc2, Error_supp, gamma_new = (
                PhaseRtrv_with_RL(
                    diffract=np.sqrt(new_helicity_pc_input),
                    mask=supportmask,
                    mode=recipe["algorithm_list_partial_coherence"][step + 2],
                    beta_zero=0.5,
                    Nit=recipe["number_iterations_partial_coherence"][step + 2],
                    beta_mode="const",
                    gamma=gamma_topo,
                    RL_freq=recipe["partial_coherence_frequency_of_RL_cycles"],
                    RL_it=recipe["partial_coherence_nr_iterations_per_RL_cycle"],
                    plot_every=24,
                    Phase=retrieved_topo_pc * norm_factor,
                    seed=False,
                    real_object=False,
                    bsmask=np.zeros_like(bsmask_new),
                    average_img=30,
                    Fourier_last=True,
                )
            )

            print("--- %s seconds ---" % np.round((time.time() - start_time), 2))

            Startimage = retrieved_topo_pc.copy()
            Startgamma = gamma_topo.copy()
        else:
            retrieved_new_pc = retrieved_new.copy()
            gamma_new = Startgamma.copy()

    print("Phase Retrieval Done!")

    return (
        retrieved_new,
        retrieved_new_pc,
        bsmask_new,
        gamma_new,
    )


def plot_phase_retrieval_errors(error, phase_retrieval_recipe, ax=None):
    """
    Plot tracked phase retrieval errors and return concatenated error list.

    Returns
    -------
    full_error_list : list
        Concatenated list of all tracked errors.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    # Initialize default algorithm lists if it is not provided
    default_recipe = default_phase_retrieval_recipe()
    for key in ["algorithm_list_full_coherence", "algorithm_list_partial_coherence"]:
        phase_retrieval_recipe.setdefault(key, default_recipe[key])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    counter = 0
    full_error_list = []

    full_keys = ["error_p_it_1", "error_p_it_2", "error_n_it_3"]
    pc_keys = ["error_p_pc_it_1", "error_p_pc_it_2", "error_n_pc_it_3"]

    # -------------------------
    # Full coherence
    # -------------------------
    for i, key in enumerate(full_keys):
        if key not in error:
            continue

        alg = phase_retrieval_recipe["algorithm_list_full_coherence"][i]
        label = f"Full coherence - {alg}"

        error_list = np.asarray(error[key])

        full_error_list.extend(error_list.tolist())

        x = counter + np.arange(len(error_list))
        counter = x[-1] + 1

        ax.plot(x, error_list, label=label)

    # -------------------------
    # Partial coherence
    # -------------------------
    if phase_retrieval_recipe["use_partial_coherence_algorithm"]:
        for i, key in enumerate(pc_keys):
            if key not in error:
                continue

            alg = phase_retrieval_recipe["algorithm_list_partial_coherence"][i]
            label = f"Partial coherence - {alg}"

            error_list = np.asarray(error[key])

            full_error_list.extend(error_list.tolist())

            x = counter + np.arange(len(error_list))
            counter = x[-1] + 1

            ax.plot(x, error_list, label=label)

    # -------------------------
    # Final formatting
    # -------------------------
    if len(full_error_list) > 0:
        ax.set_title(
            f"Smallest Error: {np.min(full_error_list):.2f} dB, Final error: {full_error_list[-1]:.2f} dB"
        )

    ax.legend()
    ax.set_xlabel("Tracked errors")
    ax.set_ylabel("log(Error) [dB]")
    ax.grid(True)

    return full_error_list, fig, ax


#############################################################
#       PHASE RETRIEVAL FUNCTIONS HELPER
# ############################################################


def _verify_valid_algorithm_list(
    algorithms: list, iterations: list, name="Full-coherence"
):
    """
    Validate algorithm and iteration lists for phase retrieval.

    Parameters
    ----------
    algorithms : sequence of str
        Algorithm names.
    iterations : sequence of int
        Iteration counts corresponding to algorithms.
    name : str
        Descriptive name used in error messages.

    Raises
    ------
    ValueError
        If validation fails.
    ------
    author: CK 2026
    """

    allowed_algorithms = {
        "ER",
        "SF",
        "HAPRE",
        "RAAR",
        "HIOs",
        "HIO",
        "OSS",
        "CHIO",
        "HPR",
    }

    # Length consistency
    if len(algorithms) != len(iterations):
        raise ValueError(
            f"{name}: algorithm list and iteration list must have the same length."
        )

    # Block structure check
    if len(algorithms) % 3 != 0:
        raise ValueError(
            f"{name}: lists must have length that is a multiple of 3 "
            "(3 steps per block)."
        )

    # Iteration validity
    if not all(isinstance(n, int) and n > 0 for n in iterations):
        raise ValueError(f"{name}: all iteration counts must be positive integers.")

    # Algorithm validity
    invalid = [a for a in algorithms if a not in allowed_algorithms]
    if invalid:
        raise ValueError(
            f"{name}: invalid algorithm(s) detected: {invalid}. "
            f"Allowed algorithms are: {sorted(allowed_algorithms)}"
        )


# ----------------------------
# Beta schedule (CPU-side)
# ----------------------------
def _beta_const(Nit, beta_zero):
    return np.full(Nit, beta_zero, dtype=np.float64)


def _beta_arctan(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (
        0.5 - np.arctan((step - min(Nit / 2, 700)) / (0.15 * Nit)) / np.pi
    ) * (0.98 - beta_zero)


def _beta_smoothstep(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    start = Nit // 50
    end = Nit - Nit // 10
    denom = max(end - start, 1.0)
    y = (step - start) / denom
    beta = 1 - (1 - beta_zero) * (6 * y**5 - 15 * y**4 + 10 * y**3)
    beta[:start] = 1
    beta[end:] = 0
    return beta


def _beta_sigmoid(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    x0 = Nit // 20
    alpha = 1 / (Nit * 0.15)
    return 1 - (1 - beta_zero) / (1 + np.exp(-(step - x0) * alpha))


def _beta_exp(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (1 - beta_zero) * (1 - np.exp(-((step / 7) ** 3)))


def _beta_linear_to_beta_zero(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    return 1 + (beta_zero - 1) / Nit * step


def _beta_linear_to_1(Nit, beta_zero):
    step = np.arange(Nit, dtype=np.float64)
    return beta_zero + (1 - beta_zero) / Nit * step


BETA_SCHEDULES = {
    "const": _beta_const,
    "arctan": _beta_arctan,
    "smoothstep": _beta_smoothstep,
    "sigmoid": _beta_sigmoid,
    "exp": _beta_exp,
    "linear_to_beta_zero": _beta_linear_to_beta_zero,
    "linear_to_1": _beta_linear_to_1,
}


def make_beta_schedule(beta_mode, Nit, beta_zero):
    """
    Create beta schedule of length Nit.

    beta_mode can be:
      - a numpy array of length Nit (returned as float64 view/copy),
      - or a string key in BETA_SCHEDULES.
    -------
    author: CK 2026
    """
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")

    if isinstance(beta_mode, np.ndarray):
        if beta_mode.shape[0] != Nit:
            raise ValueError("If beta_mode is an array, it must have length Nit.")
        return beta_mode.astype(np.float64, copy=False)

    try:
        fn = BETA_SCHEDULES[beta_mode]
    except KeyError:
        raise ValueError(
            f"Unknown beta_mode '{beta_mode}'. Allowed: {sorted(BETA_SCHEDULES)}"
        )

    beta = fn(Nit, beta_zero)
    if beta.shape[0] != Nit:
        raise RuntimeError("Beta schedule returned wrong length.")
    return beta


# ----------------------------
# Real-space projection steps
# ----------------------------
def _proj_ER(inv, prev, mask, beta, step_idx, Nit):
    return inv * mask


def _proj_SF(inv, prev, mask, beta, step_idx, Nit):
    return inv * (2 * mask - 1)


def _proj_hapre(inv, prev, mask, beta, step_idx, Nit):
    return inv + beta * (prev - 2 * inv) * (1 - mask)


def _proj_RAAR(inv, prev, mask, beta, step_idx, Nit):
    return inv + beta * (prev - 2 * inv) * (1 - mask) * (2 * inv - prev < 0)


def _proj_HIOs(inv, prev, mask, beta, step_idx, Nit):
    return inv + (1 - mask) * (prev - (beta + 1) * inv)


def _proj_HIO(inv, prev, mask, beta, step_idx, Nit):
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(inv) < 0)
    )


def _proj_OSS(inv, prev, mask, beta, step_idx, Nit):
    inv2 = (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(inv) < 0)
    )
    l = inv2.shape[0]
    alpha = l - (l - 1 / l) * xp.floor(step_idx / Nit * 10) / 10
    smoothed = ifft2(W(inv2.shape[0], inv2.shape[1], alpha) * fft2(inv2))
    return inv2 * mask + (1 - mask) * smoothed


def _proj_CHIO(inv, prev, mask, beta, step_idx, Nit):
    alpha = 0.4
    return (
        (prev - beta * inv)
        + mask * (xp.real(inv - alpha * prev) >= 0) * (-prev + (beta + 1) * inv)
        + (xp.real(-inv + alpha * prev) >= 0)
        * (xp.real(inv) >= 0)
        * ((beta - (1 - alpha) / alpha) * inv)
    )


def _proj_HPR(inv, prev, mask, beta, step_idx, Nit):
    alpha = 0.4
    return (
        inv
        + (1 - mask) * (prev - (beta + 1) * inv)
        + mask * (prev - (beta + 1) * inv) * (xp.real(prev - (beta - 3) * inv) > 0)
    )


PROJECTIONS = {
    "ER": _proj_ER,
    "SF": _proj_SF,
    "HAPRE": _proj_hapre,
    "RAAR": _proj_RAAR,
    "HIOs": _proj_HIOs,
    "HIO": _proj_HIO,
    "OSS": _proj_OSS,
    "CHIO": _proj_CHIO,
    "HPR": _proj_HPR,
}


# ----------------------------
# Other Subprocesses
# ----------------------------
def RL(Idelta, Iexp, gamma_cp, RL_it):
    """
    Richardson–Lucy update loop (CuPy).

    Parameters
    ----------
    Idelta : cupy.ndarray
        Intensity delta term (2D).
    Iexp : cupy.ndarray
        Expected intensity / measurement term (2D).
    gamma_cp : cupy.ndarray
        Current PSF / mutual optical intensity estimate (2D).
    RL_it : int
        Number of RL iterations.

    Returns
    -------
    gamma_cp : cupy.ndarray
        Updated and normalized gamma estimate.
    --------
    author: RB 2020
    """
    # Precompute FFTs of Idelta and its flipped version
    Id = fft2(Idelta)
    Id_flip = fft2(Idelta[::-1, ::-1])

    for _ in range(RL_it):
        # update
        gamma_cp = gamma_cp * ifft2(Id_flip * fft2(Iexp / (ifft2(Id * fft2(gamma_cp)))))

        # normalize (once per iteration)
        gamma_cp /= xp.nansum(gamma_cp)

    return gamma_cp


#############################################################
#       MAIN PHASE RETRIEVAL FUNCTIONS
# ############################################################


# Full coherence phase retrieval algorithm
def PhaseRtrv_GPU(
    diffract,
    mask,
    mode="ER",
    Nit=500,
    beta_zero=0.5,
    beta_mode="const",
    Phase=None,
    seed=False,
    plot_every=20,
    bsmask=None,
    real_object=False,  # kept for API compatibility (not used)
    average_img=10,
    Fourier_last=True,
):
    """
    Iterative phase retrieval with GPU acceleration (CuPy).

    Parameters
    ----------
    diffract : array_like (2D)
        Far-field amplitude target (same shape as mask).
    mask : array_like (2D)
        Support mask in real space (0/1). Same shape as diffract.
    mode : str
        Algorithm: ER, SF, mine, RAAR, HIOs, HIO, OSS, CHIO, HPR.
    Nit : int
        Number of iterations.
    beta_zero : float
        Base beta parameter.
    beta_mode : str or np.ndarray
        Beta schedule name or explicit array of length Nit.
    Phase : array_like (2D complex) or None
        Initial Fourier-domain guess. If None, random start is used.
    seed : bool
        If True, uses fixed RNG seed for reproducibility.
    plot_every : int
        Interval for computing/storing diffraction error.
    bsmask : array_like (2D) or None
        Beamstop / floating pixel mask in Fourier domain. 1 = unconstrained pixel.
    average_img : int
        Number of best guesses near the end to average.
    Fourier_last : bool
        Apply Fourier constraint one last time before returning.

    Returns
    -------
    guess : np.ndarray (2D complex)
        Final reconstructed Fourier-domain field (ifftshifted back).
    Error_diffr_list : list
        Sampled diffraction errors over iterations.
    Error_supp_list : list
        Support errors (kept for compatibility; not filled). Future implementation shrink wrap
    ------
    author: CK 2026
    """

    diffract = np.asarray(diffract)
    mask = np.asarray(mask)

    # Basic format handling
    if diffract.ndim != 2 or mask.ndim != 2:
        raise ValueError("diffract and mask must be 2D arrays.")
    if diffract.shape != mask.shape:
        raise ValueError("diffract and mask must have the same shape.")
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")
    if average_img <= 0:
        raise ValueError("average_img must be > 0.")
    if plot_every <= 0:
        raise ValueError("plot_every must be > 0.")

    l, n = diffract.shape

    if bsmask is None:
        bsmask = np.zeros((l, n), dtype=np.float32)
    else:
        bsmask = np.asarray(bsmask)
        if bsmask.shape != (l, n):
            raise ValueError("bsmask must have same shape as diffract.")

    # Get projection function
    proj_fn = PROJECTIONS.get(mode)
    if proj_fn is None:
        raise ValueError(f"Invalid mode '{mode}'. Allowed: {sorted(PROJECTIONS)}")

    # Beta schedule
    Beta = make_beta_schedule(beta_mode, Nit, beta_zero)

    # Initial guess
    if seed:
        np.random.seed(0)

    if Phase is None:
        Phase = np.exp(1j * np.random.rand(l, n) * np.pi * 2)
    Phase = np.asarray(Phase)
    if Phase.shape != (l, n):
        raise ValueError("Phase must have same shape as diffract.")

    # Initial guess
    guess = (1 - bsmask) * diffract * np.exp(1j * np.angle(Phase)) + Phase * bsmask

    # Shift to corner convention (as in original)
    bsmask = np.fft.fftshift(bsmask)
    guess = np.fft.fftshift(guess)
    mask = np.fft.fftshift(mask)
    diffract = np.fft.fftshift(diffract)

    # Move to GPU
    BSmask_cp = xp.asarray(bsmask)
    guess_cp = xp.asarray(guess)
    mask_cp = xp.asarray(mask)
    diffract_cp = xp.asarray(diffract)

    Error_diffr_list = []
    Error_supp_list = []

    # Initialize prev
    prev = fft2(
        (1 - BSmask_cp) * diffract_cp * xp.exp(1j * xp.angle(guess_cp))
        + guess_cp * BSmask_cp
    )

    # Track best guesses robustly
    Best_guess = xp.zeros((average_img, l, n), dtype=xp.complex64)
    Best_error = xp.full((average_img,), xp.inf, dtype=xp.float64)
    start_best_at = max(2, Nit - average_img * 2)

    for s in range(Nit):
        beta = float(Beta[s])

        # Fourier constraint outside beamstop
        guess_cp = xp.where(
            BSmask_cp,
            guess_cp,
            diffract_cp * xp.exp(1j * xp.angle(guess_cp)),
        )

        # Real space
        inv = fft2(guess_cp)

        # Projection step via dispatch table
        inv = proj_fn(inv, prev, mask_cp, beta, s, Nit)

        prev = inv.copy()

        # Back to Fourier space
        guess_cp = ifft2(inv)

        # Compute diffraction error sometimes + keep best guesses near end
        if s <= 2 or (s % plot_every == 0) or (s >= start_best_at):
            err = Error_diffract_cp(
                xp.abs(guess_cp) * (1 - BSmask_cp), diffract_cp * (1 - BSmask_cp)
            )
            Error_diffr_list.append(err)

            if s >= start_best_at:
                j = int(xp.argmax(Best_error).item())
                if err < Best_error[j]:
                    Best_error[j] = err
                    Best_guess[j, :, :] = guess_cp

    # Average best guesses (only slots that were filled)
    filled = xp.isfinite(Best_error)
    if bool(filled.any()):
        guess_cp = xp.mean(Best_guess[filled], axis=0)
    # else: keep the last iterate guess_cp

    # Apply Fourier constraint one last time
    if Fourier_last:
        guess_cp = (1 - BSmask_cp) * diffract_cp * xp.exp(
            1j * xp.angle(guess_cp)
        ) + guess_cp * BSmask_cp

    guess = to_numpy(guess_cp, xp)
    return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list


# Partial coherence phase retrieval algorithm
def PhaseRtrv_with_RL(
    diffract,
    mask,
    mode="ER",
    Nit=500,
    beta_zero=0.5,
    beta_mode="const",
    gamma=None,
    RL_freq=25,
    RL_it=20,
    Phase=None,
    seed=False,
    plot_every=20,
    bsmask=None,
    real_object=False,  # kept for API compatibility (not used)
    average_img=10,
    Fourier_last=True,
):
    """
    Iterative phase retrieval with GPU acceleration and Richardson–Lucy updates (partial coherence).

    This refactor applies the same improvements as in PhaseRtrv_GPU:
      - projection step dispatched by mode via a dict
      - beta schedule creation via helper
      - avoids constructing big temporaries in the Fourier constraint by updating only outside beamstop
      - robust "best guesses" accumulation (no undefined Best_guess)

    Returns
    -------
    guess : np.ndarray (2D complex)
    Error_diffr_list : list
    Error_supp_list : list
    gamma : np.ndarray (2D complex or real, depending on your RL implementation)
    ------
    author: CK 2026
    """

    diffract = np.asarray(diffract)
    mask = np.asarray(mask)

    # Format handling
    if diffract.ndim != 2 or mask.ndim != 2:
        raise ValueError("diffract and mask must be 2D arrays.")
    if diffract.shape != mask.shape:
        raise ValueError("diffract and mask must have the same shape.")
    if Nit <= 0:
        raise ValueError("Nit must be > 0.")
    if average_img <= 0:
        raise ValueError("average_img must be > 0.")
    if plot_every <= 0:
        raise ValueError("plot_every must be > 0.")
    if RL_freq <= 0:
        raise ValueError("RL_freq must be > 0.")
    if RL_it <= 0:
        raise ValueError("RL_it must be > 0.")
    if gamma is None:
        raise ValueError("gamma must be provided for PhaseRtrv_with_RL.")

    l, n = diffract.shape

    if bsmask is None:
        bsmask = np.zeros((l, n), dtype=np.float32)
    else:
        bsmask = np.asarray(bsmask)
        if bsmask.shape != (l, n):
            raise ValueError("bsmask must have same shape as diffract.")

    # Get projection function
    proj_fn = PROJECTIONS.get(mode)
    if proj_fn is None:
        raise ValueError(f"Invalid mode '{mode}'. Allowed: {PROJECTIONS.keys()}")

    # Load beta scheduler
    Beta = make_beta_schedule(beta_mode, Nit, beta_zero)

    if seed:
        np.random.seed(0)

    # In your original RL code, Phase is used as the initial Fourier-domain guess directly.
    if Phase is None:
        Phase = np.exp(1j * np.random.rand(l, n) * np.pi * 2)
        Phase = (1 - bsmask) * diffract * np.exp(1j * np.angle(Phase)) + Phase * bsmask

    guess = np.array(Phase, copy=True)

    # Shift everything to corner convention
    bsmask = np.fft.fftshift(bsmask)
    guess = np.fft.fftshift(guess)
    mask = np.fft.fftshift(mask)
    diffract = np.fft.fftshift(diffract)

    gamma = np.fft.fftshift(np.asarray(gamma))

    # Move to GPU
    BSmask_cp = xp.asarray(bsmask).astype(xp.bool_)
    obs = ~BSmask_cp  # outside beamstop (constrained)
    guess_cp = xp.asarray(guess)
    mask_cp = xp.asarray(mask)
    diffract_cp = xp.asarray(diffract)

    gamma_cp = xp.asarray(gamma)
    gamma_cp /= xp.sum(gamma_cp)

    Error_diffr_list = []
    Error_supp_list = []

    # Initial convolved intensity
    # convolved = ifft2( fft2(|guess|^2) * fft2(gamma) )
    convolved = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))

    # Initialize prev: keep your original structure
    prev = fft2(
        (1 - BSmask_cp) * diffract_cp / xp.sqrt(convolved) * guess_cp
        + guess_cp * BSmask_cp
    )

    # Best guesses storage (robust)
    Best_guess = xp.zeros((average_img, l, n), dtype=xp.complex64)
    Best_gamma = xp.zeros((average_img, l, n), dtype=xp.complex64)
    Best_error = xp.full((average_img,), xp.inf, dtype=xp.float64)
    start_best_at = max(2, Nit - average_img * 2)

    for s in range(Nit):
        beta = float(Beta[s])

        # ---- Fourier constraint without big temporaries ----
        # update only outside beamstop:
        factor = diffract_cp / xp.sqrt(convolved)
        guess_cp[obs] *= factor[obs]

        # ---- Real space ----
        inv = fft2(guess_cp)

        # ---- Projection step ----
        inv = proj_fn(inv, prev, mask_cp, beta, s, Nit)
        prev = inv.copy()

        # ---- Back to Fourier space ----
        new_guess = ifft2(inv)

        # ---- RL update for gamma ----
        if s > RL_freq and (s % RL_freq == 0):
            convolved_new = ifft2(fft2(xp.abs(new_guess) ** 2) * fft2(gamma_cp))
            Idelta = 2 * xp.abs(new_guess) ** 2 - xp.abs(guess_cp) ** 2
            I_exp = obs * (xp.abs(diffract_cp) ** 2) + convolved_new * BSmask_cp
            gamma_cp = RL(Idelta=Idelta, Iexp=I_exp, gamma_cp=gamma_cp, RL_it=RL_it)

        guess_cp = new_guess

        # Update convolved for next step
        convolved = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))

        # ---- Errors and best selection ----
        if s <= 2 or (s % plot_every == 0) or (s >= start_best_at):
            err = Error_diffract_cp(obs * xp.abs(diffract_cp) ** 2, obs * convolved)
            Error_diffr_list.append(err)

            if s >= start_best_at:
                j = int(xp.argmax(Best_error).item())
                if err < Best_error[j]:
                    Best_error[j] = err
                    Best_guess[j, :, :] = guess_cp
                    Best_gamma[j, :, :] = gamma_cp

    # Average best guesses (only slots that were filled)
    filled = xp.isfinite(Best_error)
    if bool(filled.any()):
        guess_cp = xp.mean(Best_guess[filled], axis=0)
        gamma_cp = xp.mean(Best_gamma[filled], axis=0)
    # else: keep the last iterate guess_cp / gamma_cp

    # Apply Fourier constraint one last time
    if Fourier_last:
        convolved_last = ifft2(fft2(xp.abs(guess_cp) ** 2) * fft2(gamma_cp))
        factor_last = diffract_cp / xp.sqrt(convolved_last)
        guess_cp[obs] *= factor_last[obs]

    guess = to_numpy(guess_cp, xp)
    gamma = to_numpy(gamma_cp, xp)

    return (
        np.fft.ifftshift(guess),
        Error_diffr_list,
        Error_supp_list,
        np.fft.ifftshift(gamma),
    )


#############################################################
#    FILTER FOR OSS
# ############################################################
def W(npx, npy, alpha=0.1):
    """
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: number of pixels on the image
            alpha: width of the gaussian

    OUTPUT: gaussian matrix

    --------
    author: RB 2020
    """
    Y, X = xp.meshgrid(xp.arange(npy), xp.arange(npx))
    k = xp.sqrt((X - npx // 2) ** 2 + (Y - npy // 2) ** 2)
    return xp.fft.fftshift(xp.exp(-0.5 * (k / alpha) ** 2))


#############################################################
#    ERROR FUNCTIONS
# ############################################################


def Error_diffract_cp(guess, diffract):
    """
    Error on the diffraction attern of retrieved data.
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns

    OUTPUT: Error between the two

    --------
    author: RB 2020
    """
    Num = xp.abs(diffract - guess) ** 2
    Den = xp.abs(diffract) ** 2
    Error = Num.sum() / Den.sum()
    Error = 10 * xp.log10(Error)
    return to_numpy(Error, xp)
