"""
- Read & save INI format files
- Parse content into parameter values as dict[str, dict[str, str]]
- Deserialise into class instances
"""

import os
import re
from configparser import ConfigParser

import numpy as np
import numpy.random as random

from a_package.models import SelfAffineRoughness, psd_to_height, CapillaryBridge
from a_package.numeric import AugmentedLagrangian
from a_package.grid import Grid
from a_package.utils import Sweep


def read_config_files(files: list):
    if not len(files):
        raise RuntimeError(f"Provide at least one config file.")
    config = ConfigParser()
    for file in files:
        if not os.path.exists(file):
            raise RuntimeError(f"The file {file} doesn't exist.")
        print(f"Reading configs from {file}")
        config.read(file)
    return config


def save_config_to_file(config: ConfigParser, filepath):
    with open(filepath, "w", encoding="utf-8") as fp:
        config.write(fp)


def extract_sweeps(config: dict[str, dict[str, str]], prefix):
    # allow section name followed by digits in case of more than one parameters to sweep
    pattern = re.compile(f"{prefix}(\d+)?")
    # find all the sweep sections in the config
    sweep_sections = [section for section in config.keys() if pattern.fullmatch(section)]
    if not len(sweep_sections):
        return None
    # mapping literals to the method for constructing the parameter values
    method_mapping = {
        "linspace": np.linspace,
        "logspace": np.logspace,
    }
    # construct a mapping from the pair of names to access the parameter into an array of all values to sweep
    sweep_specs = {
        (config[section]["in_section"], config[section]["in_option"]): method_mapping[
            config[section].get("method", "linspace")
        ](float(config[section]["min_value"]), float(config[section]["max_value"]), int(config[section]["nb_steps"]))
        for section in sweep_sections
    }
    # remove sweep sections in the config
    for section in sweep_sections:
        del config[section]

    return Sweep(sweep_specs)


def create_grid(grid_params: dict[str, str]):
    # grid
    a = float(grid_params["pixel_size"])
    N = int(grid_params["nb_pixels"])
    L = a * N
    grid = Grid([L, L], [N, N])
    return grid


def match_shape_and_get_height(grid: Grid, surface_params: dict[str, str]):
    shape_mapping = {
        "flat": _get_height_of_flat,
        "tip": _get_height_of_tip,
        "sinusoid": _get_height_of_sinusoid,
        "rough": _get_height_of_rough_surface,
        "pattern": _get_height_of_pattern,
    }
    surface_shape = surface_params["shape"]
    if surface_shape not in shape_mapping:
        raise ValueError(f"Unknown surface shape {surface_shape}.")
    return shape_mapping[surface_shape](grid, surface_params)


# FIXME: now the model want the first dimension to always be number of components.
# Though for now, thanks to numpy, the shape will be casted to match that of phase-field.


def _get_height_of_flat(grid: Grid, surface_params: dict[str, str]):
    constant = float(surface_params["constant"])
    height = constant * np.ones(grid.nb_elements)
    return height


def _get_height_of_tip(grid: Grid, surface_params: dict[str, str]):
    R = float(surface_params["radius"])
    [lx, ly] = grid.lengths
    x_center = 0.5 * lx
    y_center = 0.5 * ly
    [x, y] = grid.form_nodal_mesh()
    height = -np.sqrt(np.clip(R**2 - (x - x_center) ** 2 - (y - y_center) ** 2, 0, None))
    # set lowest point to zero
    height += np.amax(abs(height))
    return height


def _get_height_of_sinusoid(grid: Grid, surface_params: dict[str, str]):
    wave_num = float(surface_params["wavenumber"])
    wave_amp = float(surface_params["amplitude"])
    [x, y] = grid.form_nodal_mesh()
    [qx, qy] = grid.form_spectral_mesh()
    height = wave_amp * np.cos(qx * x) * np.cos(qy * y)
    return height


def _get_height_of_rough_surface(grid: Grid, surface_params: dict[str, str]):
    # generate roughness PSD
    C0 = float(surface_params["prefactor"])
    nR = float(surface_params["rolloff_wavelength_pixels"])
    qR = (2 * np.pi) / (grid.element_sizes[0] * nR)  # roll-off wave vector
    nS = float(surface_params["cutoff_wavelength_pixels"])
    qS = (2 * np.pi) / (grid.element_sizes[0] * nS)  # cut-off
    H = float(surface_params["hurst_exponent"])
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = grid.form_spectral_mesh()
    # q_2D = np.expand_dims(np.stack([qx, qy], axis=0), axis=1)
    [_, C_2D] = roughness.mapto_isotropic_psd(q_2D)
    # get or generate the seed
    try:
        seed = int(surface_params["seed"])
    except KeyError:
        seed = None
    seq = random.SeedSequence(seed)
    # if seed is generated, print it out
    if seed is None:
        print(f"seed={seq.entropy}")
    # generate rough surface from PSD
    rng = random.default_rng(seq)
    height = psd_to_height(C_2D, rng=rng)
    return height.squeeze(axis=0)


def _get_height_of_pattern(grid: Grid, surface_params: dict[str, str]):
    [xm, ym] = grid.form_nodal_mesh()
    x0 = float(surface_params["tip_center_x"])
    y0 = float(surface_params["tip_center_y"])

    try:
        wave_len_L = float(surface_params["wave_len_L"])
        wave_amp_L = float(surface_params["wave_amp_L"])
        wave_L = wave_amp_L * np.cos(2 * np.pi / wave_len_L * (xm - x0)) * np.cos(2 * np.pi / wave_len_L * (ym - y0))
    except KeyError:
        wave_L = 0.0

    try:
        wave_len_M = float(surface_params["wave_len_M"])
        wave_amp_M = float(surface_params["wave_amp_M"])
        wave_M = wave_amp_M * np.cos(2 * np.pi / wave_len_M * (xm - x0)) * np.cos(2 * np.pi / wave_len_M * (ym - y0))
    except KeyError:
        wave_M = 0.0

    try:
        wave_len_S = float(surface_params["wave_len_S"])
        wave_amp_S = float(surface_params["wave_amp_S"])
        wave_S = wave_amp_S * np.cos(2 * np.pi / wave_len_S * (xm - x0)) * np.cos(2 * np.pi / wave_len_S * (ym - y0))
    except KeyError:
        wave_L = 0.0

    height = wave_L + wave_M + wave_S
    return np.atleast_2d(height)


def get_capillary_args(capillary_params: dict[str, str]):
    theta = (np.pi / 180) * float(capillary_params["contact_angle_degree"])
    eta = float(capillary_params["interface_thickness"])
    return dict(eta=eta, theta=theta)


def get_optimizer_args(optimzer_params: dict[str, str]):
    i_max = int(optimzer_params["max_nb_iters"])
    l_max = int(optimzer_params["max_nb_loops"])
    tol_conver = float(optimzer_params["tol_convergence"])
    tol_constr = float(optimzer_params["tol_constraints"])
    c_init = float(optimzer_params["init_penalty_weight"])
    return dict(
        max_inner_iter=i_max, max_outer_loop=l_max, tol_convergence=tol_conver, tol_constraint=tol_constr,
        init_penalty_weight=c_init)
