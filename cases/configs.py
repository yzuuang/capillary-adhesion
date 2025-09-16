"""
- Read & save INI format files
- Parse content into parameter values as dict[str, dict[str, str]]
- Deserialise into class instances
"""

import os
import sys
import re
import functools
import itertools
import operator
from configparser import ConfigParser

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from a_package.modelling import Region, SelfAffineRoughness, wavevector_norm, PSD_to_height, CapillaryBridge
from a_package.solving import AugmentedLagrangian


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


def preview_surface_and_gap(
    grid_params: dict[str, str],
    upper_surface_params: dict[str, str],
    lower_surface_params: dict[str, str],
    trajectory_params: dict[str, str],
):
    region = get_region_specs(grid_params)
    h1 = match_shape_and_get_height(region, upper_surface_params)
    h0 = match_shape_and_get_height(region, lower_surface_params)

    border = [0, region.nx, 0, region.ny]

    fig, ax = plt.subplots()
    # image = ax.pcolormesh(region.xm/a, region.ym/a, h1/a, cmap='hot')
    image = ax.imshow(h1 / region.a, interpolation="bicubic", cmap="plasma", extent=border)
    fig.colorbar(image)

    fig, ax = plt.subplots()
    # gap at the minimal separation
    d_min = float(trajectory_params["min_separation"])
    g = h1 - h0 + d_min
    # image = ax.pcolormesh(region.xm/a, region.ym/a, gap/a, cmap='hot')
    image = ax.imshow(g / region.a, vmin=0, interpolation="bicubic", cmap="hot", extent=border)
    fig.colorbar(image)

    # Visual check before running
    plt.show()
    skip = input("Run simulation [Y/n]? ").lower() in ("n", "no")
    if skip:
        sys.exit(0)


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

    return Sweeps(sweep_specs)


class Sweeps:

    def __init__(self, sweep_specs: dict[tuple[str, str], np.ndarray]):
        self._specs = sweep_specs

    def __len__(self):
        return (
            0
            if not len(self._specs)
            else functools.reduce(operator.mul, (len(vals) for vals in self._specs.values()), 1)
        )

    def iter_config(self, config: dict[str, dict[str, str]]):
        for updates in self._iter_combos():
            for [key_pair, value] in updates:
                config[key_pair[0]][key_pair[1]] = str(value)
            yield config

    def _iter_combos(self):
        keys = list(self._specs.keys())
        values_combos = itertools.product(*(self._specs.values()))
        for values_combo in values_combos:
            yield zip(keys, values_combo)


def get_region_specs(grid_params: dict[str, str]):
    # grid
    a = float(grid_params["pixel_size"])
    N = int(grid_params["nb_pixels"])
    L = a * N
    region = Region(a, L, L, N, N)
    return region


def match_shape_and_get_height(region, surface_params: dict[str, str]):
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
    return shape_mapping[surface_shape](region, surface_params)


def _get_height_of_flat(region, surface_params: dict[str, str]):
    constant = float(surface_params["constant"])
    height = constant * np.ones([region.nx, region.ny])
    return height


def _get_height_of_tip(region, surface_params: dict[str, str]):
    R = float(surface_params["radius"])
    height = -np.sqrt(np.clip(R**2 - (region.xm - 0.5 * region.lx) ** 2 - (region.ym - 0.5 * region.ly) ** 2, 0, None))
    # set lowest point to zero
    height += np.amax(abs(height))
    return height


def _get_height_of_sinusoid(region, surface_params: dict[str, str]):
    wave_num = float(surface_params["wavenumber"])
    wave_amp = float(surface_params["amplitude"])
    xm = region.xm
    qx = (2 * np.pi / region.lx) * wave_num
    ym = region.ym
    qy = (2 * np.pi / region.ly) * wave_num
    height = wave_amp * np.cos(qx * xm) * np.cos(qy * ym)
    return height


def _get_height_of_rough_surface(region, surface_params: dict[str, str]):
    assert region.lx == region.ly
    # generate roughness PSD
    C0 = float(surface_params["prefactor"])
    nR = float(surface_params["rolloff_wavelength_pixels"])
    qR = (2 * np.pi) / (region.a * nR)  # roll-off wave vector
    nS = float(surface_params["cutoff_wavelength_pixels"])
    qS = (2 * np.pi) / (region.a * nS)  # cut-off
    H = float(surface_params["hurst_exponent"])
    roughness = SelfAffineRoughness(C0, qR, qS, H)
    q_2D = wavevector_norm(region.qx, region.qy)
    C_2D = roughness.mapto_psd(q_2D)
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
    height = PSD_to_height(C_2D, rng=rng)
    return height


def _get_height_of_pattern(region, surface_params: dict[str, str]):
    xm = region.xm
    ym = region.ym
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


def get_capillary(region: Region, capillary_params: dict[str, str], upper, lower):
    theta = (np.pi / 180) * float(capillary_params["contact_angle_degree"])
    eta = float(capillary_params["interface_thickness"])
    return CapillaryBridge(region, eta, theta, upper, lower)


def get_optimizer(optimzer_params: dict[str, str]):
    i_max = int(optimzer_params["max_nb_iters"])
    l_max = int(optimzer_params["max_nb_loops"])
    tol_conver = float(optimzer_params["tol_convergence"])
    tol_constr = float(optimzer_params["tol_constraints"])
    c_init = float(optimzer_params["init_penalty_weight"])
    return AugmentedLagrangian(i_max, l_max, tol_conver, tol_constr, c_init)
