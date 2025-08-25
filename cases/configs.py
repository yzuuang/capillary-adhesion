import os
import sys
import configparser

import numpy as np
import matplotlib.pyplot as plt

from a_package.modelling import Region, SelfAffineRoughness, wavevector_norm, PSD_to_height


def read_config_files(files: list):
    if not len(files):
        raise RuntimeError(f"Provide at least one config file.")
    config = configparser.ConfigParser()
    for file in files:
        if not os.path.exists(file):
            raise RuntimeError(f"The file {file} doesn't exist.")
        print(f"Reading configs from {file}")
        config.read(file)
    return config


def preview_surface_and_gap(config: dict[str, dict[str, str]]):
    region = get_region_specs(config["Grid"])
    h1 = match_shape_and_get_height(region, config["UpperSurface"])
    h0 = match_shape_and_get_height(region, config["LowerSurface"])

    border = [0, region.nx, 0, region.ny]

    fig, ax = plt.subplots()
    # image = ax.pcolormesh(region.xm/a, region.ym/a, h1/a, cmap='hot')
    image = ax.imshow(h1 / region.a, interpolation="bicubic", cmap="plasma", extent=border)
    fig.colorbar(image)

    fig, ax = plt.subplots()
    # gap at the minimal separation
    d_min = float(config["Trajectory"]["min_separation"])
    g = h1 - h0 + d_min
    # image = ax.pcolormesh(region.xm/a, region.ym/a, gap/a, cmap='hot')
    image = ax.imshow(g / region.a, vmin=0, interpolation="bicubic", cmap="hot", extent=border)
    fig.colorbar(image)

    # Visual check before running
    plt.show()
    skip = input("Run simulation [Y/n]? ").lower() in ("n", "no")
    if skip:
        sys.exit(0)


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
    # generate rough surface from PSD
    seed = surface_params.get("seed", None)
    height = PSD_to_height(C_2D, seed=seed)
    return height


def _get_height_of_pattern(region, surface_params: dict[str, str]):
    xm = region.xm
    ym = region.ym

    try:
        wave_len_L = float(surface_params['wave_len_L'])
        wave_amp_L = float(surface_params['wave_amp_L'])
        wave_L = wave_amp_L * np.cos(2*np.pi / wave_len_L * xm) * np.cos(2*np.pi / wave_len_L * ym)
    except KeyError:
        wave_L = 0.

    try:
        wave_len_M = float(surface_params['wave_len_M'])
        wave_amp_M = float(surface_params['wave_amp_M'])
        wave_M = wave_amp_M * np.cos(2*np.pi / wave_len_M * xm) * np.cos(2*np.pi / wave_len_M * ym)
    except KeyError:
        wave_M = 0.

    try:
        wave_len_S = float(surface_params['wave_len_S'])
        wave_amp_S = float(surface_params['wave_amp_S'])
        wave_S = wave_amp_S * np.cos(2*np.pi / wave_len_S * xm) * np.cos(2*np.pi / wave_len_S * ym)
    except KeyError:
        wave_L = 0.

    height = wave_L + wave_M + wave_S
    return np.atleast_2d(height)
