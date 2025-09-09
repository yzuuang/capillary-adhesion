import os
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt

from a_package.postprocessing import ProcessedResult
from a_package.storing import working_directory
from utils.runtime import RunDir


def main():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data"))

    # tartget_dir = os.path.join(data_root, "load_unload", "tip-over-flat", "250904-114054-7b567a")
    # [Fz, V] = get_multirun_Fz_V(RunDir(tartget_dir))
    # [fig, ax] = plt.subplots()
    # ax.plot(V, Fz, "x-", label="hydrophillic")
    # ax.set_ylabel("Adhesive forces $F_z$")
    # ax.set_xlabel("Liquid volume $V$")
    # ax.grid()
    # fig.savefig(os.path.join(data_root, "load_unload", "tip-over-flat-varying-volumes.svg"))

    # target_dir = os.path.join(data_root, "load_unload", "tip-over-sinusoid", "250904-132247-7b567a")
    # [Fz, n] = get_multirun_Fz_n(RunDir(target_dir))
    # [fig, ax] = plt.subplots()
    # ax.semilogx(n, Fz, "x-", label="hydrophillic")
    # ax.set_ylabel("Adhesive forces $F_z$")
    # ax.set_xlabel("Sinusoid wavenumber $n$")
    # ax.grid()
    # fig.savefig(os.path.join(data_root, "load_unload", "tip-over-sinusoid-varying-wavenumbers.svg"))

    target_dir = os.path.join(data_root, "load_unload", "tip-over-pattern", "250904-013422-7b567a")
    [Fz, ix, iy] = get_multirun_Fz_ix_iy(RunDir(target_dir))
    x_axis = np.unique(ix)
    y_axis = np.unique(iy)
    Fz = Fz.reshape(x_axis.size, y_axis.size)
    [fig, ax] = plt.subplots()
    image = ax.pcolormesh(x_axis, y_axis, Fz.T, shading="nearest")
    fig.colorbar(image)
    ax.set_ylabel("Tip center location $y$")
    ax.set_xlabel("Tip center location $x$")
    fig.savefig(os.path.join(data_root, "load_unload", "tip-over-pattern-varying-tip-center-locations.svg"))

    plt.show()


def get_multirun_Fz_V(run: RunDir):
    nb_subruns = len(os.listdir(run.intermediate_dir))
    forces = np.empty(nb_subruns)
    volumes = np.empty(nb_subruns)
    for index, subrun_path in enumerate(os.scandir(run.intermediate_dir)):
        subrun = RunDir(subrun_path)

        # get the force from result
        with working_directory(subrun.results_dir, read_only=True) as store:
            res: ProcessedResult = store.load("result", ProcessedResult)
        i_step = -5
        forces[index] = res.evolution.Fz[i_step]

        # get the volume from config
        config = ConfigParser()
        for params_file in os.scandir(subrun.parameters_dir):
            config.read(params_file)
        volumes[index] = config["Capillary"]["liquid_volume_percent"]

    return forces, volumes


def get_multirun_Fz_n(run: RunDir):
    config = ConfigParser()
    nb_subruns = len(os.listdir(run.intermediate_dir))
    forces = np.empty(nb_subruns)
    wavenumbers = np.empty(nb_subruns)
    for index, subrun_path in enumerate(os.scandir(run.intermediate_dir)):
        subrun = RunDir(subrun_path)

        # get the force from result
        with working_directory(subrun.results_dir, read_only=True) as store:
            res: ProcessedResult = store.load("result", ProcessedResult)
        i_step = -5
        forces[index] = res.evolution.Fz[i_step]

        # get the wavenumber from config
        for params_file in os.scandir(subrun.parameters_dir):
            config.read(params_file)
        wavenumbers[index] = config["LowerSurface"]["wavenumber"]

    return forces, wavenumbers


def get_multirun_Fz_ix_iy(run: RunDir):
    config = ConfigParser()
    nb_subruns = len(os.listdir(run.intermediate_dir))
    forces = np.empty(nb_subruns)
    tip_center_x = np.empty(nb_subruns)
    tip_center_y = np.empty(nb_subruns)
    for index, subrun_path in enumerate(os.scandir(run.intermediate_dir)):
        subrun = RunDir(subrun_path)

        # get the force from result
        with working_directory(subrun.results_dir, read_only=True) as store:
            res: ProcessedResult = store.load("result", ProcessedResult)
        i_step = -5
        forces[index] = res.evolution.Fz[i_step]

        # get the wavenumber from config
        for params_file in os.scandir(subrun.parameters_dir):
            config.read(params_file)
        tip_center_x[index] = config["LowerSurface"]["tip_center_x"]
        tip_center_y[index] = config["LowerSurface"]["tip_center_y"]

    return forces, tip_center_x, tip_center_y


if __name__ == "__main__":
    main()
