import os
import sys

from a_package.routine import ProcessedResult
from a_package.storing import working_directory
from a_package.visualizing import *

import matplotlib.pyplot as plt

from utils.runtime import retrieve_run


def main():
    anim = create_overview_animation(sys.argv[1], sys.argv[2])
    plt.show()


def create_overview_animation(sim_case, run_id):
    # retrieve processed result
    run_dir = retrieve_run(sim_case, run_id)
    with working_directory(run_dir.results_dir, read_only=True) as store:
        pr = store.load("result", ProcessedResult)
    # create anime
    latexify_plot(15)
    anim = animate_droplet_evolution_with_curves(pr)
    # save it
    filename_base = os.path.join(run_dir.visuals_dir, f"overview---{sim_case}")
    anim.save(f"{filename_base}.mp4", writer="ffmpeg")
    # return the anime, so it can be shown in interactive run
    return anim


def animate_droplet_evolution(pr: ProcessedResult):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    def update_image(i_frame: int):
        ax.clear()
        plot_combined_topography(ax, get_capillary_state(pr, i_frame))
        ax.set_xlabel(r"Position $x/\eta$")
        ax.set_ylabel(r"Position $y/\eta$")        
        return ax.images

    n_step = len(pr.evolution.t_exec)
    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


def animate_droplet_evolution_with_curves(pr: ProcessedResult):
    # 1 Figure, split into two parts
    # - LHS, 1 ax to plot combined topography
    # - RHS, 3 rows of axs to plot energy, F_z, F_x + F_y curves respectively
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    sf1, sf2 = fig.subfigures(1, 2, width_ratios=[1, 1])
    axs = sf2.subplots(3, 1, sharex=True)
    axs_lhs = sf1.subplots(2, 1, sharex=True, height_ratios=[1, 4])
    axs = np.append(axs, axs_lhs)

    n_step = len(pr.evolution.t_exec)
    idx_row = pr.modelling.region.nx // 2
    a = pr.modelling.region.a

    # decides the view limit
    view_margin_scale = 0.05

    h_min = np.amin(pr.modelling.h2[idx_row, :]) / a
    h_max = (np.amax(pr.modelling.h1[idx_row, :]) + np.amax(pr.evolution.r[:, -1])) / a
    h_margin = view_margin_scale * (h_max - h_min)
    h_min = h_min - h_margin
    h_max = h_max + 10*h_margin  # for the legend

    e_min = np.amin(pr.evolution.E) / a**2
    e_max = np.amax(pr.evolution.E) / a**2
    e_margin = view_margin_scale * (e_max - e_min)
    e_min = e_min - e_margin
    e_max = e_max + e_margin

    F_n_min = np.amin(pr.evolution.Fz) / a
    F_n_max = np.amax(pr.evolution.Fz) / a
    F_n_margin = view_margin_scale * (F_n_max - F_n_min)
    F_n_min = F_n_min - F_n_margin
    F_n_max = F_n_max + F_n_margin

    P_min = np.amin(pr.evolution.P) / a
    P_max = np.amax(pr.evolution.P) / a
    P_margin = view_margin_scale * (P_max - P_min)
    P_min = P_min - P_margin
    P_max = P_max + P_margin

    def update_image(i_frame: int):
        for ax in axs:
            ax.clear()

        plot_gibbs_free_energy(axs[0], pr, i_frame + 1)
        axs[0].set_ylim(e_min, e_max)
        axs[0].set_ylabel(r"Energy $E/\gamma_\mathrm{lv} a^2$")
        axs[0].set_title("Evolution")

        plot_normal_force(axs[1], pr, i_frame + 1)
        axs[1].set_ylim(F_n_min, F_n_max)
        axs[1].set_ylabel(r"Normal force $F/\gamma_\mathrm{lv} a$")

        plot_perimeter(axs[2], pr, i_frame + 1)
        axs[2].set_ylim(P_min, P_max)
        axs[2].set_ylabel(r"Perimeter $P/a$")

        axs[2].set_xlim([0, n_step])
        axs[2].set_xlabel(r"Step (size=$0.1 a$)")

        plot_cross_section_sketch(axs[-2], get_capillary_state(pr, i_frame), idx_row)
        axs[-2].set_ylim(h_min, h_max)
        axs[-2].set_ylabel(r"Position $z/a$")
        axs[-2].set_title("Cross section")

        plot_combined_topography(axs[-1], get_capillary_state(pr, i_frame))
        axs[-1].axhline(pr.modelling.region.y[idx_row] / a, color="k")
        axs[-1].set_ylabel(r"Position $y/a$")
        axs[-1].set_title("Gap & Phase")

        axs[-1].set_xlabel(r"Position $x/a$")

        return [axs[0].lines, axs[1].lines, axs[2].lines, axs[-1].images]

    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


if __name__ == '__main__':
    main()
