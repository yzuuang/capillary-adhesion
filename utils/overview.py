import os
import matplotlib.pyplot as plt

from a_package.routine import ProcessedResult
from a_package.storing import working_directory
from a_package.visualizing import *


# data_folder = "parallel_flat_plates.data"
# data_folder = "one_peak.data"
data_folder = "rough_surfaces_at_contact.data"
# data_folder = "rough_surfaces_slide.data"


def main():
    path = os.path.join(os.path.dirname(__file__), data_folder)
    with working_directory(path, read_only=True) as store:
        pr = store.load("Processed", "result", ProcessedResult)
    filename_base = __file__.replace(".py", f"---{data_folder}")

    latexify_plot(15)

    # anim = animate_droplet_evolution(pr)
    # anim.save(f"{filename_base}.mp4", writer="ffmpeg")

    [fig, ax] = plt.subplots(1, 1, figsize=(10, 3), constrained_layout=True)
    plot_normal_force(ax, pr, None)
    ax.set_ylabel(r"Force $F/\gamma\eta$")
    ax.set_xlabel('Simulation progress')
    fig.savefig(f"{filename_base}.svg", dpi=450)
    plt.show()


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
    idx_row = 64
    eta = pr.modelling.eta

    # decides the view limit
    view_margin_scale = 0.05

    h_min = np.amin(pr.modelling.h1[idx_row, :]) / eta
    h_max = (np.amax(pr.modelling.h2[idx_row, :]) + np.amax(pr.evolution.r[:, -1])) / eta
    h_margin = view_margin_scale * (h_max - h_min)
    h_min = h_min - h_margin
    h_max = h_max + 10*h_margin  # for the legend

    e_min = np.amin(pr.evolution.E) / eta**2
    e_max = np.amax(pr.evolution.E) / eta**2
    e_margin = view_margin_scale * (e_max - e_min)
    e_min = e_min - e_margin
    e_max = e_max + e_margin

    F_n_min = np.amin(pr.evolution.F[:, -1]) / eta
    F_n_max = np.amax(pr.evolution.F[:, -1]) / eta
    F_n_margin = view_margin_scale * (F_n_max - F_n_min)
    F_n_min = F_n_min - F_n_margin
    F_n_max = F_n_max + F_n_margin

    F_t_min = np.amin(pr.evolution.F[:, 0:2]) / eta
    F_t_max = np.amax(pr.evolution.F[:, 0:2]) / eta
    F_t_margin = view_margin_scale * (F_t_max - F_t_min)
    F_t_min = F_t_min - F_t_margin
    F_t_max = F_t_max + F_t_margin

    def update_image(i_frame: int):
        for ax in axs:
            ax.clear()

        plot_gibbs_free_energy(axs[0], pr, i_frame + 1)
        axs[0].set_ylim(e_min, e_max)
        axs[0].set_ylabel(r"Energy $E/\gamma_\mathrm{lv} a^2$")
        axs[0].set_title("Energy evolution")

        plot_normal_force(axs[1], pr, i_frame + 1)
        axs[1].set_ylim(F_n_min, F_n_max)
        axs[1].set_ylabel(r"Force $F/\gamma_\mathrm{lv} a$")
        axs[1].set_title("Force evolution")

        plot_shear_force(axs[2], pr, i_frame + 1)
        axs[2].set_ylim(F_t_min, F_t_max)
        axs[2].set_ylabel(r"Force $F/\gamma_\mathrm{lv} a$")

        axs[2].set_xlim([0, n_step])
        axs[2].set_xlabel(r"Step (size=$0.1 a$)")

        plot_cross_section_sketch(axs[-2], get_capillary_state(pr, i_frame), idx_row)
        axs[-2].set_ylim(h_min, h_max)
        axs[-2].set_ylabel(r"Position $z/a$")
        axs[-2].set_title("Cross section")

        plot_combined_topography(axs[-1], get_capillary_state(pr, i_frame))
        axs[-1].axhline(pr.modelling.region.y[idx_row] / eta, color="k")
        axs[-1].set_ylabel(r"Position $y/a$")
        axs[-1].set_title("Gap & Phase")

        axs[-1].set_xlabel(r"Position $x/a$")

        return [axs[0].lines, axs[1].lines, axs[2].lines, axs[-1].images]

    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


if __name__ == '__main__':
    main()
