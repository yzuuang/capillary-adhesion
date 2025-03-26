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

    # anim = animate_droplet_evolution_with_curves(pr)
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
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    sf1, sf2 = fig.subfigures(1, 2, width_ratios=[3, 2])
    ax_lhs = sf1.subplots()
    axs = sf2.subplots(3, 1, sharex=True)
    axs = np.append(axs, ax_lhs)

    n_step = len(pr.evolution.t_exec)

    def update_image(i_frame: int):
        for ax in axs:
            ax.clear()

        plot_gibbs_free_energy(axs[0], pr, i_frame + 1)
        axs[0].set_ylabel(r"Energy $E/\gamma\eta^2$")
        axs[0].set_title('Energy evolution')

        plot_normal_force(axs[1], pr, i_frame + 1)
        axs[1].set_ylabel(r"Force $F/\gamma\eta$")
        axs[1].set_title('Force evolution')

        plot_shear_force(axs[2], pr, i_frame + 1)
        axs[2].set_ylabel(r"Force $F/\gamma\eta$")

        axs[2].set_xlim([0, n_step])
        axs[2].set_xlabel(r"Step (size=$0.01\eta$)")

        plot_combined_topography(axs[-1], get_capillary_state(pr, i_frame))
        axs[-1].set_xlabel(r"Position $x/\eta$")
        axs[-1].set_ylabel(r"Position $y/\eta$")

        return [axs[0].lines, axs[1].lines, axs[2].lines, axs[-1].images]

    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


if __name__ == '__main__':
    main()
