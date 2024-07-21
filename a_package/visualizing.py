import dataclasses as dc

import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.modelling import Region, SelfAffineRoughness


@dc.dataclass
class DropletData:
    V: float         # volume of the droplet
    eta: float       # interfacial width
    L: float         # length of the plate
    M: int           # number of pixels along x-axis
    N: int           # number of pixels along y-axis
    phi: np.ndarray  # phase-field in 2D-array
    h1: np.ndarray   # roughness of the 1 plate in 2D-array
    h2: np.ndarray   # roughness of the 2 plate in 2D-array
    d: float         # displacement between the baselines of two plates
    x: np.ndarray    # grid locations along x-axis in 1D-array
    y: np.ndarray    # grid locations along y-axis in 1D-array
    dx: float        # size of pixels along x-axis
    dy: float        # size of pixels along y-axis


@dc.dataclass
class Record:
    data: list[DropletData]  # think as if taking snapshots
    init_guess: np.ndarray


eps = 1e-2  # cut off value to decide one phase


def plot_cross_section_sketch(ax: plt.Axes, data: DropletData, idx_row: int):
    h1 = (data.h1[idx_row,:] + data.d) / data.eta
    h2 = data.h2[idx_row,:] / data.eta
    x = data.x / data.eta

    # add border values due to periodic boundary condition
    h1 = np.append(h1, h1[0])
    h2 = np.append(h2, h2[0])
    x = np.append(x, data.L/data.eta)

    # change penetration to contact
    at_contact = np.where(h1 < h2)
    h1[at_contact] = h2[at_contact]

    # plot the two plates
    ax.plot(x, h1, "k-")
    ax.plot(x, h2, "k-")

    # highlight the contacting part
    # TODO: consider when the contact line is composed of multiple curves
    ax.plot(x[at_contact], h2[at_contact], "r.")

    # draw the water phase
    phi = data.phi[idx_row,:]
    water_phase = np.where(phi >= 1-eps)
    # TODO: consider when the water phase is composed of multiple sections
    ax.fill_between(x[water_phase], h2[water_phase], h1[water_phase], color="C0")

    ax.set_xlim(x[0], x[-1])


def plot_cross_section_phase_field(ax: plt.Axes, data: DropletData, idx_row: int):
    phi = data.phi[idx_row,:]
    x_dimensionless = data.x / data.eta
    ax.plot(x_dimensionless, phi)


def plot_phase_field_topography(ax: plt.Axes, data: DropletData):
    g = (data.h1 - data.h2 + data.d) / data.eta
    phi = data.phi.copy()
    # NOTE: the value 2.0 is for solid phase, while that of the water / vapor phase is 1.0 / 0.0 respectively.
    # these values are to match the color in the "afmhot" colormap.
    value_contact = 2.0
    phi[g < 0] = value_contact
    L = data.L / data.eta
    ax.imshow(phi, vmin=0.0, vmax=value_contact, cmap="afmhot", extent=[0., L, 0., L])


def plot_roughness_topography(ax: plt.Axes, data: DropletData):
    h = data.h1 / data.eta
    L = data.L / data.eta
    ax.imshow(h, cmap="coolwarm", extent=[0, L, 0, L])


def plot_phase_field_over_gap_topography(ax: plt.Axes, data: DropletData):
    L = data.L / data.eta
    border = [0, L, 0, L]

    # plot the gaps
    g_dimensionless = (data.h1 - data.h2 + data.d) / data.eta
    ax.imshow(g_dimensionless, cmap="coolwarm", extent=border)

    # plot the water/vapor interface
    phi = data.phi.copy()
    phi_perimeter = np.ma.masked_where((phi < 0+eps) | (phi > 1-eps), phi)
    ax.imshow(phi_perimeter, vmin=0.0, vmax=2.0, cmap="afmhot", extent=border)
    # ax.imshow(phi_perimeter, vmin=0.0, vmax=1.0, cmap="cividis", extent=border)

    # plot the solid at contact
    phi[g_dimensionless <= 0] = 2.0
    phi_solid = np.ma.masked_where(g_dimensionless > 0, phi)
    ax.imshow(phi_solid, vmin=0.0, vmax=2.0, cmap="afmhot", extent=border)


def demonstrate_dynamics(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.hstack([data.d for data in many_data])

    ax.plot(many_d/eta)


def plot_energy(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.stack([data.d for data in many_data])
    # TODO: better data_record?
    many_E = np.stack([getattr(data, "E") for data in many_data])

    # split the data into "pull" and "push"
    # TODO: cannot tell "orphan" data points
    pull = np.diff(many_d, prepend=many_d[0]) >= 0
    push = np.diff(many_d, prepend=many_d[0]) < 0

    all_d_pull = many_d[pull]
    all_E_pull = many_E[pull]
    all_d_push = many_d[push]
    all_E_push = many_E[push]

    # plotting
    ax.plot(all_d_pull/eta, all_E_pull/eta**2, "rx-", ms=8, mfc="none", label="retraction")
    ax.plot(all_d_push/eta, all_E_push/eta**2, "bo--", ms=8, mfc="none", label="approach")

    # format the plot
    ax.legend(loc="lower right")
    ax.grid()


def plot_PSD(ax: plt.Axes):
    # TODO: change to sample the PSD from the height profile of a rough surface
    L = 10           # spatial dimension
    n_grid = 200     # samples in spatial domain
    region = Region(L, L, n_grid, n_grid)

    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    C0 = 1e7  # prefactor
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)

    # isotropic PSD
    q_iso = region.qx
    ax.loglog(fft.fftshift(q_iso), fft.fftshift(roughness.mapto_psd(q_iso)))
    ax.axvline(abs(q_iso[q_iso.nonzero()]).min(), color="r", linestyle="--")
    ax.axvline(q_iso.max(), color="r", linestyle="--")

    ax.grid()


def plot_force(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.stack([data.d for data in many_data])
    # TODO: better data_record?
    many_F = np.stack([getattr(data, "F") for data in many_data])

    # split the data into "pull" and "push"
    # TODO: cannot tell "orphan" data points
    pull = np.diff(many_d, prepend=many_d[0]) >= 0
    push = np.diff(many_d, prepend=many_d[0]) < 0

    d_pull = np.array(many_d[pull])
    f_pull = np.array(many_F[pull])
    d_push = np.array(many_d[push])
    f_push = np.array(many_F[push])

    # compute force and plotting
    ax.plot(d_pull/eta, f_pull/eta, color="r", linestyle="-", marker="o", mfc="none", label="retraction")
    ax.plot(d_push/eta, f_push/eta, color="b", linestyle="--", marker="x", mfc="none", label="approach")

    # format the plot
    ax.legend()
    ax.grid()


def hide_border(ax: plt.Axes):
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)


def hide_ticks(ax: plt.Axes):
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)


# TO ORGANIZE: The following functions do not have an `ax` parameter, should be organized differently?
def latexify_plot(font_size: int):
    params = {
        'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
    }
    matplotlib.rcParams.update(params)


def animate_droplet_evolution_with_force_curve(many_data: list[DropletData]):
    eta = many_data[0].eta
    many_d = np.stack([data.d for data in many_data])
    d_ax_min = many_d.min() / eta - 0.5
    d_ax_max = many_d.max() / eta + 0.5

    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [0.6, 1]})

    def update_image(frame: int):
        for ax in axs:
            ax.clear()
        plot_force(axs[0], many_data[0:1+frame])
        axs[0].set_xlim(d_ax_min, d_ax_max)
        plot_phase_field_over_gap_topography(axs[1], many_data[frame])
        return [axs[0].lines, axs[1].images]

    return animation.FuncAnimation(fig, update_image, len(many_data), interval=500, repeat_delay=3000)
