import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from a_package.data_record import DropletData
from a_package.roughness import generate_isotropic_psd
from a_package.droplet import QuadratureRoughDroplet


eps = 1e-1  # cut off value to decide one phase


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

    # plot the solid at contact
    phi[g_dimensionless <= 0] = 2.0
    phi_solid = np.ma.masked_where(g_dimensionless > 0, phi)
    ax.imshow(phi_solid, vmin=0.0, vmax=2.0, cmap="afmhot", extent=border)


def demonstrate_dynamics(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.hstack([data.d for data in many_data])

    ax.plot(many_d/eta)


# TODO: have a separate plot energy and plot force
# IDEA: should Energy be considered as "post-processing"

def plot_energy(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.stack([data.d for data in many_data])

    # compute energy from the given data
    many_E = np.empty(len(many_data))
    for i, data in enumerate(many_data):
        droplet = QuadratureRoughDroplet(
            data.phi, data.h1, data.h2, data.d, data.eta, data.M, data.N, data.dx, data.dy
        )
        droplet.update_separation(data.d)
        droplet.update_phase_field(data.phi.ravel())
        E = droplet.compute_energy()
        many_E[i] = E

    # split departing and approaching by the index at half size
    key_index = len(many_d) // 2
    all_d_pull = many_d[:key_index]
    all_d_push = many_d[key_index:]
    all_E_pull = many_E[:key_index]
    all_E_push = many_E[key_index:]

    # plotting
    ax.plot(all_d_pull/eta, all_E_pull/eta, "rx-", ms=8, mfc="none", label="retraction")
    ax.plot(all_d_push/eta, all_E_push/eta, "bo--", ms=8, mfc="none", label="approach")

    # format the plot
    ax.legend(loc="upper right")
    ax.grid()


def plot_force(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.stack([data.d for data in many_data])

    # compute energy from the given data
    many_E = np.empty(len(many_data))
    for i, data in enumerate(many_data):
        droplet = QuadratureRoughDroplet(
            data.phi, data.h1, data.h2, data.d, data.eta, data.M, data.N, data.dx, data.dy
        )
        droplet.update_separation(data.d)
        droplet.update_phase_field(data.phi.ravel())
        E = droplet.compute_energy()
        many_E[i] = E

    # split departing and approaching by the index at half size
    key_index = len(many_d) // 2

    # compute force with finite difference method
    all_F_pull = -np.diff(many_E[:key_index]) / np.diff(many_d[:key_index])
    all_F_push = -np.diff(many_E[key_index:]) / np.diff(many_d[key_index:])

    # as a result of FD, the mean distance should be middle values
    all_mid_d_pull = (many_d[:key_index - 1] + many_d[1:key_index]) / 2
    all_mid_d_push = (many_d[key_index:-1] + many_d[key_index + 1:]) / 2

    # plotting
    ax.plot(all_mid_d_pull/eta, all_F_pull/eta, "rx-", ms=8, mfc="none", label="retraction")
    ax.plot(all_mid_d_push/eta, all_F_push/eta, "bo--", ms=8, mfc="none", label="approach")

    # format the plot
    ax.legend(loc="upper right")
    ax.grid()


def plot_PSD(ax: plt.Axes):
    # TODO: change to sample the PSD from the height profile of a rough surface
    L = 10  # spatial dimension
    qL = 2*np.pi / L
    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    C0 = 1e7  # prefactor
    H = 0.95  # Hurst exponent
    n_spectrum = 100  # samples in spectral domain
    q_iso, C_iso = generate_isotropic_psd(C0, qL, qR, qS, H, n_spectrum)
    ax.loglog(q_iso, np.absolute(C_iso))

    n_grid = 200
    dx = L / n_grid
    qx = 2*np.pi * np.fft.fftfreq(n_grid, dx)
    ax.axvline(abs(qx[qx.nonzero()]).min(), color="r", linestyle="--")
    ax.axvline(qx.max(), color="r", linestyle="--")

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

def overview_record(many_data: list[DropletData]):
    # Each result is showed as 1 topography and 1 cross-section at top and bottom respectively.
    # There are maximal `num_result_per_column` x `num_column` shown in one figure.
    num_result_per_column = 3
    num_row = num_result_per_column * 2
    num_column = 10
    max_result_per_figure = num_result_per_column * num_column

    # compute the number of figures required
    num_result = len(many_data)
    num_fig = num_result // max_result_per_figure
    if num_result % max_result_per_figure > 0:
        num_fig += 1

    # create and pack figs, axs
    figs = []
    axs_topo = []
    axs_cross = []
    for _ in range(num_fig):
        fig, axs = plt.subplots(num_row, num_column, figsize=(1.6 * num_column, 1.6 * num_row),
                                gridspec_kw={'height_ratios': np.tile([1, 0.5], num_result_per_column)},
                                sharex="col", sharey="row", constrained_layout=True)
        figs.append(fig)
        axs_topo.extend(axs[0::2].ravel())
        axs_cross.extend(axs[1::2].ravel())

    # plotting
    N = many_data[0].N
    idx_row = N // 2
    for j, data in enumerate(many_data):
        # plot_phase_field_topography(axs_topo[j], data)
        plot_phase_field_over_gap_topography(axs_topo[j], data)
        axs_topo[j].axhline(data.y[idx_row] / data.eta, color="C0")
        plot_cross_section_sketch(axs_cross[j], data, idx_row)

    # add figure super-title
    data = many_data[0]
    for idx_fig, fig in enumerate(figs):
        fig.suptitle(fr"Figure ({idx_fig+1}): Resolution {data.M}x{data.N}, $L={data.L/data.eta}\eta, V={data.V/data.eta**3}\eta^3$")
