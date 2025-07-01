import dataclasses as dc

import numpy as np
import numpy.fft as fft
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.modelling import SelfAffineRoughness
from a_package.computing import Grid
from a_package.simulating import ProcessedResult, post_process


def get_capillary_state(pr: ProcessedResult, index: int):
    return DropletData(
        pr.modelling.region,
        pr.modelling.eta,
        pr.modelling.h1,
        pr.modelling.h2,
        pr.evolution.r[index],
        pr.evolution.g[index],
        pr.evolution.phi[index],
    )


# TODO: 'plot...' shall return the 'Artist' as the behaviour of 'matplotlib'.
# TODO: change parameter lists into something like (data, ax, **extra_plot_args)

# Setup for colours
cmap_height = "plasma"
cmap_gap = "hot"
cmap_contact = "Greys"
cmap_phase_field = "Blues"

color_gas_phase = "w"
color_liquid_phase = "C0"
color_solid_phase = "C7"


@dc.dataclass
class DropletData:
    """Somewhat flat data with all necessary values for visualizing one capillary state."""
    region: Grid
    eta: float       # interfacial width
    h1: np.ndarray   # roughness of the 1 plate in 2D-array
    h2: np.ndarray   # roughness of the 2 plate in 2D-array
    r: np.ndarray    # displacement between the baselines of two plates
    g: np.ndarray    # gap
    phi: np.ndarray  # phase-field in 2D-array


@dc.dataclass
class Record:
    data: list[DropletData]  # think as if taking snapshots
    init_guess: np.ndarray


eps = 2e-1  # cut off value to decide one phase


def plot_cross_section_sketch(ax: plt.Axes, data: DropletData, idx_row: int, value_water=1-eps):
    # Get the data of the cross section at specified row index
    # FIXME: only shift in x-axis is considered.
    h1 = np.roll(data.h1[idx_row,:], int(data.r[1]/data.region.dx))
    h1 = (h1 + data.r[-1]) / data.eta
    h2 = data.h2[idx_row,:] / data.eta
    x = data.region.x / data.eta

    # add border values due to periodic boundary condition
    h1 = np.append(h1, h1[0])
    h2 = np.append(h2, h2[0])
    x = np.append(x, data.region.lx/data.eta)

    # plot the two plates
    ax.plot(x, h1, "k-")
    ax.plot(x, h2, "k-")

    # highlight the contacting part (if any)
    at_contact = np.asanyarray(h1 < h2).nonzero()[0]
    if np.size(at_contact):
        i_diff = np.diff(at_contact, prepend=at_contact[0] - 1)
        i_break = np.hstack((i_diff > 1).nonzero())
        for contact_part in np.split(at_contact, i_break):
            ax.fill_between(x[contact_part], h2[contact_part], h1[contact_part], color=color_solid_phase)

    # highlight the water phase (if any)
    phi = data.phi[idx_row,:]
    water_phase = np.asarray(phi >= value_water).nonzero()[0]
    if np.size(water_phase):
        i_diff = np.diff(water_phase, prepend=water_phase[0] - 1)
        i_break = np.hstack((i_diff > 1).nonzero())
        for water_drop in np.split(water_phase, i_break):
            ax.fill_between(x[water_drop], h2[water_drop], h1[water_drop], color=color_liquid_phase)

    # Because the hightlight might not necessarily exist, we have to manually create the legend
    [p_liquid] = ax.fill(np.nan, np.nan, color_liquid_phase, label="Liquid")
    [p_solid] = ax.fill(np.nan, np.nan, color_solid_phase, label="Solid")
    ax.legend(handles=[p_liquid, p_solid], loc='upper center', ncol=2)

    # No view margin along x-axis.
    ax.set_xlim(x[0], x[-1])


def plot_cross_section_phase_field(ax: plt.Axes, data: DropletData, idx_row: int):
    phi = data.phi[idx_row,:]
    x_dimensionless = data.region.x / data.eta
    ax.plot(x_dimensionless, phi, color='C0')


def plot_height_topography(ax: plt.Axes, data: DropletData):
    # make value dimension less
    h = data.h1 / data.eta

    border = np.array([0, data.region.lx, 0, data.region.ly]) / data.eta
    im = ax.imshow(h, cmap=cmap_height, extent=border)

    return im


def plot_gap_topography(ax: plt.Axes, data: DropletData):
    # nondimensionalize by 'eta'
    g = data.g / data.eta
    border = np.array([0, data.region.lx, 0, data.region.ly]) / data.eta
    
    # Set a negative 'vmin' so that the map still looks blue
    vmax = g.max()
    vmin = 0
    # im = ax.imshow(g, cmap='Blues', vmin=vmin, vmax=vmax, interpolation='nearest', extent=border)
    im = ax.imshow(g, cmap=cmap_gap, vmin=vmin, vmax=vmax, interpolation='nearest', extent=border)

    return im


def plot_contact_topography(ax: plt.Axes, data: DropletData):
    # mask the non-contact part
    contact = np.ma.masked_where(data.g > 0, data.g / data.eta)

    # nondimensionalize by 'eta'
    contact = contact / data.eta
    border = np.array([0, data.region.lx, 0, data.region.ly]) / data.eta

    im = ax.imshow(contact, cmap=cmap_contact, vmin=-1, vmax=1, alpha=0.4, interpolation='nearest', extent=border)
    return im


def plot_liquid_topography(ax: plt.Axes, data:DropletData):
    liquid = np.ma.masked_where(data.phi <= 1-eps, data.phi)
    # liquid = data.phi
    border = np.array([0, data.region.lx, 0, data.region.ly]) / data.eta
    # im = ax.imshow(liquid, cmap="Reds", vmin=0, vmax=2, alpha=0.5, extent=border)
    im = ax.imshow(liquid, cmap=cmap_phase_field, vmin=0, vmax=1.5, alpha=0.8, interpolation='nearest', extent=border)
    return im


def plot_phase_field_topography(ax: plt.Axes, data: DropletData):
    phi = data.phi
    # NOTE: the value 2.0 is for solid phase, while that of the water / vapor phase is 1.0 / 0.0 respectively.
    # these values are to match the color in the "afmhot" colormap.
    # value_contact = 2.0
    # phi[g <= 0] = value_contact
    border = np.array([0, data.region.lx, 0, data.region.ly]) / data.eta
    im = ax.imshow(phi, vmin=0, vmax=2, cmap='Blues', interpolation='nearest', extent=border)
    return im


def plot_interface_topography(ax: plt.Axes, data:DropletData):
    dphi = np.gradient(data.phi)
    edge = sum(dphi_i**2 for dphi_i in dphi) > 1e-6
    interface = np.where(edge, data.phi, 0)
    # NOTE: hard-coded pixel size 0.1
    border = np.array([0, data.region.lx, 0, data.region.ly]) / 1e-1
    im = ax.imshow(interface, cmap="binary", vmin=0.0, vmax=None, interpolation='nearest', extent=border)
    return im


def plot_combined_topography(ax: plt.Axes, data: DropletData):
    im1 = plot_gap_topography(ax, data)
    im2 = plot_contact_topography(ax, data)
    # im3 = plot_interface_topography(ax, data)
    im4 = plot_liquid_topography(ax, data)
    # return im1, im2, im4


def demonstrate_dynamics(ax: plt.Axes, many_data: list[DropletData]):
    # extract data
    eta = many_data[0].eta
    many_d = np.hstack([data.d for data in many_data])

    ax.plot(many_d/eta)


def plot_gibbs_free_energy(ax: plt.Axes, pr: ProcessedResult, n_step: int=None):

    if n_step is None:
        n_step = len(pr.evolution.t_exec)

    # Get the first few data points
    E = pr.evolution.E[:n_step]
    p = pr.evolution.p[:n_step]
    V = pr.evolution.V[:n_step]
    G = E - p * V

    # Non-dimensionalize
    eta = pr.modelling.eta
    G = G / (eta**2)  # NOTE: actually needs to be divided again by 'gamma', but 'gamma' is symbolic so far.

    # Plot the x, y, z component of forces
    steps = np.arange(n_step)
    ax.plot(steps, G, color="C1", linestyle="-", marker="x", ms=5, mfc="none", label=r"$G$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()


def plot_PSD(ax: plt.Axes):
    # TODO: change to sample the PSD from the height profile of a rough surface
    L = 10           # spatial dimension
    n_grid = 200     # samples in spatial domain
    region = Grid(L, L, n_grid, n_grid)

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


def plot_normal_force(ax: plt.Axes, pr: ProcessedResult, n_step: int=None):

    if n_step is None:
        n_step = len(pr.evolution.t_exec)

    # Get the first few data points
    f = pr.evolution.F[:n_step]

    # Non-dimensionalize
    eta = pr.modelling.eta
    f = f / eta  # NOTE: actually needs to be divided by 'eta gamma', but 'gamma' is symbolic so far.

    # Plot the x, y, z component of forces
    steps = np.arange(n_step)
    ax.plot(steps, f[:,2], color="b", linestyle="-", marker="o", ms=3, mfc="none", label=r"$F_z$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()


def plot_shear_force(ax: plt.Axes, pr: ProcessedResult, n_step: int=None):

    if n_step is None:
        n_step = len(pr.evolution.t_exec)

    # Get the first few data points
    f = pr.evolution.F[:n_step]

    # Non-dimensionalize
    eta = pr.modelling.eta
    f = f / eta  # NOTE: actually needs to be divided by 'eta gamma', but 'gamma' is symbolic so far.

    # Plot the x, y, z component of forces
    steps = np.arange(n_step)
    ax.plot(steps, f[:,0], color="r", linestyle="-", marker="o", ms=5, mfc="none", label=r"$F_x$")
    ax.plot(steps, f[:,1], color="g", linestyle="--", marker="^", ms=5, mfc="none", label=r"$F_y$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()


def plot_force_curves(ax: plt.Axes, pr: ProcessedResult, idx_stop: int=None):
    """
    idx_stop: the index to stop; in case of plotting the first few data points.
    """

    # Get the first 'idx_stop' data
    r = pr.evolution.r[:idx_stop]
    f = pr.evolution.F[:idx_stop]

    # Non-dimensionalize by using 'eta'
    eta = pr.modelling.eta
    r = la.norm(r, axis=-1)
    r = r / eta
    f = f / eta  # NOTE: actually needs to be divided by 'eta gamma', but 'gamma' is symbolic so far.

    # Plot the x, y, z component of forces
    [l1] = ax.plot(r, f[:,0], color="r", linestyle="-", marker="o", mfc="none", label=r"$f_x$")
    [l2] = ax.plot(r, f[:,1], color="g", linestyle="--", marker="^", mfc="none", label=r"$f_y$")
    [l3] = ax.plot(r, f[:,2], color="b", linestyle=":", marker="x", mfc="none", label=r"$f_z$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()

    return l1, l2, l3


def s(ax: plt.Axes, pr: ProcessedResult):
    z = pr.evolution.z1
    z_steps = np.diff(z, prepend=z[0])
    # TODO: cannot tell "orphan" data points
    is_pull = z_steps > 0
    is_push = z_steps < 0


def hide_border(ax: plt.Axes):
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)


def hide_ticks(ax: plt.Axes):
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)


# TO ORGANIZE: The following functions do not have an `ax` parameter, should be organized differently?
def latexify_plot(font_size: int):
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['helvetica'],
        # 'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
    }
    matplotlib.rcParams.update(params)
