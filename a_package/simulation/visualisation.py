import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

from a_package.simulation.io import SimulationIO, Term
from a_package.physics.models import SelfAffineRoughness
from a_package.grid import Grid


# Setup for colours
cmap_height = "plasma"
cmap_gap = "hot"
cmap_contact = "Greys"
cmap_phase_field = "Blues"

color_gas_phase = "w"
color_solid_phase = "C7"
# color_liquid_phase = "steelblue"
# color_transition_phase = "lightskyblue"
# color_vapour_phase = "aliceblue"
color_liquid_phase = "steelblue"
color_transition_phase = "lightblue"


eps = 1e-2  # cut off value to decide one phase


def plot_cross_section_sketch(ax: plt.Axes, io: SimulationIO, idx_step: int, idx_row: int, value_cutoff=eps):
    # Get the data of the cross section at specified row index
    # FIXME: only shift in x-axis is considered.
    grid = io.grid
    unit = min(grid.element_sizes)
    data = io.load_step(
        idx_step, field_names=[Term.upper_solid, Term.lower_solid, Term.phase],
        single_value_names=[Term.separation])

    h1 = data[Term.upper_solid][0,0,idx_row]
    sep = data[Term.separation]
    h1 = (h1 + sep) / unit
    h2 = data[Term.lower_solid][0,0,idx_row] / unit
    x = grid.form_nodal_axis(0) / unit

    # add border values due to periodic boundary condition
    h1 = np.append(h1, h1[0])
    h2 = np.append(h2, h2[0])
    x = np.append(x, grid.lengths[0]/unit)

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

    # highlight the liquid phase (if any)
    phi = data[Term.phase][0, 0, idx_row, :]
    water_phase = np.asarray(phi >= 1 - value_cutoff).nonzero()[0]
    if np.size(water_phase):
        i_diff = np.diff(water_phase, prepend=water_phase[0] - 1)
        i_break = np.hstack((i_diff > 1).nonzero())
        for section in np.split(water_phase, i_break):
            ax.fill_between(x[section], h2[section], h1[section], color=color_liquid_phase)

    # # highlight the vapour phase (if any)
    # phi = data[Term.phase][0, 0, idx_row, :]
    # vapour_phase = np.asarray(phi <= 0 + value_cutoff).nonzero()[0]
    # if np.size(vapour_phase):
    #     i_diff = np.diff(vapour_phase, prepend=vapour_phase[0] - 1)
    #     i_break = np.hstack((i_diff > 1).nonzero())
    #     for section in np.split(vapour_phase, i_break):
    #         ax.fill_between(x[section], h2[section], h1[section], color=color_vapour_phase)

    # highlight the transition phase (if any)
    phi = data[Term.phase][0, 0, idx_row, :]
    transition_phase = np.asarray((phi <= 1 - value_cutoff) & (phi >= 0 + value_cutoff)).nonzero()[0]
    if np.size(transition_phase):
        i_diff = np.diff(transition_phase, prepend=transition_phase[0] - 1)
        i_break = np.hstack((i_diff > 1).nonzero())
        for section in np.split(transition_phase, i_break):
            ax.fill_between(x[section], h2[section], h1[section], color=color_transition_phase)

    # Because the hightlight might not necessarily exist, we have to manually create the legend
    [p_solid] = ax.fill(np.nan, np.nan, color_solid_phase, label="Solid")
    [p_liquid] = ax.fill(np.nan, np.nan, color_liquid_phase, label="Liquid")
    # [p_vapour] = ax.fill(np.nan, np.nan, color_vapour_phase, label="Vapour")
    [p_interface] = ax.fill(np.nan, np.nan, color_transition_phase, label="Interface")
    # ax.legend(handles=[p_vapour, p_interface, p_liquid, p_solid], loc="upper center", ncol=2)
    ax.legend(handles=[p_interface, p_liquid, p_solid], loc="upper center", ncol=3)

    # No view margin along x-axis.
    ax.set_xlim(x[0], x[-1])


def plot_cross_section_phase_field(ax: plt.Axes, io: SimulationIO, idx_step: int, idx_row: int):
    data = io.load_step(idx_step, field_names=[Term.phase])
    phi = data[Term.phase][0, 0, idx_row,:]
    x_dimensionless = io.grid.form_nodal_axis(0) / min(io.grid.element_sizes)
    ax.plot(x_dimensionless, phi, color='C0')


def plot_height_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    data = io.load_step(idx_step, field_names=[Term.upper_solid])
    unit = min(io.grid.element_sizes)
    # make value dimension less
    h = data[Term.upper_solid].squeeze() / unit

    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / unit
    im = ax.imshow(h, interpolation='none', cmap=cmap_height, extent=tuple(border))

    return im


def plot_gap_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    # nondimensionalize by 'eta'
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = min(io.grid.element_sizes)
    g = data[Term.gap].squeeze() / unit
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / unit

    # Set a negative 'vmin' so that the map still looks blue
    vmax = g.max()
    vmin = 0
    # im = ax.imshow(g, cmap='Blues', vmin=vmin, vmax=vmax, interpolation='nearest', extent=border)
    im = ax.imshow(g, interpolation='none', cmap=cmap_gap, vmin=vmin, vmax=vmax, extent=border)

    return im


def plot_contact_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    # nondimensionalize by 'eta'
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = min(io.grid.element_sizes)
    g = data[Term.gap].squeeze() / unit

    # mask the non-contact part
    contact = np.ma.masked_where(g > 0, g)

    # nondimensionalize by 'eta'
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / unit
    im = ax.imshow(contact, cmap=cmap_contact, vmin=-1, vmax=1, alpha=0.4, interpolation='nearest', extent=border)
    return im


def plot_droplet_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    data = io.load_step(idx_step, field_names=[Term.phase])
    unit = min(io.grid.element_sizes)
    # only use a part of the colour map, as the bluest blue is too dark
    vmin = 0
    vmax = 1.5
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / unit

    phi = data[Term.phase].squeeze()
    liquid = np.ma.masked_where(phi <= 1 - eps, phi)
    im = ax.imshow(
        liquid, cmap=cmap_phase_field, vmin=vmin, vmax=vmax, alpha=0.85, interpolation="nearest", extent=border
    )

    transition = np.ma.masked_where((phi <= 0 + eps) | (phi > 1 - eps), phi)
    im = ax.imshow(
        transition, cmap=cmap_phase_field, vmin=vmin, vmax=vmax, alpha=0.7, interpolation="nearest", extent=border
    )

    return im


def plot_phase_field_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    data = io.load_step(idx_step, field_names=[Term.phase]) 
    unit = min(io.grid.element_sizes)
    # NOTE: the value 2.0 is for solid phase, while that of the water / vapor phase is 1.0 / 0.0 respectively.
    # these values are to match the color in the "afmhot" colormap.
    # value_contact = 2.0
    # phi[g <= 0] = value_contact
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / unit
    im = ax.imshow(data[Term.phase].squeeze(), vmin=0, vmax=2, cmap='Blues', interpolation='nearest', extent=border)
    return im


def plot_interface_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    data = io.load_step(idx_step, field_names=[Term.phase]) 
    dphi = np.gradient(data[Term.phase].squeeze())
    edge = sum(dphi_i**2 for dphi_i in dphi) > 1e-6
    interface = np.where(edge, data[Term.phase].squeeze(), 0)
    # NOTE: hard-coded pixel size 0.1
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / 1e-1
    im = ax.imshow(interface, cmap="binary", vmin=0.0, vmax=None, interpolation='nearest', extent=border)
    return im


def plot_combined_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    im1 = plot_gap_topography(ax, io, idx_step)
    im2 = plot_contact_topography(ax, io, idx_step)
    # im3 = plot_interface_topography(ax, io, idx_step)
    im4 = plot_droplet_topography(ax, io, idx_step)
    # im4 = plot_phase_field_topography(ax, io, idx_step)
    # return im1, im2, im4


def demonstrate_dynamics(ax: plt.Axes, io: SimulationIO):
    # extract data
    data = io.load_trajectory(single_value_names=[Term.separation])
    unit = min(io.grid.element_sizes)
    ax.plot(data[Term.separation] / unit)


def plot_gibbs_free_energy(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):

    data = io.load_trajectory(single_value_names=[Term.energy])
    energy = data[Term.energy][:nb_steps]

    # Non-dimensionalize
    unit = min(io.grid.element_sizes)
    # NOTE: actually needs to be divided again by 'gamma'(surface tension), but 'gamma' is symbolic so far.
    energy = energy / (unit**2)  

    # Plot the x, y, z component of forces
    steps = np.arange(nb_steps)
    ax.plot(steps, energy, color="C1", linestyle="-", marker="x", ms=5, mfc="none", label=r"$G$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()


def plot_PSD(ax: plt.Axes):
    # TODO: change to sample the PSD from the height profile of a rough surface
    L = 10           # spatial dimension
    n_grid = 200     # samples in spatial domain
    grid = Grid([L, L], [n_grid, n_grid])

    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    C0 = 1e7  # prefactor
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)

    # isotropic PSD
    q_iso = grid.form_spectral_axis(0)
    ax.loglog(fft.fftshift(q_iso), fft.fftshift(roughness.mapto_isotropic_psd(q_iso)))
    ax.axvline(abs(q_iso[q_iso.nonzero()]).min(), color="r", linestyle="--")
    ax.axvline(q_iso.max(), color="r", linestyle="--")

    ax.grid()


def plot_normal_force(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    data = io.load_trajectory(single_value_names=[Term.energy, Term.separation])
    energy = data[Term.energy][:nb_steps]
    displ_z = data[Term.separation][:nb_steps]
    # numerical difference to get force
    force = -(energy[1:] - energy[:-1]) / (displ_z[1:] - displ_z[:-1])

    # Non-dimensionalize
    unit = min(io.grid.element_sizes)
    force = force / unit  # NOTE: actually needs to be divided by 'eta gamma', but 'gamma' is symbolic so far.

    # Plot the x, y, z component of forces
    steps = np.arange(nb_steps)
    steps = (steps[1:] + steps[:-1]) / 2
    ax.plot(steps, force, color="b", linestyle="-", marker="o", ms=3, mfc="none", label=r"$F_z$")

    # Format the plot
    ax.legend(loc='upper right')
    ax.grid()


def plot_pressure(ax: plt.Axes, io: SimulationIO, nb_steps: int=None):

    data = io.load_trajectory(single_value_names=[Term.pressure])
    pressure = data[Term.pressure][:nb_steps]

    # Non-dimensionalize
    unit = min(io.grid.element_sizes)
    pressure = pressure * unit

    # Plot the x, y, z component of forces
    steps = np.arange(nb_steps)
    ax.plot(steps, pressure, color="r", linestyle="-", marker="o", ms=5, mfc="none", label=r"$P/\gamma a^{-1}$")

    # Format the plot
    ax.legend(loc='upper right')
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
