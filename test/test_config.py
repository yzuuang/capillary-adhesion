"""
Tests for TOML configuration loading and sweep expansion.
"""

import os
import tempfile

import pytest
import numpy as np

from a_package.config import (
    load_config,
    save_config,
    expand_sweeps,
    count_sweep_combinations,
    get_surface_shape,
    Config,
)
from a_package.config.schema import (
    GeometryConfig,
    GridConfig,
    SurfaceConfig,
    PhysicsConfig,
    CapillaryConfig,
    SimulationConfig,
    TrajectoryConfig,
    SolverConfig,
    SweepConfig,
)
from a_package.physics.surfaces import generate_surface
from a_package.domain import Grid


@pytest.fixture
def sample_toml_content():
    return """
[geometry]
[geometry.grid]
pixel_size = 0.05
nb_pixels = 64

[geometry.upper]
shape = "tip"
radius = 10.0

[geometry.lower]
shape = "flat"
constant = 0.0

[physics]
[physics.capillary]
interface_thickness = 0.05
contact_angle_degree = 45.0
liquid_volume_percent = 15.0

[simulation]
[simulation.trajectory]
min_separation = 0.0
max_separation = 0.1
step_size = 0.01

[simulation.solver]
max_nb_iters = 1000
max_nb_loops = 30
tol_convergence = 1e-6
tol_constraints = 1e-8
init_penalty_weight = 0.1
"""


@pytest.fixture
def sample_toml_with_sweep():
    return """
[geometry]
[geometry.grid]
pixel_size = 0.05
nb_pixels = 64

[geometry.upper]
shape = "tip"
radius = 10.0

[geometry.lower]
shape = "flat"
constant = 0.0

[physics]
[physics.capillary]
interface_thickness = 0.05
contact_angle_degree = 45.0
liquid_volume_percent = 15.0

[simulation]
[simulation.trajectory]
min_separation = 0.0
max_separation = 0.1
step_size = 0.01

[simulation.solver]
max_nb_iters = 1000
max_nb_loops = 30
tol_convergence = 1e-6
tol_constraints = 1e-8
init_penalty_weight = 0.1

[[sweep]]
path = "physics.capillary.liquid_volume_percent"
linspace = [20.0, 80.0, 4]
"""


@pytest.fixture
def temp_toml_file(sample_toml_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(sample_toml_content)
        filepath = f.name
    yield filepath
    os.unlink(filepath)


def test_load_config(temp_toml_file):
    """Test loading a TOML config file."""
    config = load_config(temp_toml_file)

    assert isinstance(config, Config)
    assert config.geometry.grid.pixel_size == 0.05
    assert config.geometry.grid.nb_pixels == 64
    assert isinstance(config.geometry.upper, SurfaceConfig)
    assert config.geometry.upper.shape == "tip"
    assert config.geometry.upper.params["radius"] == 10.0
    assert isinstance(config.geometry.lower, SurfaceConfig)
    assert config.geometry.lower.shape == "flat"
    assert get_surface_shape(config.geometry.upper) == "tip"
    assert get_surface_shape(config.geometry.lower) == "flat"
    assert config.physics.capillary.contact_angle_degree == 45.0
    assert config.simulation.trajectory.step_size == 0.01
    assert config.simulation.solver.max_nb_iters == 1000


def test_save_and_reload_config(temp_toml_file):
    """Test saving and reloading a config."""
    config = load_config(temp_toml_file)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        output_path = f.name

    try:
        save_config(config, output_path)
        reloaded = load_config(output_path)

        assert reloaded.geometry.grid.pixel_size == config.geometry.grid.pixel_size
        assert get_surface_shape(reloaded.geometry.upper) == get_surface_shape(config.geometry.upper)
        assert reloaded.physics.capillary.liquid_volume_percent == config.physics.capillary.liquid_volume_percent
    finally:
        os.unlink(output_path)


def test_expand_sweeps_no_sweep(temp_toml_file):
    """Test expansion when no sweeps are defined."""
    config = load_config(temp_toml_file)

    expanded = list(expand_sweeps(config))
    assert len(expanded) == 1
    assert expanded[0].physics.capillary.liquid_volume_percent == 15.0


def test_expand_sweeps_with_linspace(sample_toml_with_sweep):
    """Test sweep expansion with linspace."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(sample_toml_with_sweep)
        filepath = f.name

    try:
        config = load_config(filepath)

        assert len(config.sweeps) == 1
        assert count_sweep_combinations(config) == 4

        expanded = list(expand_sweeps(config))
        assert len(expanded) == 4

        # Check the swept values
        volumes = [c.physics.capillary.liquid_volume_percent for c in expanded]
        np.testing.assert_array_almost_equal(volumes, [20.0, 40.0, 60.0, 80.0])

        # Verify sweeps are removed from expanded configs
        for c in expanded:
            assert len(c.sweeps) == 0
    finally:
        os.unlink(filepath)


def test_expand_sweeps_with_multiple_sweeps():
    """Test sweep expansion with multiple sweep parameters."""
    # Create a config programmatically
    config = Config(
        geometry=GeometryConfig(
            grid=GridConfig(pixel_size=0.05, nb_pixels=64),
            upper=SurfaceConfig(shape="tip", params={"radius": 10.0}),
            lower=SurfaceConfig(shape="flat", params={"constant": 0.0}),
        ),
        physics=PhysicsConfig(
            capillary=CapillaryConfig(
                interface_thickness=0.05,
                contact_angle_degree=45.0,
                liquid_volume_percent=15.0,
            )
        ),
        simulation=SimulationConfig(
            trajectory=TrajectoryConfig(min_separation=0.0, max_separation=0.1, step_size=0.01),
            solver=SolverConfig(
                max_nb_iters=1000, max_nb_loops=30,
                tol_convergence=1e-6, tol_constraints=1e-8, init_penalty_weight=0.1
            ),
        ),
        sweeps=[
            SweepConfig(path="physics.capillary.liquid_volume_percent", linspace=[20.0, 40.0, 3]),
            SweepConfig(path="physics.capillary.contact_angle_degree", values=[30.0, 60.0]),
        ],
    )

    assert count_sweep_combinations(config) == 6  # 3 * 2

    expanded = list(expand_sweeps(config))
    assert len(expanded) == 6


def test_generate_surface_flat():
    """Test flat surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "flat", constant=0.5)

    assert height.shape == (32, 32)
    np.testing.assert_array_almost_equal(height, 0.5 * np.ones((32, 32)))


def test_generate_surface_tip():
    """Test tip surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "tip", radius=10.0)

    assert height.shape == (32, 32)
    # Minimum should be at center, value should be 0
    center_idx = 16
    assert height[center_idx, center_idx] == np.min(height)


def test_generate_surface_sinusoid():
    """Test sinusoidal surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "sinusoid", wavenumber=2.0, amplitude=0.1)

    assert height.shape == (32, 32)
    assert np.max(height) <= 0.1
    assert np.min(height) >= -0.1


def test_generate_surface_unknown_type():
    """Test that unknown surface type raises error."""
    grid = Grid([1.0, 1.0], [32, 32])

    with pytest.raises(ValueError, match="Unknown surface shape"):
        generate_surface(grid, "unknown_shape")
