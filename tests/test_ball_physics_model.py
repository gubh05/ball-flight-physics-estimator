"""Tests for BallPhysicsModel — trajectory physics, spin effects, and output validity."""
import numpy as np
import pytest

from src.physics.ball_physics_model import BallParams, BallPhysicsModel


@pytest.fixture
def model():
    return BallPhysicsModel()


@pytest.fixture
def base_traj(model):
    return model.simulate(launch_speed_mps=20.0, launch_angle_deg=30.0, spin_rpm=0.0)


def test_trajectory_starts_at_origin(base_traj):
    assert base_traj.x[0] == 0.0
    assert base_traj.y[0] == 0.0


def test_ball_lands(base_traj):
    assert base_traj.y[-1] < 0.1, f"Ball should land near y=0, got y={base_traj.y[-1]:.3f}"


def test_ball_moves_forward(base_traj):
    assert base_traj.x[-1] > 0.0


def test_arrays_same_length(base_traj):
    lengths = [len(base_traj.t), len(base_traj.x), len(base_traj.y),
               len(base_traj.vx), len(base_traj.vy), len(base_traj.ax), len(base_traj.ay)]
    assert len(set(lengths)) == 1, f"Array lengths differ: {lengths}"


def test_vx_decreases_due_to_drag(model):
    """Drag should decelerate horizontal velocity."""
    traj = model.simulate(20.0, 30.0, spin_rpm=0.0)
    # vx should generally decrease over the first half of flight
    half = len(traj.vx) // 2
    assert traj.vx[half] < traj.vx[0], "Drag should reduce vx over time"


def test_no_spin_gives_no_magnus(model):
    """Zero spin → symmetric trajectory (no lateral Magnus deflection)."""
    traj_zero = model.simulate(20.0, 45.0, spin_rpm=0.0)
    traj_back = model.simulate(20.0, 45.0, spin_rpm=2000.0)
    # With backspin the ball should travel further
    assert model.range_m(traj_back) > model.range_m(traj_zero)


def test_topspin_reduces_range(model):
    """Topspin (negative RPM) forces ball down → shorter range than no-spin."""
    traj_none = model.simulate(20.0, 30.0, spin_rpm=0.0)
    traj_top  = model.simulate(20.0, 30.0, spin_rpm=-2000.0)
    assert model.range_m(traj_top) < model.range_m(traj_none)


def test_higher_speed_longer_range(model):
    """Higher launch speed should produce longer horizontal range."""
    traj_slow = model.simulate(10.0, 30.0, spin_rpm=0.0)
    traj_fast = model.simulate(25.0, 30.0, spin_rpm=0.0)
    assert model.range_m(traj_fast) > model.range_m(traj_slow)


def test_time_of_flight_positive(model, base_traj):
    assert model.time_of_flight(base_traj) > 0.0


def test_max_height_positive(model, base_traj):
    assert model.max_height(base_traj) > 0.0


def test_no_nan_in_trajectory(base_traj):
    for arr in [base_traj.x, base_traj.y, base_traj.vx, base_traj.vy,
                base_traj.ax, base_traj.ay]:
        assert not np.isnan(arr).any(), "NaN found in trajectory arrays"


def test_custom_params():
    """BallPhysicsModel accepts custom BallParams."""
    params = BallParams(mass_kg=0.5, drag_coeff=0.3)
    model_custom = BallPhysicsModel(params=params)
    traj = model_custom.simulate(20.0, 30.0, 0.0)
    assert len(traj.t) > 0


@pytest.mark.parametrize("speed,angle", [
    (5.0, 5.0), (35.0, 60.0), (15.0, 45.0)
])
def test_param_boundary_conditions(model, speed, angle):
    """Simulation should run without error at parameter boundaries."""
    traj = model.simulate(speed, angle, spin_rpm=0.0)
    assert len(traj.t) > 0
    assert traj.x[-1] >= 0.0
