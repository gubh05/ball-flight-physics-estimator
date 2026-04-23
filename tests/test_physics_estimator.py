"""Tests for PhysicsEstimator — pure analytical inverse estimation."""
import numpy as np
import pytest

from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.models.physics_estimator import PhysicsEstimator


@pytest.fixture
def model():
    return BallPhysicsModel()


@pytest.fixture
def sim():
    return SensorNoiseSimulator(seed=0)


@pytest.fixture
def estimator():
    return PhysicsEstimator()


def test_estimate_returns_required_keys(model, sim, estimator):
    traj = model.simulate(20.0, 30.0, spin_rpm=500.0)
    imu = sim.simulate_imu(traj, noise_level=0.0)
    est = estimator.estimate(imu)
    assert "launch_speed_mps" in est
    assert "launch_angle_deg" in est
    assert "spin_rpm" in est


def test_speed_estimate_within_20pct_at_zero_noise(model, sim, estimator):
    """With no noise the speed estimate must be within 20% of true value."""
    true_speed = 20.0
    traj = model.simulate(true_speed, 30.0, spin_rpm=500.0)
    imu = sim.simulate_imu(traj, noise_level=0.0)
    est = estimator.estimate(imu)
    err = abs(est["launch_speed_mps"] - true_speed) / true_speed
    assert err < 0.20, f"Speed error {err:.1%} exceeds 20%"


def test_spin_sign_preserved(model, sim, estimator):
    """Positive spin_rpm input should yield positive spin estimate."""
    traj = model.simulate(20.0, 30.0, spin_rpm=1000.0)
    imu = sim.simulate_imu(traj, noise_level=0.0)
    est = estimator.estimate(imu)
    assert est["spin_rpm"] > 0, "Positive spin not preserved"


def test_zero_spin_estimate_near_zero(model, sim, estimator):
    """Zero spin should give near-zero gyro estimate."""
    traj = model.simulate(20.0, 30.0, spin_rpm=0.0)
    imu = sim.simulate_imu(traj, noise_level=0.0)
    est = estimator.estimate(imu)
    assert abs(est["spin_rpm"]) < 200, f"Spin estimate {est['spin_rpm']} too far from 0"


def test_output_values_are_finite(model, sim, estimator):
    """Estimates must be finite numbers, not NaN or inf."""
    traj = model.simulate(15.0, 25.0, spin_rpm=-500.0)
    imu = sim.simulate_imu(traj, noise_level=1.0)
    est = estimator.estimate(imu)
    for key, val in est.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"
