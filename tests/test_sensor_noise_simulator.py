"""Tests for SensorNoiseSimulator — shapes, noise properties, and gyro calibration."""
import math

import numpy as np
import pytest

from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import IMUReading, SensorNoiseSimulator


@pytest.fixture(scope="module")
def base_traj():
    model = BallPhysicsModel()
    return model.simulate(20.0, 30.0, spin_rpm=500.0)


@pytest.fixture(scope="module")
def sim():
    return SensorNoiseSimulator(seed=42)


def test_output_shape_matches_trajectory(sim, base_traj):
    imu = sim.simulate_imu(base_traj, noise_level=1.0)
    n = len(base_traj.t)
    assert imu.acc_x.shape == (n,)
    assert imu.acc_y.shape == (n,)
    assert imu.gyro_z.shape == (n,)
    assert imu.t.shape == (n,)


def test_imu_reading_is_dataclass(sim, base_traj):
    imu = sim.simulate_imu(base_traj)
    assert isinstance(imu, IMUReading)


def test_noise_changes_signal(sim, base_traj):
    imu = sim.simulate_imu(base_traj, noise_level=1.0)
    assert not np.allclose(imu.acc_x, base_traj.ax), "Noise should change acc_x"
    assert not np.allclose(imu.acc_y, base_traj.ay), "Noise should change acc_y"


def test_noise_level_zero_gives_small_noise(sim, base_traj):
    """With noise_level=0 only quantization remains — max error should be small."""
    imu0 = sim.simulate_imu(base_traj, noise_level=0.0)
    max_dev_acc = np.max(np.abs(imu0.acc_x - base_traj.ax))
    # 12-bit over ±16g: LSB ≈ 0.077 m/s², half-LSB ≈ 0.039 m/s²
    assert max_dev_acc < 0.1, f"Noiseless acc deviation too large: {max_dev_acc:.4f}"


def test_gyro_mean_near_spin_rate(base_traj):
    """Gyroscope mean should be close to the true spin rate in rad/s."""
    sim = SensorNoiseSimulator(seed=0)
    true_rad_s = base_traj.spin_rpm * 2 * math.pi / 60.0
    # Average over many seeds to reduce bias
    means = []
    for seed in range(20):
        s = SensorNoiseSimulator(seed=seed)
        imu = s.simulate_imu(base_traj, noise_level=1.0)
        means.append(imu.gyro_z.mean())
    avg_mean = np.mean(means)
    assert abs(avg_mean - true_rad_s) < 0.5, (
        f"Gyro mean {avg_mean:.3f} too far from true {true_rad_s:.3f} rad/s"
    )


def test_true_trajectory_stored(sim, base_traj):
    imu = sim.simulate_imu(base_traj)
    assert imu.true_trajectory is base_traj


def test_no_nans(sim, base_traj):
    imu = sim.simulate_imu(base_traj, noise_level=2.0)
    assert not np.isnan(imu.acc_x).any()
    assert not np.isnan(imu.acc_y).any()
    assert not np.isnan(imu.gyro_z).any()


def test_higher_noise_level_larger_deviation(base_traj):
    """Higher noise_level should produce larger deviations from true signal."""
    s1 = SensorNoiseSimulator(seed=7)
    s2 = SensorNoiseSimulator(seed=7)
    imu_low  = s1.simulate_imu(base_traj, noise_level=0.1)
    imu_high = s2.simulate_imu(base_traj, noise_level=3.0)
    dev_low  = np.std(imu_low.acc_x  - base_traj.ax)
    dev_high = np.std(imu_high.acc_x - base_traj.ax)
    assert dev_high > dev_low, "Higher noise_level should produce larger std deviation"
