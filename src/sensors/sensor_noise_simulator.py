import logging
import math
from dataclasses import dataclass

import numpy as np

from src.physics.ball_physics_model import TrajectoryResult

logger = logging.getLogger(__name__)

# MEMS IMU noise characteristics
_ACC_WHITE_NOISE_STD = 0.05    # m/s²
_ACC_BIAS_STD = 0.02           # m/s²  (per-shot constant offset)
_GYRO_WHITE_NOISE_STD = 0.01   # rad/s
_GYRO_BIAS_STD = 0.005         # rad/s (per-shot constant offset)

# Quantization parameters
_ACC_RANGE = 16 * 9.81         # ±16g in m/s²
_ACC_BITS = 12
_GYRO_RANGE = 2000 * math.pi / 180  # ±2000°/s in rad/s
_GYRO_BITS = 16


@dataclass
class IMUReading:
    t: np.ndarray                    # timestamps (s)
    acc_x: np.ndarray                # accelerometer x (m/s²) — noisy
    acc_y: np.ndarray                # accelerometer y (m/s²) — noisy
    gyro_z: np.ndarray               # gyroscope z (rad/s) — noisy spin reading
    true_trajectory: TrajectoryResult  # ground truth reference


class SensorNoiseSimulator:
    """Converts a clean physics trajectory into realistic noisy IMU readings.

    Applies three noise layers (in order):
    1. Per-shot constant bias (simulates sensor calibration offset)
    2. Gaussian white noise (thermal / electronic noise)
    3. Quantization noise (ADC resolution)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _quantize(self, signal: np.ndarray, full_range: float, bits: int) -> np.ndarray:
        """Simulate ADC quantization to `bits` bits over ±full_range."""
        n_levels = 2 ** bits
        lsb = 2 * full_range / n_levels
        return np.round(signal / lsb) * lsb

    def simulate_imu(
        self,
        trajectory: TrajectoryResult,
        noise_level: float = 1.0,
    ) -> IMUReading:
        """Generate noisy IMU readings from a clean trajectory.

        Parameters
        ----------
        trajectory   : Output of BallPhysicsModel.simulate()
        noise_level  : Multiplier for all noise magnitudes (0.0 = noiseless)

        Returns
        -------
        IMUReading with acc_x, acc_y, gyro_z arrays matching trajectory length
        """
        n = len(trajectory.t)

        # --- Accelerometer ---
        # Per-shot bias (constant over the shot)
        acc_bias_x = self.rng.normal(0.0, _ACC_BIAS_STD * noise_level)
        acc_bias_y = self.rng.normal(0.0, _ACC_BIAS_STD * noise_level)

        # White noise
        acc_wn_x = self.rng.normal(0.0, _ACC_WHITE_NOISE_STD * noise_level, n)
        acc_wn_y = self.rng.normal(0.0, _ACC_WHITE_NOISE_STD * noise_level, n)

        acc_x_noisy = trajectory.ax + acc_bias_x + acc_wn_x
        acc_y_noisy = trajectory.ay + acc_bias_y + acc_wn_y

        # Quantization
        acc_x_noisy = self._quantize(acc_x_noisy, _ACC_RANGE, _ACC_BITS)
        acc_y_noisy = self._quantize(acc_y_noisy, _ACC_RANGE, _ACC_BITS)

        # --- Gyroscope ---
        # True angular velocity = spin rate (constant throughout flight in this model)
        spin_rad_s = trajectory.spin_rpm * 2 * math.pi / 60.0
        gyro_true = np.full(n, spin_rad_s)

        # Per-shot bias
        gyro_bias = self.rng.normal(0.0, _GYRO_BIAS_STD * noise_level)

        # White noise
        gyro_wn = self.rng.normal(0.0, _GYRO_WHITE_NOISE_STD * noise_level, n)

        gyro_z_noisy = gyro_true + gyro_bias + gyro_wn
        gyro_z_noisy = self._quantize(gyro_z_noisy, _GYRO_RANGE, _GYRO_BITS)

        logger.debug(
            "simulate_imu: n=%d, noise_level=%.1f, spin=%.1f rad/s",
            n, noise_level, spin_rad_s,
        )

        return IMUReading(
            t=trajectory.t.copy(),
            acc_x=acc_x_noisy,
            acc_y=acc_y_noisy,
            gyro_z=gyro_z_noisy,
            true_trajectory=trajectory,
        )
