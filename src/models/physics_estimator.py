import logging
import math

import numpy as np

from src.physics.ball_physics_model import BallParams
from src.sensors.sensor_noise_simulator import IMUReading

logger = logging.getLogger(__name__)


class PhysicsEstimator:
    """Analytical inverse estimator — recovers shot parameters from IMU readings.

    No ML, no training. Works purely by interpreting the physics of the signal:

    - launch_speed  : estimated from peak total acceleration magnitude near launch.
      At t≈0 the aerodynamic forces dominate; combining with drag/Magnus physics
      we back-calculate speed from the peak resultant force.
    - launch_angle  : estimated from the ratio of initial ax to ay (arctangent).
      At launch vx = v0·cos(θ), vy = v0·sin(θ), so vy/vx = tan(θ).
    - spin_rpm      : estimated from the mean gyroscope reading (gyro ≈ spin rate).

    This estimator is intentionally simple to highlight where ML outperforms it.
    """

    def __init__(self, params: BallParams = None):
        self.params = params or BallParams()

    def estimate(self, imu: IMUReading) -> dict:
        """Estimate shot parameters from noisy IMU data.

        Parameters
        ----------
        imu : IMUReading from SensorNoiseSimulator

        Returns
        -------
        dict with keys:
            'launch_speed_mps' : float
            'launch_angle_deg' : float
            'spin_rpm'         : float
        """
        p = self.params
        area = math.pi * p.radius_m ** 2
        drag_const = 0.5 * p.air_density * area * p.drag_coeff
        lift_const = 0.5 * p.air_density * area * p.lift_coeff

        # --- Window around launch (first ~5% of signal or at least 3 samples) ---
        n_window = max(3, len(imu.t) // 20)
        ax_init = imu.acc_x[:n_window]
        ay_init = imu.acc_y[:n_window]

        ax_mean = float(np.mean(ax_init))
        ay_mean = float(np.mean(ay_init))

        # --- Launch angle from initial acceleration direction ---
        # At t≈0: ax ∝ drag·vx + magnus·(-vy), ay ∝ drag·vy + magnus·vx
        # For a first-order estimate, the velocity direction mirrors the accel direction.
        # Use atan2 on the initial velocity inferred from the sign of accelerations.
        # Since drag opposes motion: ax_drag = -k·vx, ay_drag = -k·vy
        # => vx ∝ -ax, vy ∝ -ay  (dominated by drag)
        if abs(ax_mean) < 1e-6 and abs(ay_mean) < 1e-6:
            angle_rad = math.radians(30.0)  # fallback
        else:
            angle_rad = math.atan2(-ay_mean, -ax_mean)
            # Clamp to valid launch range [5°, 60°]
            angle_rad = max(math.radians(5.0), min(math.radians(60.0), abs(angle_rad)))

        launch_angle_deg = math.degrees(angle_rad)

        # --- Launch speed from peak acceleration magnitude ---
        # Total aero acceleration at launch: a_aero = (F_drag + F_magnus) / m
        # |a_aero| ≈ (drag_const·v² + lift_const·v²) / m  (for simplicity)
        # => v ≈ sqrt(|a_aero| * m / (drag_const + lift_const))
        accel_mag = float(np.sqrt(ax_mean ** 2 + ay_mean ** 2))
        combined_const = drag_const + lift_const
        if combined_const > 0 and accel_mag > 0:
            v_sq = accel_mag * p.mass_kg / combined_const
            launch_speed_mps = float(np.sqrt(max(v_sq, 0.0)))
        else:
            launch_speed_mps = 10.0  # fallback

        # Clamp to valid range
        launch_speed_mps = float(np.clip(launch_speed_mps, 5.0, 35.0))

        # --- Spin rate from gyroscope mean ---
        # gyro_z ≈ spin_rad_s + bias + noise  =>  mean ≈ spin_rad_s
        mean_gyro_rad_s = float(np.mean(imu.gyro_z))
        spin_rpm = mean_gyro_rad_s * 60.0 / (2 * math.pi)

        # Clamp to valid range
        spin_rpm = float(np.clip(spin_rpm, -3000.0, 3000.0))

        logger.debug(
            "PhysicsEstimator: speed=%.2f angle=%.2f spin=%.1f",
            launch_speed_mps, launch_angle_deg, spin_rpm,
        )

        return {
            "launch_speed_mps": launch_speed_mps,
            "launch_angle_deg": launch_angle_deg,
            "spin_rpm": spin_rpm,
        }
