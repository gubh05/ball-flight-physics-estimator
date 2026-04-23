"""Benchmarker: compares PhysicsEstimator vs. NNEstimator on held-out test trajectories."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.physics_estimator import PhysicsEstimator
from src.models.nn_estimator import NNEstimator
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.physics.ball_physics_model import TrajectoryResult

logger = logging.getLogger(__name__)

_NOISE_LEVELS = [0.5, 1.0, 2.0]


class Benchmarker:
    """Head-to-head evaluation of PhysicsEstimator and NNEstimator."""

    def __init__(
        self,
        physics_estimator: PhysicsEstimator,
        nn_estimator: NNEstimator,
        noise_sim: SensorNoiseSimulator,
    ) -> None:
        self.physics_estimator = physics_estimator
        self.nn_estimator = nn_estimator
        self.noise_sim = noise_sim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
        """Compute MAE and RMSE for speed, angle, spin (columns 0, 1, 2)."""
        err = np.abs(true_vals - pred_vals)
        sq_err = (true_vals - pred_vals) ** 2
        return {
            "speed_mae":  float(err[:, 0].mean()),
            "angle_mae":  float(err[:, 1].mean()),
            "spin_mae":   float(err[:, 2].mean()),
            "speed_rmse": float(np.sqrt(sq_err[:, 0].mean())),
            "angle_rmse": float(np.sqrt(sq_err[:, 1].mean())),
            "spin_rmse":  float(np.sqrt(sq_err[:, 2].mean())),
        }

    def _imu_to_nn_input(self, imu, max_timesteps: int = 300) -> np.ndarray:
        """Convert a single IMUReading to (1, max_timesteps, 3) numpy array."""
        length = len(imu.t)
        arr = np.zeros((max_timesteps, 3), dtype=np.float32)
        n = min(length, max_timesteps)
        arr[:n, 0] = imu.acc_x[:n]
        arr[:n, 1] = imu.acc_y[:n]
        arr[:n, 2] = imu.gyro_z[:n]
        return arr[np.newaxis]  # (1, T, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test_trajectories: list) -> pd.DataFrame:
        """
        For each trajectory, simulate IMU at each noise level and run both estimators.

        Returns DataFrame with columns:
            estimator, noise_level,
            speed_mae, angle_mae, spin_mae,
            speed_rmse, angle_rmse, spin_rmse
        """
        records = []

        for noise_level in _NOISE_LEVELS:
            true_vals = []
            phys_preds = []
            nn_preds_list = []

            for traj in test_trajectories:
                imu = self.noise_sim.simulate_imu(traj, noise_level=noise_level)

                true_vals.append([
                    traj.launch_speed_mps,
                    traj.launch_angle_deg,
                    traj.spin_rpm,
                ])

                # PhysicsEstimator
                p_est = self.physics_estimator.estimate(imu)
                phys_preds.append([
                    p_est["launch_speed_mps"],
                    p_est["launch_angle_deg"],
                    p_est["spin_rpm"],
                ])

                # NNEstimator
                X = self._imu_to_nn_input(imu)
                nn_pred = self.nn_estimator.predict(X)  # (1, 3) denormalised
                nn_preds_list.append(nn_pred[0])

            true_arr  = np.array(true_vals,      dtype=np.float64)
            phys_arr  = np.array(phys_preds,     dtype=np.float64)
            nn_arr    = np.array(nn_preds_list,  dtype=np.float64)

            records.append({"estimator": "PhysicsEstimator", "noise_level": noise_level,
                            **self._metrics(true_arr, phys_arr)})
            records.append({"estimator": "NNEstimator",      "noise_level": noise_level,
                            **self._metrics(true_arr, nn_arr)})

        df = pd.DataFrame(records)
        logger.info(
            "Benchmarker: %d trajectories × %d noise levels evaluated",
            len(test_trajectories), len(_NOISE_LEVELS),
        )
        return df

    def plot_comparison(self, results: pd.DataFrame, output_dir: Path = None) -> None:
        """Grouped bar chart: Physics vs. NN per metric at each noise level."""
        from src.utils.visualization import plot_benchmark_comparison

        if output_dir is None:
            output_dir = Path("outputs/plots")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "benchmark_comparison.png"
        plot_benchmark_comparison(results, save_path=save_path)
        logger.info("Benchmark comparison chart saved → %s", save_path)
