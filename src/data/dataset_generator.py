import logging

import numpy as np
from tqdm import tqdm

from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.utils.config import Config

logger = logging.getLogger(__name__)

_MAX_TIMESTEPS = 300   # 3 seconds at 100 Hz


class DatasetGenerator:
    """Generates synthetic (IMU readings → shot parameters) pairs for ML training.

    For each sample:
    1. Sample random shot parameters (speed, angle, spin)
    2. Simulate trajectory via BallPhysicsModel
    3. Add IMU noise via SensorNoiseSimulator (at a randomly chosen noise level)
    4. Pad/truncate IMU channels to max_timesteps
    5. Normalise shot parameters to [0, 1]
    """

    def __init__(
        self,
        physics_model: BallPhysicsModel,
        noise_sim: SensorNoiseSimulator,
        config: Config = None,
    ):
        self.physics_model = physics_model
        self.noise_sim = noise_sim
        self.config = config or Config()

    def _normalise_targets(self, speed: float, angle: float, spin: float) -> np.ndarray:
        """Map raw shot parameters to [0, 1] using Config bounds."""
        cfg = self.config
        speed_n = (speed - cfg.speed_min) / (cfg.speed_max - cfg.speed_min)
        angle_n = (angle - cfg.angle_min) / (cfg.angle_max - cfg.angle_min)
        spin_n  = (spin  - cfg.spin_min)  / (cfg.spin_max  - cfg.spin_min)
        return np.array([speed_n, angle_n, spin_n], dtype=np.float32)

    def denormalise(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalised [0,1] targets back to physical units.

        Parameters
        ----------
        y_norm : np.ndarray, shape (..., 3)  — [speed_n, angle_n, spin_n]

        Returns
        -------
        np.ndarray same shape — [speed_mps, angle_deg, spin_rpm]
        """
        cfg = self.config
        out = y_norm.copy().astype(np.float32)
        out[..., 0] = y_norm[..., 0] * (cfg.speed_max - cfg.speed_min) + cfg.speed_min
        out[..., 1] = y_norm[..., 1] * (cfg.angle_max - cfg.angle_min) + cfg.angle_min
        out[..., 2] = y_norm[..., 2] * (cfg.spin_max  - cfg.spin_min)  + cfg.spin_min
        return out

    def _pad_or_truncate(self, arr: np.ndarray, max_len: int) -> np.ndarray:
        """Zero-pad or truncate a 1-D array to exactly max_len."""
        if len(arr) >= max_len:
            return arr[:max_len]
        padded = np.zeros(max_len, dtype=arr.dtype)
        padded[:len(arr)] = arr
        return padded

    def generate(
        self,
        n_samples: int = 10_000,
        noise_levels: list[float] = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a dataset of IMU windows and normalised shot parameters.

        Parameters
        ----------
        n_samples    : Number of (X, y) pairs to generate
        noise_levels : List of noise level multipliers to randomly select from
        seed         : RNG seed for reproducibility

        Returns
        -------
        X : np.ndarray, shape (n_samples, max_timesteps, 3)
            Channels: [acc_x, acc_y, gyro_z]
        y : np.ndarray, shape (n_samples, 3)
            Normalised [speed, angle, spin] in [0, 1]
        """
        if noise_levels is None:
            noise_levels = [0.5, 1.0, 2.0]

        cfg = self.config
        rng = np.random.default_rng(seed)
        # Re-seed noise sim for full reproducibility
        self.noise_sim.rng = np.random.default_rng(seed + 1)

        max_ts = _MAX_TIMESTEPS
        X = np.zeros((n_samples, max_ts, 3), dtype=np.float32)
        y = np.zeros((n_samples, 3), dtype=np.float32)

        for i in tqdm(range(n_samples), desc="Generating dataset", leave=False):
            speed = float(rng.uniform(cfg.speed_min, cfg.speed_max))
            angle = float(rng.uniform(cfg.angle_min, cfg.angle_max))
            spin  = float(rng.uniform(cfg.spin_min,  cfg.spin_max))
            noise = float(rng.choice(noise_levels))

            traj = self.physics_model.simulate(speed, angle, spin)
            imu  = self.noise_sim.simulate_imu(traj, noise_level=noise)

            # Partial trajectory: reveal only a random fraction of timesteps.
            # The remainder is zero-padded, simulating early-observation inference.
            n_full = len(imu.t)
            frac = float(rng.uniform(cfg.partial_traj_min, cfg.partial_traj_max))
            n_reveal = max(1, int(frac * n_full))

            X[i, :, 0] = self._pad_or_truncate(imu.acc_x[:n_reveal],  max_ts)
            X[i, :, 1] = self._pad_or_truncate(imu.acc_y[:n_reveal],  max_ts)
            X[i, :, 2] = self._pad_or_truncate(imu.gyro_z[:n_reveal], max_ts)
            y[i] = self._normalise_targets(speed, angle, spin)

        logger.info("Generated dataset: X=%s y=%s", X.shape, y.shape)
        return X, y

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple:
        """Split dataset into train / val / test with fixed seed shuffle.

        Returns
        -------
        (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        rng = np.random.default_rng(self.config.random_seed)
        idx = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        return (
            X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx],
        )
