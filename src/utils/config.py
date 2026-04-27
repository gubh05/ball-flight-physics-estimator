from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data generation
    n_samples: int = 10_000
    max_timesteps: int = 300
    sample_rate_hz: int = 100
    random_seed: int = 42

    # Shot parameter bounds (for normalization)
    speed_min: float = 5.0
    speed_max: float = 35.0
    angle_min: float = 5.0
    angle_max: float = 60.0
    spin_min: float = -1000.0
    spin_max: float = 1000.0

    # Partial trajectory (fraction of IMU signal revealed at generation time)
    partial_traj_min: float = 0.3   # minimum fraction of timesteps revealed
    partial_traj_max: float = 1.0   # maximum fraction (1.0 = full trajectory)

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    mc_dropout_passes: int = 30     # forward passes for MC Dropout uncertainty

    # Paths
    output_dir: Path = Path("outputs/")
    checkpoint_path: Path = Path("outputs/nn_estimator_best.pt")
