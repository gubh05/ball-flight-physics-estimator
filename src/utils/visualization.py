import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

from src.physics.ball_physics_model import BallPhysicsModel, TrajectoryResult
from src.sensors.sensor_noise_simulator import IMUReading, SensorNoiseSimulator

logger = logging.getLogger(__name__)

_STYLE = "seaborn-v0_8-whitegrid"


def plot_trajectory(
    trajectory: TrajectoryResult,
    save_path: Path,
) -> None:
    """Single trajectory: x vs y position, labelled with launch parameters."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(trajectory.x, trajectory.y, lw=2, color="steelblue")
        ax.fill_between(trajectory.x, 0, trajectory.y, alpha=0.08, color="steelblue")
        ax.axhline(0, color="saddlebrown", lw=1.5, ls="--", alpha=0.6, label="Ground")
        label = (
            f"v0 = {trajectory.launch_speed_mps:.1f} m/s  "
            f"angle = {trajectory.launch_angle_deg:.1f} deg  "
            f"spin = {trajectory.spin_rpm:.0f} rpm"
        )
        ax.set_title(f"Ball Trajectory\n{label}", fontsize=11)
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Height (m)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Saved trajectory plot → %s", save_path)


def plot_trajectory_family(
    model: BallPhysicsModel,
    save_path: Path,
) -> None:
    """3×3 grid of trajectories varying speed (low/med/high) × spin (topspin/none/backspin)."""
    speeds = [10.0, 20.0, 30.0]
    spins = [1500.0, 0.0, -1500.0]
    spin_labels = ["Topspin (+1500 rpm)", "No spin (0 rpm)", "Backspin (−1500 rpm)"]
    speed_labels = ["Low (10 m/s)", "Medium (20 m/s)", "High (30 m/s)"]
    colors = ["#e15759", "#59a14f", "#4e79a7"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=False)
        for row, (spin, spin_lbl) in enumerate(zip(spins, spin_labels)):
            for col, (speed, speed_lbl) in enumerate(zip(speeds, speed_labels)):
                ax = axes[row][col]
                traj = model.simulate(speed, 35.0, spin_rpm=spin)
                ax.plot(traj.x, traj.y, lw=1.8, color=colors[col])
                ax.fill_between(traj.x, 0, traj.y, alpha=0.07, color=colors[col])
                ax.axhline(0, color="saddlebrown", lw=1, ls="--", alpha=0.5)
                ax.set_ylim(bottom=0)
                if row == 0:
                    ax.set_title(speed_lbl, fontsize=9, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(spin_lbl + "\nHeight (m)", fontsize=8)
                if row == 2:
                    ax.set_xlabel("Distance (m)", fontsize=8)
        fig.suptitle(
            "Trajectory Family: Speed × Spin (Magnus Effect)", fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Saved trajectory family plot → %s", save_path)


def plot_imu_signal(
    imu: IMUReading,
    save_path: Path,
    noisy_imu: IMUReading = None,
) -> None:
    """acc_x, acc_y, gyro_z over time — clean signal, with optional noisy overlay."""
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channels = [
            (imu.acc_x, "Acc X (m/s²)", "#4e79a7"),
            (imu.acc_y, "Acc Y (m/s²)", "#f28e2b"),
            (imu.gyro_z, "Gyro Z (rad/s)", "#e15759"),
        ]
        noisy_channels = (
            [noisy_imu.acc_x, noisy_imu.acc_y, noisy_imu.gyro_z] if noisy_imu else None
        )

        for i, (ax, (signal, label, color)) in enumerate(zip(axes, channels)):
            ax.plot(imu.t, signal, lw=1.5, color=color, label="Clean" if noisy_imu else label)
            if noisy_channels is not None:
                ax.plot(
                    noisy_imu.t,
                    noisy_channels[i],
                    lw=0.8,
                    alpha=0.55,
                    color="grey",
                    label="Noisy",
                )
                ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel(label, fontsize=9)

        axes[-1].set_xlabel("Time (s)")
        traj = imu.true_trajectory
        title = (
            f"IMU Signal — v0={traj.launch_speed_mps:.1f} m/s, "
            f"angle={traj.launch_angle_deg:.1f} deg, spin={traj.spin_rpm:.0f} rpm"
        )
        axes[0].set_title(title, fontsize=10)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Saved IMU signal plot → %s", save_path)


def plot_3d_shot_arc(
    trajectory: TrajectoryResult,
    save_path: Path,
) -> None:
    """3D plot: x (distance), y (height), z-axis = time — time-evolving arc."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    t = trajectory.t
    x = trajectory.x
    y = trajectory.y

    # Colour by time for visual flair
    n = len(t)
    cmap = plt.cm.plasma
    for i in range(n - 1):
        frac = i / max(n - 2, 1)
        ax.plot(
            x[i : i + 2],
            [t[i], t[i + 1]],
            y[i : i + 2],
            color=cmap(frac),
            lw=1.5,
        )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Height (m)")
    label = (
        f"v0={trajectory.launch_speed_mps:.1f} m/s, "
        f"angle={trajectory.launch_angle_deg:.1f} deg, "
        f"spin={trajectory.spin_rpm:.0f} rpm"
    )
    ax.set_title(f"3D Shot Arc\n{label}", fontsize=10)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved 3D shot arc plot → %s", save_path)


def plot_benchmark_comparison(
    results,  # pd.DataFrame with columns [estimator, noise_level, speed_mae, angle_mae, spin_mae]
    save_path: Path,
) -> None:
    """Grouped bar chart: Physics vs. NN MAE per parameter at each noise level."""
    import pandas as pd

    with plt.style.context(_STYLE):
        metrics = ["speed_mae", "angle_mae", "spin_mae"]
        metric_labels = ["Speed MAE (m/s)", "Angle MAE (°)", "Spin MAE (rpm)"]
        estimators = results["estimator"].unique()
        noise_levels = sorted(results["noise_level"].unique())

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        bar_colors = {"PhysicsEstimator": "#4e79a7", "NNEstimator": "#e15759"}
        x = np.arange(len(noise_levels))
        width = 0.35

        for ax, metric, ylabel in zip(axes, metrics, metric_labels):
            for k, est in enumerate(estimators):
                subset = results[results["estimator"] == est]
                means = [
                    subset[subset["noise_level"] == nl][metric].mean()
                    for nl in noise_levels
                ]
                offset = (k - (len(estimators) - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    means,
                    width,
                    label=est,
                    color=bar_colors.get(est, f"C{k}"),
                    alpha=0.85,
                )
                ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"σ={nl}" for nl in noise_levels])
            ax.set_xlabel("Noise level")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" (")[0])
            ax.legend(fontsize=8)

        fig.suptitle("Physics Estimator vs. NN Estimator — MAE Comparison", fontsize=12)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Saved benchmark comparison plot → %s", save_path)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    param_names: list = None,
) -> None:
    """Parity (predicted vs. true) scatter plots for all 3 shot parameters."""
    if param_names is None:
        param_names = ["Launch Speed (m/s)", "Launch Angle (°)", "Spin (rpm)"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        colors = ["#4e79a7", "#f28e2b", "#59a14f"]

        for ax, col, name, color in zip(axes, range(3), param_names, colors):
            true_col = y_true[:, col]
            pred_col = y_pred[:, col]
            ax.scatter(true_col, pred_col, s=12, alpha=0.45, color=color, edgecolors="none")
            lo = min(true_col.min(), pred_col.min())
            hi = max(true_col.max(), pred_col.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
            mae = np.mean(np.abs(true_col - pred_col))
            ax.set_xlabel(f"True {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_title(f"{name}\nMAE = {mae:.2f}")
            ax.legend(fontsize=8)

        fig.suptitle("NN Estimator — Parity Plots (Predicted vs. True)", fontsize=12)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Saved parity plot → %s", save_path)
