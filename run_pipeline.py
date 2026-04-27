"""
run_pipeline.py — End-to-end pipeline for Ball Flight Physics Estimator.

Steps:
1. Generate (or load cached) 10,000-sample synthetic dataset
2. Split into train / val / test
3. Train NNEstimator (50 epochs), save best checkpoint
4. Benchmark PhysicsEstimator vs. NNEstimator on test set at noise levels [0.5, 1.0, 2.0]
5. Print final results table (MAE + RMSE per parameter per estimator per noise level)
6. Save all visualisation plots to outputs/plots/
"""
import logging
import numpy as np
from pathlib import Path

from src.utils.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("outputs")
PLOT_DIR     = OUTPUT_DIR / "plots"
CACHE_X      = OUTPUT_DIR / "dataset_X.npy"
CACHE_Y      = OUTPUT_DIR / "dataset_y.npy"
CHECKPOINT   = OUTPUT_DIR / "nn_estimator_best.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Imports ────────────────────────────────────────────────────────────────────
from src.utils.config import Config
from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.data.dataset_generator import DatasetGenerator
from src.models.physics_estimator import PhysicsEstimator
from src.models.nn_estimator import NNEstimator
from src.evaluation.benchmarker import Benchmarker
from src.utils.visualization import (
    plot_trajectory,
    plot_trajectory_family,
    plot_imu_signal,
    plot_3d_shot_arc,
    plot_benchmark_comparison,
    plot_parity,
)


def main() -> None:
    config = Config()

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    physics_model = BallPhysicsModel()
    noise_sim     = SensorNoiseSimulator(seed=config.random_seed)
    generator     = DatasetGenerator(physics_model, noise_sim)

    if CACHE_X.exists() and CACHE_Y.exists():
        logger.info("Loading cached dataset from %s", OUTPUT_DIR)
        X = np.load(CACHE_X)
        y = np.load(CACHE_Y)
    else:
        logger.info("Generating %d-sample dataset …", config.n_samples)
        X, y = generator.generate(n_samples=config.n_samples, seed=config.random_seed)
        np.save(CACHE_X, X)
        np.save(CACHE_Y, y)
        logger.info("Dataset cached → %s", OUTPUT_DIR)

    logger.info("Dataset shapes — X: %s  y: %s", X.shape, y.shape)

    # ── 2. Split ───────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = generator.split(X, y)
    logger.info("Train %d | Val %d | Test %d", len(X_train), len(X_val), len(X_test))

    # ── 3. Train NNEstimator ───────────────────────────────────────────────────
    train_logger = ExperimentLogger(
        run_name=f"nn_e{config.epochs}_bs{config.batch_size}",
        config=config,
        run_type="training",
    )
    train_logger.log_dataset_summary(
        n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
        n_samples_total=config.n_samples,
    )

    nn_estimator = NNEstimator(config)
    if CHECKPOINT.exists():
        logger.info("Loading existing NN checkpoint from %s", CHECKPOINT)
        nn_estimator = NNEstimator.load(str(CHECKPOINT))
    else:
        logger.info("Training NNEstimator for %d epochs …", config.epochs)
        nn_estimator.fit(X_train, y_train, X_val, y_val,
                         epoch_callback=train_logger.log_epoch)
        nn_estimator.save(str(CHECKPOINT))
        logger.info("Checkpoint saved → %s", CHECKPOINT)

    test_metrics = nn_estimator.evaluate_on_test(X_test, y_test)
    train_logger.log_final_metrics(test_metrics)
    train_logger.log_artefact(config.checkpoint_path)
    train_logger.log_artefact("outputs/plots/training_curves.png")
    train_run_dir = train_logger.finish()
    logger.info("Training run saved to: %s", train_run_dir)

    # ── 4. Benchmark ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(config.random_seed)
    test_trajectories = [
        physics_model.simulate(
            float(generator.denormalise(y_test[i:i+1])[0, 0]),
            float(generator.denormalise(y_test[i:i+1])[0, 1]),
            float(generator.denormalise(y_test[i:i+1])[0, 2]),
        )
        for i in range(min(200, len(y_test)))
    ]

    bench_logger = ExperimentLogger(
        run_name="benchmark_noise_sweep",
        config=config,
        run_type="benchmark",
    )
    bench_logger.log_dataset_summary(0, 0, len(test_trajectories), len(test_trajectories))

    benchmarker = Benchmarker(PhysicsEstimator(), nn_estimator, SensorNoiseSimulator(0))
    results_df  = benchmarker.evaluate(test_trajectories)

    bench_logger.log_benchmark_results(results_df)
    bench_logger.log_artefact("outputs/plots/benchmark_comparison.png")
    bench_logger.log_artefact("outputs/benchmark_results.csv")

    # ── 5. Print results table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS — MAE and RMSE per estimator per noise level")
    print("=" * 72)
    cols = ["estimator", "noise_level",
            "speed_mae", "angle_mae", "spin_mae",
            "speed_rmse", "angle_rmse", "spin_rmse"]
    print(results_df[cols].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print("=" * 72 + "\n")

    # ── 6. Visualisations ──────────────────────────────────────────────────────
    logger.info("Saving visualisation plots …")

    # Sample trajectory
    sample_traj = physics_model.simulate(20.0, 35.0, spin_rpm=1000.0)
    plot_trajectory(sample_traj, save_path=PLOT_DIR / "trajectory.png")

    # Trajectory family (3×3 grid)
    plot_trajectory_family(physics_model, save_path=PLOT_DIR / "trajectory_family.png")

    # IMU signal (clean + noisy overlay)
    clean_imu = SensorNoiseSimulator(seed=0).simulate_imu(sample_traj, noise_level=0.0)
    noisy_imu = SensorNoiseSimulator(seed=0).simulate_imu(sample_traj, noise_level=1.0)
    plot_imu_signal(clean_imu, save_path=PLOT_DIR / "imu_signal.png", noisy_imu=noisy_imu)

    # 3D shot arc
    plot_3d_shot_arc(sample_traj, save_path=PLOT_DIR / "shot_arc_3d.png")

    # Benchmark comparison bar chart
    plot_benchmark_comparison(results_df, save_path=PLOT_DIR / "benchmark_comparison.png")

    # Parity plots — NN predictions on test set
    X_test_sub = X_test[:500]
    y_test_sub = generator.denormalise(y_test[:500])
    nn_preds   = nn_estimator.predict(X_test_sub)
    plot_parity(y_test_sub, nn_preds, save_path=PLOT_DIR / "parity_plots.png")

    # List saved plots
    saved = sorted(PLOT_DIR.glob("*.png"))
    logger.info("Plots saved (%d):", len(saved))
    for p in saved:
        logger.info("  %s", p)

    # ── 7. Export benchmark CSV and finish benchmark logger ────────────────────
    results_df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    bench_run_dir = bench_logger.finish()
    logger.info("Benchmark run saved to: %s", bench_run_dir)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
