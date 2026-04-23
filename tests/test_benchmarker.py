"""Tests for Benchmarker — head-to-head evaluation of estimators."""
import numpy as np
import pandas as pd
import pytest

from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.models.physics_estimator import PhysicsEstimator
from src.models.nn_estimator import NNEstimator
from src.evaluation.benchmarker import Benchmarker
from src.data.dataset_generator import DatasetGenerator
from src.utils.config import Config


@pytest.fixture(scope="module")
def trained_nn():
    config = Config()
    config.epochs = 1
    gen = DatasetGenerator(BallPhysicsModel(), SensorNoiseSimulator(42))
    X, y = gen.generate(n_samples=50, seed=42)
    nn = NNEstimator(config)
    nn.fit(X[:40], y[:40], X[40:], y[40:])
    return nn


@pytest.fixture(scope="module")
def test_trajectories():
    model = BallPhysicsModel()
    rng = np.random.default_rng(7)
    return [
        model.simulate(rng.uniform(10, 25), rng.uniform(15, 45), rng.uniform(-1000, 1000))
        for _ in range(8)
    ]


@pytest.fixture(scope="module")
def benchmark_results(trained_nn, test_trajectories):
    benchmarker = Benchmarker(PhysicsEstimator(), trained_nn, SensorNoiseSimulator(0))
    return benchmarker.evaluate(test_trajectories)


def test_results_is_dataframe(benchmark_results):
    assert isinstance(benchmark_results, pd.DataFrame)


def test_required_columns_present(benchmark_results):
    required = ["estimator", "noise_level", "speed_mae", "angle_mae", "spin_mae"]
    for col in required:
        assert col in benchmark_results.columns, f"Missing column: {col}"


def test_rmse_columns_present(benchmark_results):
    for col in ["speed_rmse", "angle_rmse", "spin_rmse"]:
        assert col in benchmark_results.columns, f"Missing RMSE column: {col}"


def test_no_nans_in_results(benchmark_results):
    assert not benchmark_results.isnull().values.any(), "NaNs found in benchmark results"


def test_both_estimators_present(benchmark_results):
    estimators = set(benchmark_results["estimator"].unique())
    assert "PhysicsEstimator" in estimators
    assert "NNEstimator" in estimators


def test_all_noise_levels_present(benchmark_results):
    noise_levels = set(benchmark_results["noise_level"].unique())
    assert {0.5, 1.0, 2.0}.issubset(noise_levels)


def test_mae_values_are_positive(benchmark_results):
    for col in ["speed_mae", "angle_mae", "spin_mae"]:
        assert (benchmark_results[col] >= 0).all(), f"Negative MAE in {col}"


def test_row_count(benchmark_results):
    # 2 estimators × 3 noise levels = 6 rows
    assert len(benchmark_results) == 6, f"Expected 6 rows, got {len(benchmark_results)}"
