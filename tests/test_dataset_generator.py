"""Tests for DatasetGenerator — shapes, normalisation, reproducibility, split."""
import numpy as np
import pytest

from src.data.dataset_generator import DatasetGenerator
from src.physics.ball_physics_model import BallPhysicsModel
from src.sensors.sensor_noise_simulator import SensorNoiseSimulator
from src.utils.config import Config


@pytest.fixture(scope="module")
def gen():
    return DatasetGenerator(BallPhysicsModel(), SensorNoiseSimulator(seed=42), Config())


@pytest.fixture(scope="module")
def small_dataset(gen):
    return gen.generate(n_samples=50, seed=42)


def test_X_shape(small_dataset):
    X, _ = small_dataset
    assert X.shape == (50, 300, 3), f"X shape wrong: {X.shape}"


def test_y_shape(small_dataset):
    _, y = small_dataset
    assert y.shape == (50, 3), f"y shape wrong: {y.shape}"


def test_y_normalised_in_unit_interval(small_dataset):
    _, y = small_dataset
    assert y.min() >= 0.0, f"y.min()={y.min():.4f} < 0"
    assert y.max() <= 1.0, f"y.max()={y.max():.4f} > 1"


def test_no_nans_in_X(small_dataset):
    X, _ = small_dataset
    assert not np.isnan(X).any(), "NaN found in X"


def test_no_nans_in_y(small_dataset):
    _, y = small_dataset
    assert not np.isnan(y).any(), "NaN found in y"


def test_reproducibility(gen):
    X1, y1 = gen.generate(n_samples=30, seed=99)
    X2, y2 = gen.generate(n_samples=30, seed=99)
    assert np.allclose(X1, X2), "X not reproducible with same seed"
    assert np.allclose(y1, y2), "y not reproducible with same seed"


def test_different_seeds_give_different_data(gen):
    X1, _ = gen.generate(n_samples=20, seed=1)
    X2, _ = gen.generate(n_samples=20, seed=2)
    assert not np.allclose(X1, X2), "Different seeds should give different data"


def test_split_sizes(gen, small_dataset):
    X, y = small_dataset
    X_tr, X_v, X_te, y_tr, y_v, y_te = gen.split(X, y)
    assert len(X_tr) + len(X_v) + len(X_te) == 50
    assert len(X_tr) > len(X_v)
    assert len(X_tr) > len(X_te)


def test_split_no_overlap(gen, small_dataset):
    """All split indices together should cover the full dataset."""
    X, y = small_dataset
    X_tr, X_v, X_te, y_tr, y_v, y_te = gen.split(X, y)
    total = len(X_tr) + len(X_v) + len(X_te)
    assert total == len(X)


def test_X_dtype_float32(small_dataset):
    X, _ = small_dataset
    assert X.dtype == np.float32


def test_y_dtype_float32(small_dataset):
    _, y = small_dataset
    assert y.dtype == np.float32


def test_denormalise_round_trip(gen):
    """Denormalise(normalise(params)) should recover original params."""
    cfg = gen.config
    test_y_norm = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    denorm = gen.denormalise(test_y_norm)
    speed_expected = 0.5 * (cfg.speed_max - cfg.speed_min) + cfg.speed_min
    angle_expected = 0.5 * (cfg.angle_max - cfg.angle_min) + cfg.angle_min
    spin_expected  = 0.5 * (cfg.spin_max  - cfg.spin_min)  + cfg.spin_min
    np.testing.assert_allclose(denorm[0, 0], speed_expected, rtol=1e-5)
    np.testing.assert_allclose(denorm[0, 1], angle_expected, rtol=1e-5)
    np.testing.assert_allclose(denorm[0, 2], spin_expected,  rtol=1e-5)
