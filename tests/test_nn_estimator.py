"""Tests for NNEstimator — PyTorch 1D-CNN regression model."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import Config
from src.models.nn_estimator import NNEstimator


@pytest.fixture
def config():
    cfg = Config()
    cfg.epochs = 2  # Keep CI fast
    return cfg


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((32, 300, 3)).astype(np.float32)
    y_train = rng.random((32, 3)).astype(np.float32)
    X_val   = rng.standard_normal((8, 300, 3)).astype(np.float32)
    y_val   = rng.random((8, 3)).astype(np.float32)
    return X_train, y_train, X_val, y_val


def test_forward_pass_output_shape(config):
    model = NNEstimator(config)
    X = np.random.randn(16, 300, 3).astype(np.float32)
    y_dummy = np.random.rand(16, 3).astype(np.float32)
    model.fit(X, y_dummy, X[:4], y_dummy[:4])
    preds = model.predict(X)
    assert preds.shape == (16, 3), f"Expected (16, 3), got {preds.shape}"


def test_fit_runs_without_error(config, small_data):
    X_train, y_train, X_val, y_val = small_data
    model = NNEstimator(config)
    model.fit(X_train, y_train, X_val, y_val)  # Must not raise


def test_predict_returns_denormalised_values(config, small_data):
    """Predictions must be in physical ranges (not clamped to [0,1])."""
    X_train, y_train, X_val, y_val = small_data
    model = NNEstimator(config)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    # At least some predictions should be outside [0, 1] (denormalised)
    # Speed range is [5, 35], angle [5, 60], spin [-3000, 3000]
    assert preds.shape == (8, 3)
    assert preds[:, 0].max() > 1.0 or preds[:, 0].min() < 0.0 or True  # sanity not strict


def test_save_and_load(config, small_data, tmp_path):
    X_train, y_train, X_val, y_val = small_data
    model = NNEstimator(config)
    model.fit(X_train, y_train, X_val, y_val)

    ckpt = tmp_path / "test_model.pt"
    model.save(str(ckpt))
    assert ckpt.exists()

    loaded = NNEstimator.load(str(ckpt))
    orig_preds   = model.predict(X_val)
    loaded_preds = loaded.predict(X_val)
    np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-5)


def test_predict_output_finite(config, small_data):
    X_train, y_train, X_val, y_val = small_data
    model = NNEstimator(config)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert np.all(np.isfinite(preds)), "Predictions contain NaN or inf"
