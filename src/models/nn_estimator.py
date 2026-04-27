import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import Config

logger = logging.getLogger(__name__)


class _CNN(nn.Module):
    """1D-CNN backbone for shot-parameter regression."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class NNEstimator:
    """PyTorch 1D-CNN that estimates shot parameters from raw IMU sequences.

    Input to fit/predict: X of shape (N, 300, 3) — [acc_x, acc_y, gyro_z].
    Internally transposed to (N, 3, 300) for Conv1d (channels-first).

    fit() trains on normalised labels (y in [0,1]).
    predict() returns *denormalised* physical values:
        column 0: launch_speed_mps
        column 1: launch_angle_deg
        column 2: spin_rpm
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _CNN().to(self.device)
        self._trained = False

        # Denormalisation scale/shift (derived from Config bounds)
        cfg = self.config
        self._scale = np.array(
            [cfg.speed_max - cfg.speed_min, cfg.angle_max - cfg.angle_min, cfg.spin_max - cfg.spin_min],
            dtype=np.float32,
        )
        self._shift = np.array(
            [cfg.speed_min, cfg.angle_min, cfg.spin_min],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epoch_callback=None) -> "NNEstimator":
        """Train the model.

        Parameters
        ----------
        X_train, X_val : (N, 300, 3) float32 arrays
        y_train, y_val : (N, 3) float32 arrays with normalised labels in [0,1]
        epoch_callback : optional callable(epoch, train_loss, val_loss) called each epoch
        """
        cfg = self.config
        model = self._model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = math.inf
        best_state = None

        for epoch in range(cfg.epochs):
            # --- train ---
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            # --- validate ---
            val_loss = self._eval_loss(val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            logger.info(
                "Epoch %3d/%d  train=%.5f  val=%.5f", epoch + 1, cfg.epochs, train_loss, val_loss
            )
            if epoch_callback is not None:
                epoch_callback(epoch + 1, train_loss, val_loss)

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        self._trained = True
        logger.info("Training complete. Best val MSE: %.5f", best_val_loss)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return denormalised predictions.

        Parameters
        ----------
        X : (N, 300, 3) float32 array

        Returns
        -------
        np.ndarray of shape (N, 3): [speed_mps, angle_deg, spin_rpm]
        """
        self._model.eval()
        loader = self._make_loader(X, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                preds.append(self._model(xb).cpu().numpy())
        y_norm = np.concatenate(preds, axis=0)
        return y_norm * self._scale + self._shift

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to a .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info("NNEstimator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "NNEstimator":
        """Load a previously saved NNEstimator."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        estimator = cls(checkpoint["config"])
        estimator._model.load_state_dict(checkpoint["model_state"])
        estimator._model.to(estimator.device)
        estimator._trained = True
        logger.info("NNEstimator loaded from %s", path)
        return estimator

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_on_test(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate on test set, returning MAE/RMSE per parameter in physical units.

        Parameters
        ----------
        X : (N, 300, 3) float32 array
        y : (N, 3) float32 array with normalised labels in [0, 1]

        Returns
        -------
        dict with keys: speed_mae, angle_mae, spin_mae,
                        speed_rmse, angle_rmse, spin_rmse, overall_mse
        """
        preds = self.predict(X)          # denormalised (N, 3)
        y_denorm = y * self._scale + self._shift  # (N, 3)

        errors = preds - y_denorm
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        return {
            "speed_mae":   float(abs_errors[:, 0].mean()),
            "angle_mae":   float(abs_errors[:, 1].mean()),
            "spin_mae":    float(abs_errors[:, 2].mean()),
            "speed_rmse":  float(np.sqrt(sq_errors[:, 0].mean())),
            "angle_rmse":  float(np.sqrt(sq_errors[:, 1].mean())),
            "spin_rmse":   float(np.sqrt(sq_errors[:, 2].mean())),
            "overall_mse": float(sq_errors.mean()),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X: np.ndarray, y: np.ndarray = None,
                     shuffle: bool = False) -> DataLoader:
        """Build a DataLoader; transpose X from (N,T,C) to (N,C,T)."""
        X_t = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))
        if y is not None:
            dataset = TensorDataset(X_t, torch.from_numpy(y))
        else:
            dataset = TensorDataset(X_t)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self._model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total_loss += criterion(self._model(xb), yb).item() * len(xb)
                total_n += len(xb)
        return total_loss / total_n if total_n > 0 else 0.0
