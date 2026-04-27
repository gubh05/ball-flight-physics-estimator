"""
experiment_logger.py — Lightweight, file-based experiment tracker.

Supports two run types:
  - "training"  : logs epoch losses + final per-parameter MSE/MAE
  - "benchmark" : logs Physics vs. NN comparison results across noise levels

Creates a timestamped run directory under outputs/runs/ and writes:
  - run_config.json   : hyperparameters, environment, git info, dataset summary
  - metrics.json      : training curves OR benchmark results table
  - run_report.md     : human-readable summary
"""
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


# ── JSON helpers ────────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Environment / git helpers ────────────────────────────────────────────────────


def _get_env_info() -> dict:
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
    except ImportError:
        env["torch_version"] = "not installed"
    env["numpy_version"] = np.__version__
    return env


def _get_git_info() -> dict:
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return {"commit_hash": commit_hash, "branch": branch, "dirty": bool(dirty_output)}
    except Exception:
        return {"commit_hash": "unknown", "branch": "unknown", "dirty": False}


# ── Formatting helpers ───────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ── Main class ───────────────────────────────────────────────────────────────────


class ExperimentLogger:
    """
    Lightweight, file-based experiment tracker.

    Supports two run types:
      - "training"  : logs epoch losses + final per-parameter MSE/MAE
      - "benchmark" : logs Physics vs. NN comparison results across noise levels

    Creates a timestamped run directory under outputs/runs/ and writes:
      - run_config.json   : hyperparameters, environment, git info, dataset summary
      - metrics.json      : training curves OR benchmark results table
      - run_report.md     : human-readable summary
    """

    def __init__(
        self,
        run_name: str,
        config,
        run_type: str = "training",
        base_dir: Path = Path("outputs/runs"),
    ):
        """
        Args:
            run_name:  Short label, e.g. "nn_e50_bs128" or "benchmark_noise_sweep".
            config:    The project Config dataclass instance.
            run_type:  "training" or "benchmark".
            base_dir:  Root directory where run folders are created.
        """
        self.run_name = run_name
        self.config = config
        self.run_type = run_type
        self.base_dir = Path(base_dir)

        now = datetime.now(timezone.utc)
        self.run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{run_name}"
        self.started_at = now
        self._start_time = time.time()

        try:
            self.run_dir = self.base_dir / self.run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log.warning("ExperimentLogger: could not create run dir: %s", exc)
            self.run_dir = self.base_dir / self.run_id

        # Internal state
        self._epoch_log: list = []
        self._dataset_summary: dict = {}
        self._final_metrics: dict = {}
        self._benchmark_results: list = []
        self._artefacts: list = []

    # ── Training run methods ─────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, extra: dict = None) -> None:
        """Append one epoch's metrics. Call once per epoch during NN training."""
        try:
            entry = {"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)}
            if extra:
                entry.update({k: _to_native(v) for k, v in extra.items()})
            self._epoch_log.append(entry)
        except Exception as exc:
            _log.warning("ExperimentLogger.log_epoch: %s", exc)

    def log_dataset_summary(self, n_train: int, n_val: int, n_test: int, n_samples_total: int) -> None:
        """Record the dataset split used in this run."""
        try:
            self._dataset_summary = {
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(n_test),
                "n_samples_total": int(n_samples_total),
            }
        except Exception as exc:
            _log.warning("ExperimentLogger.log_dataset_summary: %s", exc)

    def log_final_metrics(self, metrics: dict) -> None:
        """Record end-of-training evaluation on the test set."""
        try:
            self._final_metrics = _to_native(metrics)
        except Exception as exc:
            _log.warning("ExperimentLogger.log_final_metrics: %s", exc)

    # ── Benchmark run methods ────────────────────────────────────────────────────

    def log_benchmark_results(self, results_df) -> None:
        """Record the full benchmark comparison DataFrame."""
        try:
            self._benchmark_results = [_to_native(row) for row in results_df.to_dict(orient="records")]
        except Exception as exc:
            _log.warning("ExperimentLogger.log_benchmark_results: %s", exc)

    # ── Shared methods ───────────────────────────────────────────────────────────

    def log_artefact(self, path) -> None:
        """Record the path of a saved artefact."""
        try:
            self._artefacts.append(str(path))
        except Exception as exc:
            _log.warning("ExperimentLogger.log_artefact: %s", exc)

    def finish(self) -> Path:
        """Finalise the run. Writes run_config.json, metrics.json, run_report.md."""
        try:
            finished_at = datetime.now(timezone.utc)
            duration = time.time() - self._start_time

            config_out = self._extract_config_dict()
            run_config = {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "run_type": self.run_type,
                "started_at": self.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_seconds": round(duration, 1),
                "config": config_out,
                "environment": _get_env_info(),
                "git": _get_git_info(),
                "artefacts": self._artefacts,
            }

            if self.run_type == "training":
                run_config["dataset"] = self._dataset_summary
            else:
                noise_levels = sorted(set(r["noise_level"] for r in self._benchmark_results))
                estimators = sorted(set(r["estimator"] for r in self._benchmark_results))
                n_test = self._dataset_summary.get("n_test", 0)
                run_config["benchmark"] = {
                    "noise_levels": noise_levels,
                    "n_test_trajectories": n_test,
                    "estimators": estimators,
                }

            metrics = (
                self._build_training_metrics()
                if self.run_type == "training"
                else self._build_benchmark_metrics()
            )

            self._write_json(self.run_dir / "run_config.json", run_config)
            self._write_json(self.run_dir / "metrics.json", metrics)

            try:
                report = self._build_report(run_config, metrics, duration)
                (self.run_dir / "run_report.md").write_text(report, encoding="utf-8")
            except Exception as exc:
                _log.warning("ExperimentLogger: failed to write run_report.md: %s", exc)

        except Exception as exc:
            _log.warning("ExperimentLogger.finish: unexpected error: %s", exc)

        return self.run_dir

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _extract_config_dict(self) -> dict:
        try:
            from dataclasses import asdict
            raw = asdict(self.config)
            numeric_keys = [
                "n_samples", "max_timesteps", "sample_rate_hz", "epochs",
                "batch_size", "learning_rate", "random_seed",
                "speed_min", "speed_max", "angle_min", "angle_max",
                "spin_min", "spin_max",
            ]
            return {k: raw[k] for k in numeric_keys if k in raw}
        except Exception:
            return {}

    def _write_json(self, path: Path, data: dict) -> None:
        try:
            path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder), encoding="utf-8")
        except Exception as exc:
            _log.warning("ExperimentLogger: failed to write %s: %s", path.name, exc)

    def _build_training_metrics(self) -> dict:
        best_val = None
        best_epoch = None
        for entry in self._epoch_log:
            if best_val is None or entry["val_loss"] < best_val:
                best_val = entry["val_loss"]
                best_epoch = entry["epoch"]
        return {
            "epoch_log": self._epoch_log,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "final_test": self._final_metrics,
        }

    def _build_benchmark_metrics(self) -> dict:
        results = self._benchmark_results
        params = ["speed", "angle", "spin"]
        mae_keys = {"speed": "speed_mae", "angle": "angle_mae", "spin": "spin_mae"}

        winner_by_parameter = {}
        for param in params:
            mae_key = mae_keys[param]
            phys_maes = [r[mae_key] for r in results if r.get("estimator") == "PhysicsEstimator" and mae_key in r]
            nn_maes = [r[mae_key] for r in results if r.get("estimator") == "NNEstimator" and mae_key in r]
            if phys_maes and nn_maes:
                winner_by_parameter[param] = (
                    "PhysicsEstimator" if sum(phys_maes) / len(phys_maes) < sum(nn_maes) / len(nn_maes)
                    else "NNEstimator"
                )
            else:
                winner_by_parameter[param] = "unknown"

        noise_levels = sorted(set(r["noise_level"] for r in results))
        physics_wins_at = None
        for nl in noise_levels:
            phys = next((r for r in results if r.get("estimator") == "PhysicsEstimator" and r.get("noise_level") == nl), None)
            nn = next((r for r in results if r.get("estimator") == "NNEstimator" and r.get("noise_level") == nl), None)
            if phys and nn:
                phys_overall = (phys.get("speed_mae", 0) + phys.get("angle_mae", 0) + phys.get("spin_mae", 0)) / 3
                nn_overall = (nn.get("speed_mae", 0) + nn.get("angle_mae", 0) + nn.get("spin_mae", 0)) / 3
                if phys_overall < nn_overall:
                    physics_wins_at = nl
                    break

        return {
            "results": results,
            "winner_by_parameter": winner_by_parameter,
            "physics_wins_at_noise_level": physics_wins_at,
        }

    def _build_report(self, run_config: dict, metrics: dict, duration: float) -> str:
        if self.run_type == "training":
            return self._build_training_report(run_config, metrics, duration)
        return self._build_benchmark_report(run_config, metrics, duration)

    def _build_training_report(self, run_config: dict, metrics: dict, duration: float) -> str:
        cfg = run_config.get("config", {})
        dataset = run_config.get("dataset", {})
        git = run_config.get("git", {})
        env = run_config.get("environment", {})
        final_test = metrics.get("final_test", {})
        epoch_log = metrics.get("epoch_log", [])
        best_val = metrics.get("best_val_loss")
        best_epoch = metrics.get("best_epoch")

        n_total = dataset.get("n_samples_total", cfg.get("n_samples", "?"))
        n_train = dataset.get("n_train", "?")
        n_val = dataset.get("n_val", "?")
        n_test = dataset.get("n_test", "?")

        final_train_loss = epoch_log[-1]["train_loss"] if epoch_log else None
        final_val_loss = epoch_log[-1]["val_loss"] if epoch_log else None

        artefacts = run_config.get("artefacts", [])
        artefacts_md = "\n".join(f"- `{a}`" for a in artefacts) if artefacts else "_None_"

        n_total_str = f"{n_total:,}" if isinstance(n_total, int) else str(n_total)
        best_val_str = f"{best_val:.4f} (epoch {best_epoch})" if best_val is not None else "N/A"
        final_train_str = f"{final_train_loss:.4f}" if final_train_loss is not None else "N/A"
        final_val_str = f"{final_val_loss:.4f}" if final_val_loss is not None else "N/A"

        def _fmt_mae_rmse(mae_key, rmse_key, fmt=".2f"):
            mae = final_test.get(mae_key)
            rmse = final_test.get(rmse_key)
            if mae is not None and rmse is not None:
                return f"{mae:{fmt}}", f"{rmse:{fmt}}"
            return "N/A", "N/A"

        sp_mae, sp_rmse = _fmt_mae_rmse("speed_mae", "speed_rmse")
        an_mae, an_rmse = _fmt_mae_rmse("angle_mae", "angle_rmse")
        sn_mae, sn_rmse = _fmt_mae_rmse("spin_mae", "spin_rmse", fmt=".1f")

        lines = [
            "# Training Run Report",
            f"**Run ID:** {run_config['run_id']}",
            f"**Date:** {run_config['started_at'].replace('T', ' ').replace('Z', ' UTC')}",
            f"**Duration:** {_fmt_duration(duration)}",
            "",
            "## Configuration",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Epochs | {cfg.get('epochs', '?')} |",
            f"| Batch Size | {cfg.get('batch_size', '?')} |",
            f"| Learning Rate | {cfg.get('learning_rate', '?')} |",
            f"| Dataset Size | {n_total_str} shots |",
            f"| Train / Val / Test | {n_train} / {n_val} / {n_test} |",
            f"| Random Seed | {cfg.get('random_seed', '?')} |",
            "",
            "## Training Summary",
            f"- Best validation loss: {best_val_str}",
            f"- Final train loss: {final_train_str}",
            f"- Final val loss: {final_val_str}",
            "",
            "## Test Set Results",
            "| Parameter | MAE | RMSE |",
            "|-----------|-----|------|",
            f"| Launch Speed (m/s) | {sp_mae} | {sp_rmse} |",
            f"| Launch Angle (°) | {an_mae} | {an_rmse} |",
            f"| Spin Rate (RPM) | {sn_mae} | {sn_rmse} |",
            "",
            "## Saved Artefacts",
            artefacts_md,
            "",
            "## Environment",
            f"- Python: {env.get('python_version', '?')} | PyTorch: {env.get('torch_version', '?')}",
            f"- Git commit: {git.get('commit_hash', '?')} ({git.get('branch', '?')})",
        ]
        return "\n".join(lines) + "\n"

    def _build_benchmark_report(self, run_config: dict, metrics: dict, duration: float) -> str:
        benchmark = run_config.get("benchmark", {})
        noise_levels = benchmark.get("noise_levels", [])
        n_test = benchmark.get("n_test_trajectories", "?")
        estimators = benchmark.get("estimators", [])
        results = metrics.get("results", [])
        winner_by_param = metrics.get("winner_by_parameter", {})
        physics_wins_at = metrics.get("physics_wins_at_noise_level")
        git = run_config.get("git", {})
        env = run_config.get("environment", {})
        artefacts = run_config.get("artefacts", [])
        artefacts_md = "\n".join(f"- `{a}`" for a in artefacts) if artefacts else "_None_"

        n_test_str = f"{n_test:,}" if isinstance(n_test, int) else str(n_test)
        noise_str = ", ".join(str(n) for n in noise_levels)
        estimators_str = ", ".join(estimators)

        lines = [
            "# Benchmark Run Report",
            f"**Run ID:** {run_config['run_id']}",
            f"**Date:** {run_config['started_at'].replace('T', ' ').replace('Z', ' UTC')}",
            f"**Duration:** {_fmt_duration(duration)}",
            "",
            "## Setup",
            f"- Test trajectories: {n_test_str}",
            f"- Noise levels evaluated: {noise_str}",
            f"- Estimators: {estimators_str}",
            "",
        ]

        param_configs = [
            ("Launch Speed", "speed_mae", "m/s"),
            ("Launch Angle", "angle_mae", "°"),
            ("Spin Rate", "spin_mae", "RPM"),
        ]
        for param_name, mae_key, unit in param_configs:
            lines.append(f"## Results — {param_name} MAE ({unit})")
            lines.append("| Noise Level | Physics | NN | Winner |")
            lines.append("|-------------|---------|-----|--------|")
            for nl in noise_levels:
                phys_row = next((r for r in results if r.get("estimator") == "PhysicsEstimator" and r.get("noise_level") == nl), None)
                nn_row = next((r for r in results if r.get("estimator") == "NNEstimator" and r.get("noise_level") == nl), None)
                if phys_row and nn_row:
                    pv = phys_row.get(mae_key, float("nan"))
                    nv = nn_row.get(mae_key, float("nan"))
                    winner = "Physics ✓" if pv < nv else "NN ✓"
                    lines.append(f"| {nl} | {pv:.2f} | {nv:.2f} | {winner} |")
            lines.append("")

        lines.append("## Key Insight")
        lines.append(self._generate_key_insight(winner_by_param, physics_wins_at, noise_levels))
        lines.append("")

        lines.append("## Saved Artefacts")
        lines.append(artefacts_md)
        lines.append("")

        lines.append("## Environment")
        lines.append(f"- Python: {env.get('python_version', '?')} | PyTorch: {env.get('torch_version', '?')}")
        lines.append(f"- Git commit: {git.get('commit_hash', '?')} ({git.get('branch', '?')})")

        return "\n".join(lines) + "\n"

    def _generate_key_insight(self, winner_by_param: dict, physics_wins_at, noise_levels: list) -> str:
        nn_wins = [p for p, w in winner_by_param.items() if w == "NNEstimator"]
        phys_wins = [p for p, w in winner_by_param.items() if w == "PhysicsEstimator"]

        if len(nn_wins) == 3:
            insight = "NNEstimator outperforms PhysicsEstimator across all parameters tested."
        elif len(phys_wins) == 3:
            insight = "PhysicsEstimator outperforms NNEstimator across all parameters tested."
        else:
            nn_list = ", ".join(nn_wins) if nn_wins else "none"
            phys_list = ", ".join(phys_wins) if phys_wins else "none"
            insight = f"NNEstimator wins on {nn_list}; PhysicsEstimator wins on {phys_list}."

        if physics_wins_at is None:
            min_noise = min(noise_levels) if noise_levels else None
            if min_noise is not None:
                insight += (
                    f" Physics-based estimation never wins in this evaluation — "
                    f"consider testing at noise_level < {min_noise} to find the crossover point."
                )
        else:
            insight += f" PhysicsEstimator gains an edge at noise_level = {physics_wins_at}."

        return insight
