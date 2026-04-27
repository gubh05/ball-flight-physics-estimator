"""
summarise_runs.py — CLI tool to compare training and benchmark runs.

Usage:
    python summarise_runs.py                           # Show all runs
    python summarise_runs.py --type training           # Show only training runs
    python summarise_runs.py --type benchmark          # Show only benchmark runs
    python summarise_runs.py --top 5 --sort speed_mae  # Top 5 by speed MAE (training)
    python summarise_runs.py --export outputs/runs_summary.md
"""
import argparse
import json
from pathlib import Path

RUNS_DIR = Path("outputs/runs")

# ── Helpers ──────────────────────────────────────────────────────────────────────


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "N/A"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _load_runs(runs_dir: Path):
    """Walk runs_dir and return (training_runs, benchmark_runs) as lists of dicts."""
    training_runs, benchmark_runs = [], []

    if not runs_dir.exists():
        return training_runs, benchmark_runs

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_file = run_dir / "run_config.json"
        metrics_file = run_dir / "metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            print(f"  [warning] Skipping incomplete run: {run_dir.name}")
            continue

        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            run_type = config.get("run_type", "unknown")
            entry = {"config": config, "metrics": metrics}
            if run_type == "training":
                training_runs.append(entry)
            elif run_type == "benchmark":
                benchmark_runs.append(entry)
        except Exception as exc:
            print(f"  [warning] Could not load run {run_dir.name}: {exc}")

    return training_runs, benchmark_runs


def _get_training_row(entry: dict) -> dict:
    config = entry["config"]
    metrics = entry["metrics"]
    run_id = config.get("run_id", "?")
    epochs = config.get("config", {}).get("epochs", "?")
    best_val = metrics.get("best_val_loss")
    best_epoch = metrics.get("best_epoch")
    final_test = metrics.get("final_test", {})
    speed_mae = final_test.get("speed_mae")
    angle_mae = final_test.get("angle_mae")
    spin_mae = final_test.get("spin_mae")
    duration = config.get("duration_seconds")

    best_val_str = f"{best_val:.4f} (ep {best_epoch})" if best_val is not None else "N/A"

    return {
        "run_id":       run_id,
        "epochs":       str(epochs),
        "best_val_loss": best_val_str,
        "speed_mae":    f"{speed_mae:.2f}" if speed_mae is not None else "N/A",
        "angle_mae":    f"{angle_mae:.2f}" if angle_mae is not None else "N/A",
        "spin_mae":     f"{spin_mae:.1f}"  if spin_mae  is not None else "N/A",
        "duration":     _fmt_duration(duration),
        # sort keys (raw floats)
        "_speed_mae":   speed_mae if speed_mae is not None else float("inf"),
        "_angle_mae":   angle_mae if angle_mae is not None else float("inf"),
        "_spin_mae":    spin_mae  if spin_mae  is not None else float("inf"),
    }


def _get_benchmark_row(entry: dict) -> dict:
    config = entry["config"]
    metrics = entry["metrics"]
    run_id = config.get("run_id", "?")
    benchmark = config.get("benchmark", {})
    noise_levels = benchmark.get("noise_levels", [])
    noise_str = ",".join(str(n) for n in noise_levels)
    winner_by_param = metrics.get("winner_by_parameter", {})
    duration = config.get("duration_seconds")

    nn_wins = sum(1 for w in winner_by_param.values() if w == "NNEstimator")
    phys_wins = sum(1 for w in winner_by_param.values() if w == "PhysicsEstimator")
    total = len(winner_by_param)

    if total == 0:
        overall_winner = "N/A"
    elif nn_wins == total:
        overall_winner = "NNEstimator (all params, all noise)"
    elif phys_wins == total:
        overall_winner = "PhysicsEstimator (all params, all noise)"
    elif nn_wins > phys_wins:
        overall_winner = f"NNEstimator ({nn_wins}/{total} params)"
    else:
        overall_winner = f"PhysicsEstimator ({phys_wins}/{total} params)"

    return {
        "run_id":         run_id,
        "noise_levels":   noise_str,
        "overall_winner": overall_winner,
        "duration":       _fmt_duration(duration),
    }


# ── Table rendering ───────────────────────────────────────────────────────────────


def _box_table(headers: list, keys: list, rows: list) -> str:
    """Render a Unicode box-drawing table."""
    col_widths = [
        max(len(h), max((len(str(r.get(k, ""))) for r in rows), default=0))
        for h, k in zip(headers, keys)
    ]

    def _row(values):
        cells = [f" {str(v).center(w)} " for v, w in zip(values, col_widths)]
        return "│" + "│".join(cells) + "│"

    seps = ["─" * (w + 2) for w in col_widths]
    top = "┌" + "┬".join(seps) + "┐"
    mid = "├" + "┼".join(seps) + "┤"
    bot = "└" + "┴".join(seps) + "┘"

    lines = [top, _row(headers), mid]
    for r in rows:
        lines.append(_row([r.get(k, "") for k in keys]))
    lines.append(bot)
    return "\n".join(lines)


# ── Print functions ───────────────────────────────────────────────────────────────


def _print_training(training_runs: list, sort_key: str = None, top: int = None):
    if not training_runs:
        print("  (no training runs found)")
        return

    rows = [_get_training_row(e) for e in training_runs]

    sort_col = f"_{sort_key}" if sort_key else None
    if sort_col and sort_col in rows[0]:
        rows.sort(key=lambda r: r[sort_col])

    if top:
        rows = rows[:top]

    headers = ["Run ID", "Epochs", "Best Val Loss", "Speed MAE", "Angle MAE", "Spin MAE", "Duration"]
    keys    = ["run_id", "epochs", "best_val_loss", "speed_mae", "angle_mae", "spin_mae", "duration"]
    print(_box_table(headers, keys, rows))


def _print_benchmark(benchmark_runs: list, top: int = None):
    if not benchmark_runs:
        print("  (no benchmark runs found)")
        return

    rows = [_get_benchmark_row(e) for e in benchmark_runs]
    if top:
        rows = rows[:top]

    headers = ["Run ID", "Noise Lvls", "Overall Winner", "Duration"]
    keys    = ["run_id", "noise_levels", "overall_winner", "duration"]
    print(_box_table(headers, keys, rows))


# ── Export ────────────────────────────────────────────────────────────────────────


def _export_markdown(training_runs: list, benchmark_runs: list, path: Path):
    lines = ["# Experiment Runs Summary", ""]

    lines += ["## Training Runs", ""]
    if not training_runs:
        lines.append("_No training runs found._")
    else:
        lines.append("| Run ID | Epochs | Best Val Loss | Speed MAE | Angle MAE | Spin MAE | Duration |")
        lines.append("|--------|--------|---------------|-----------|-----------|----------|----------|")
        for e in training_runs:
            r = _get_training_row(e)
            lines.append(
                f"| {r['run_id']} | {r['epochs']} | {r['best_val_loss']} "
                f"| {r['speed_mae']} | {r['angle_mae']} | {r['spin_mae']} | {r['duration']} |"
            )
    lines.append("")

    lines += ["## Benchmark Runs", ""]
    if not benchmark_runs:
        lines.append("_No benchmark runs found._")
    else:
        lines.append("| Run ID | Noise Levels | Overall Winner | Duration |")
        lines.append("|--------|-------------|----------------|----------|")
        for e in benchmark_runs:
            r = _get_benchmark_row(e)
            lines.append(f"| {r['run_id']} | {r['noise_levels']} | {r['overall_winner']} | {r['duration']} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary exported to: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Summarise experiment runs from outputs/runs/")
    parser.add_argument("--type", choices=["training", "benchmark"], help="Filter by run type")
    parser.add_argument("--top", type=int, help="Show top N runs")
    parser.add_argument("--sort", metavar="METRIC", help="Sort training runs by metric (e.g. speed_mae)")
    parser.add_argument("--export", type=Path, metavar="PATH", help="Export summary to Markdown file")
    args = parser.parse_args()

    training_runs, benchmark_runs = _load_runs(RUNS_DIR)

    if not training_runs and not benchmark_runs:
        print("No completed runs found in outputs/runs/")
        return

    if args.export:
        _export_markdown(training_runs, benchmark_runs, args.export)
        return

    if args.type != "benchmark":
        print("\nTRAINING RUNS")
        _print_training(training_runs, sort_key=args.sort, top=args.top)

    if args.type != "training":
        print("\nBENCHMARK RUNS")
        _print_benchmark(benchmark_runs, top=args.top)


if __name__ == "__main__":
    main()
