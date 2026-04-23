# Ball Flight Physics Estimator

> **Physics-informed pipeline that simulates football shot trajectories, generates synthetic IMU sensor data, and trains a neural network to invert the physics model — estimating launch speed, angle, and spin from raw sensor readings.**

---

## Project Summary

Most ML engineers treat sensor data as black-box inputs. This project demonstrates the rare ability to reason from physics first — understanding *why* sensors read what they read, then using that understanding to build better models.

The pipeline:
1. **Simulates** realistic football trajectories using a forward physics model (projectile motion + Magnus effect via RK4 integration)
2. **Generates** synthetic IMU readings with realistic MEMS noise characteristics (Gaussian white noise, per-shot bias, quantization)
3. **Trains** a 1D-CNN neural network to invert the physics: estimating shot parameters from raw sensor sequences
4. **Benchmarks** the ML model against a pure physics-based analytical estimator to reveal where each approach wins

---

## Physics Primer

### Projectile Motion
```
x(t) = v₀·cos(θ)·t
y(t) = v₀·sin(θ)·t - ½·g·t²
```

### Magnus Effect (spin-induced lift)
A spinning ball curves because spin creates differential air pressure — the *Magnus effect*:
```
F_Magnus = ½ · ρ · A · C_L · |v|² · (spin_unit × velocity_unit)
```
Where ρ = 1.225 kg/m³, A = π·r² (r=0.11m), C_L = 0.25 (football empirical).

Backspin (positive RPM) → upward lift → longer range.  
Topspin (negative RPM) → downward force → shorter range.

### Drag Force
```
F_drag = -½ · ρ · A · C_D · |v| · v       (C_D ≈ 0.47 for sphere)
```

### IMU Reading Model
The accelerometer measures *specific force* (aerodynamic forces only, gravity cancelled in free-fall):
```
a_sensor(t) = (F_Magnus + F_drag) / m
```
The gyroscope measures the constant spin rate in rad/s.

---

## Pipeline Diagram

```
┌─────────────────────────┐
│   BallPhysicsModel      │  RK4 forward simulation
│  (launch params → traj) │  drag + Magnus + gravity
└──────────┬──────────────┘
           │ TrajectoryResult
           ▼
┌─────────────────────────┐
│  SensorNoiseSimulator   │  MEMS IMU noise model
│  (traj → IMU readings)  │  white noise + bias + quantization
└──────────┬──────────────┘
           │ IMUReading (acc_x, acc_y, gyro_z)
           ▼
┌─────────────────────────┐
│   DatasetGenerator      │  10,000 shots, zero-padded to 300 timesteps
│   (IMU → (X, y) pairs)  │  normalised labels [0,1]
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ Physics │  │ NNEstimator  │  1D-CNN regression (PyTorch)
│Estimator│  │  (trained)   │  Conv × 3 → AdaptiveAvgPool → MLP
└────┬────┘  └──────┬───────┘
     └──────┬───────┘
            ▼
     ┌─────────────┐
     │  Benchmarker │  MAE + RMSE per param per noise level
     └─────────────┘
```

---

## Setup

```bash
# Create environment (Python 3.11, PyTorch with CUDA)
conda env create -f environment.yml
conda activate IOTIS-P

# Or install manually
conda create -n IOTIS-P python=3.11 -y
conda activate IOTIS-P
pip install -r requirements.txt
```

---

## How to Run

```bash
# Full end-to-end pipeline (generates data, trains NN, benchmarks, saves plots)
python run_pipeline.py

# Run all tests
pytest tests/ -v

# Explore notebooks
jupyter notebook notebooks/
```

The pipeline:
- Caches the 10,000-sample dataset to `outputs/dataset_X.npy` / `dataset_y.npy`
- Saves the best NN checkpoint to `outputs/nn_estimator_best.pt`
- Saves all plots to `outputs/plots/`

---

## Results Table

Benchmark on 200 held-out test trajectories at three noise levels (σ multiplier):

| Estimator | Noise σ | Speed MAE (m/s) | Angle MAE (°) | Spin MAE (rpm) | Speed RMSE | Angle RMSE | Spin RMSE |
|-----------|---------|----------------|--------------|---------------|-----------|-----------|---------|
| PhysicsEstimator | 0.5 | 3.493 | 20.300 | **0.020** | 3.881 | 22.220 | 0.025 |
| NNEstimator      | 0.5 | **0.562** | **1.245** | 79.168 | **0.726** | **2.989** | 109.452 |
| PhysicsEstimator | 1.0 | 3.494 | 20.333 | **0.038** | 3.881 | 22.284 | 0.048 |
| NNEstimator      | 1.0 | **0.554** | **1.263** | 79.120 | **0.723** | **3.034** | 109.315 |
| PhysicsEstimator | 2.0 | 3.495 | 20.293 | **0.076** | 3.884 | 22.218 | 0.095 |
| NNEstimator      | 2.0 | **0.565** | **1.266** | 79.245 | **0.728** | **3.160** | 109.493 |

---

## Key Insight

**Neither estimator dominates universally — and that's the point.**

The **PhysicsEstimator** wins decisively on *spin estimation* (MAE < 0.1 rpm across all noise levels). This is unsurprising: the gyroscope directly measures spin rate, and the physics estimator simply converts rad/s → rpm. No ML required.

The **NNEstimator** wins on *launch speed and angle* by a factor of 6–16×. Speed and angle estimation requires integrating information across the full IMU time-series — the shape of the acceleration curve, its duration, the peak magnitude — a task where learned temporal patterns beat simple analytical heuristics.

Crucially, the NN's advantage on speed/angle is *robust to noise*: MAE barely changes from σ=0.5 to σ=2.0, suggesting the 1D-CNN learned to denoise the signal implicitly during training.

**Takeaway for hardware companies:** When you know what a sensor measures directly (gyro → spin), use physics. When the relationship is indirect and non-linear (acceleration sequence → launch params), use ML. A production system would use both.

---

## Example Trajectory Plot

Generated by `run_pipeline.py` — 3×3 grid showing speed × spin interaction (Magnus effect):

![Trajectory Family](outputs/plots/trajectory_family.png)

---

## Project Structure

```
ball-flight-physics-estimator/
├── src/
│   ├── physics/ball_physics_model.py      # Forward simulation (RK4)
│   ├── sensors/sensor_noise_simulator.py  # IMU noise model
│   ├── data/dataset_generator.py          # Synthetic dataset builder
│   ├── models/
│   │   ├── physics_estimator.py           # Analytical inverse estimator
│   │   └── nn_estimator.py                # 1D-CNN (PyTorch)
│   ├── evaluation/benchmarker.py          # Head-to-head comparison
│   └── utils/
│       ├── config.py                      # Central config dataclass
│       └── visualization.py              # 6 plot functions
├── tests/                                 # 53 pytest tests (all passing)
├── notebooks/
│   ├── 01_physics_exploration.ipynb
│   ├── 02_sensor_data_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── run_pipeline.py                        # End-to-end entry point
├── requirements.txt
└── environment.yml
```
