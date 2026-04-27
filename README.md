# Ball Flight Physics Estimator

> **An inverse problem in sports physics: estimate launch speed, angle, and spin from partial, noisy IMU observations — using a physics-based analytical estimator, a 1D-CNN, and MC Dropout uncertainty quantification.**

---

## Problem Statement

A football is kicked. An IMU sensor records acceleration and gyroscope data — but only for a fraction of the flight, and with realistic MEMS noise. Can we recover the initial launch parameters?

This is a classic **inverse problem**: the forward model (physics → sensor readings) is well understood; the inverse (sensor readings → physics) is non-trivial because:

- The signal is **partial** — we observe a random 30–100% window of the trajectory
- The signal is **noisy** — additive white noise, per-shot sensor bias, ADC quantization
- The relationship between acceleration shape and launch parameters is **non-linear**

The project pits a pure physics-based analytical estimator against a trained neural network, and adds MC Dropout to quantify prediction uncertainty.

---

## Methods

### 1. Forward Simulation (`src/physics/ball_physics_model.py`)

RK4 integration of projectile motion with drag and Magnus effect:

```
F_drag   = -½ · ρ · A · C_D · |v| · v          (C_D = 0.47)
F_Magnus =  ½ · ρ · A · C_L · |v|² · spin_dir  (C_L = 0.25)
```

The IMU reads aerodynamic specific force only (gravity cancelled in free-fall):
```
a_sensor(t) = (F_drag + F_Magnus) / m
gyro_z(t)   = spin_rad_s  (constant)
```

### 2. Synthetic Dataset (`src/data/dataset_generator.py`)

10,000 shots sampled uniformly over:
- Speed: 5–35 m/s · Angle: 5–60° · Spin: −3000 to +3000 rpm
- Noise level: randomly drawn from {0.5, 1.0, 2.0}σ per shot
- **Partial trajectory**: each sample reveals a random 30–100% window of the IMU signal; the remainder is zero-padded. This forces models to infer launch parameters from incomplete observations.

### 3. Physics-Based Estimator (`src/models/physics_estimator.py`)

Analytical inverse using the launch-window signal:
- **Spin**: gyroscope mean → convert rad/s to rpm directly
- **Angle**: atan2 of initial acceleration direction (drag opposes velocity)
- **Speed**: peak acceleration magnitude back-calculated via drag + Magnus constants

Simple, fast, interpretable. Sets the baseline.

### 4. Neural Network Estimator (`src/models/nn_estimator.py`)

1D-CNN regression on the full padded IMU sequence (300 timesteps × 3 channels):

```
Conv1d(3→32, k=7) → BN → ReLU → MaxPool
Conv1d(32→64, k=5) → BN → ReLU → MaxPool
Conv1d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool
Flatten → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→3)
```

Trained with Adam + ReduceLROnPlateau, MSE loss on normalised targets.

### 5. Uncertainty Estimation — MC Dropout (`src/models/nn_estimator.py`)

At inference, dropout stays **active** across `N` stochastic forward passes. The spread of predictions across passes gives a per-sample uncertainty estimate:

```
mean, std = nn_estimator.predict_with_uncertainty(X, n_passes=30)
```

High `std` → model is uncertain (e.g. very short trajectory window or high noise). This is useful for flagging unreliable predictions in a production system.

---

## Pipeline

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
           │ partial window (30–100% of trajectory)
           ▼
┌─────────────────────────┐
│   DatasetGenerator      │  10,000 shots, zero-padded to 300 timesteps
│   (IMU → (X, y) pairs)  │  normalised labels [0,1], train/val/test split
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────────┐
│ Physics │  │ NNEstimator (1D-CNN) │
│Estimator│  │ + MC Dropout (σ est) │
└────┬────┘  └──────────┬───────────┘
     └──────┬───────────┘
            ▼
     ┌─────────────┐
     │  Benchmarker │  MAE + RMSE per param per noise level
     └─────────────┘
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate IOTIS-P

# Or manually
conda create -n IOTIS-P python=3.11 -y
conda activate IOTIS-P
pip install -r requirements.txt
```

---

## How to Run

```bash
# Full end-to-end pipeline
python run_pipeline.py

# Run all tests (53 passing)
pytest tests/ -v

# Explore notebooks
jupyter notebook notebooks/
```

To regenerate the dataset with partial trajectories (or after changing `Config`):
```bash
rm outputs/dataset_X.npy outputs/dataset_y.npy outputs/nn_estimator_best.pt
python run_pipeline.py
```

---

## Results

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

## Key Insights

**Neither estimator dominates — and that's the point.**

The **PhysicsEstimator** wins decisively on spin (MAE < 0.1 rpm). The gyroscope directly measures spin rate; no ML is needed. Converting rad/s → rpm is exact physics.

The **NNEstimator** wins on speed and angle by 6–16×. These require integrating information across the full time-series — peak magnitude, curve shape, duration — a task where learned temporal patterns beat analytical heuristics. Critically, the NN's advantage is robust across noise levels (MAE barely changes from σ=0.5 to σ=2.0), suggesting the 1D-CNN learned implicit denoising.

**MC Dropout** provides calibrated uncertainty per prediction. Short trajectory windows and high noise inflate `std`, making it practical to flag low-confidence estimates before acting on them.

**Takeaway:** When a sensor directly measures what you need, use physics. When the relationship is indirect and non-linear, use ML. A production system would combine both — and know when to say "I'm not sure."

---

## Project Structure

```
ball-flight-physics-estimator/
├── src/
│   ├── physics/ball_physics_model.py      # Forward simulation (RK4)
│   ├── sensors/sensor_noise_simulator.py  # IMU noise model
│   ├── data/dataset_generator.py          # Synthetic dataset + partial trajectories
│   ├── models/
│   │   ├── physics_estimator.py           # Analytical inverse estimator
│   │   └── nn_estimator.py                # 1D-CNN + MC Dropout uncertainty
│   ├── evaluation/benchmarker.py          # Head-to-head comparison
│   └── utils/
│       ├── config.py                      # Central config dataclass
│       ├── visualization.py               # Plot functions incl. uncertainty bands
│       └── experiment_logger.py           # Run tracking and artefact logging
├── tests/                                 # 53 pytest tests (all passing)
├── notebooks/
│   ├── 01_physics_exploration.ipynb
│   ├── 02_sensor_data_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── run_pipeline.py                        # End-to-end entry point
├── BUILD_CONTEXT_project2_ball_physics_estimator.md
├── requirements.txt
└── environment.yml
```
