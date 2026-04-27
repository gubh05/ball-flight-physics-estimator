# Build Context — Ball Flight Physics Estimator (Improvement Phase)

Source plan: `IMPROVEMENT_project2_ball_physics_estimator.md`
Initial build: Steps 1–14 per `SPEC_project2_ball_physics_estimator.md`
This phase starts at **Step 15**.

---

## Scope (agreed plan)

### Included
1. **Config additions** — `partial_traj_min`, `partial_traj_max`, `mc_dropout_passes`
2. **Partial trajectories** — randomly truncate IMU signal in `DatasetGenerator.generate()`
3. **MC Dropout uncertainty** — `predict_with_uncertainty()` on `NNEstimator`
4. **Uncertainty plot** — `plot_uncertainty_bands()` in `visualization.py`
5. **Wire up** — `run_pipeline.py` uses new features; README rewritten

### Excluded (deliberately cut)
- Optimization-based `estimate_optimized()` on PhysicsEstimator (complex, slow, low ROI)
- `plot_trajectory_comparison()` (fiddly, low value vs. effort)

---

## Step Status

- [x] Step 15 — Config: add `partial_traj_min`, `partial_traj_max`, `mc_dropout_passes`
- [x] Step 16 — DatasetGenerator: partial trajectory truncation in `generate()`
- [x] Step 17 — NNEstimator: add `predict_with_uncertainty()` (MC Dropout)
- [x] Step 18 — Visualization: add `plot_uncertainty_bands()`
- [ ] Step 19 — Wire up `run_pipeline.py` + rewrite README

---

## Commit Messages (this phase)

- Step 15: `Step 15: add partial_traj_min/max and mc_dropout_passes to Config`
- Step 16: `Step 16: add partial trajectory truncation to DatasetGenerator`
- Step 17: `Step 17: add predict_with_uncertainty() (MC Dropout) to NNEstimator`
- Step 18: `Step 18: add plot_uncertainty_bands() to visualization`
- Step 19: `Step 19: wire uncertainty into run_pipeline.py and rewrite README`
