"""The engine-independent benchmark truth: MuJoCo C float64 forward truth of every corpus state, MuJoCo C-only state
labels and groups, the model check against native MuJoCo Warp 3.17.0, and the finite-difference gradient truth of
seeded samples (mjc.py: the MuJoCo C machinery and its definitions; store.py: production and the truth file format;
mjw_labels.py: the native MuJoCo Warp run; cli.py: python -m benchmarks.bench truth)."""
