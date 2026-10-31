# Accuracy on the held-out corpora (accuracy platform: Tesla V100-SXM2-16GB)

Every number below is in `docs/results/paper_accuracy_heldout_v100.json` (generated 2026-10-29T11:46:03Z by `python -m benchmarks.bench.report.paper_accuracy` from the benchmark harness's accuracy jobs). GPU Tesla V100-SXM2-16GB (driver 550.90.07), CPU Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz. The V100 is the accuracy platform only: no timing here.

Corpora: the four held-out benchmark corpora mf_s0, mf_s1, mf_s2, mf_s3 (4096 states each, `benchmarks/data/`); one control step = 4 x 5 ms substeps of the mjlab G1 with ctrl held, Newton, per-world foot friction and torso COM offset. Truth: MuJoCo C 3.17.0 float64, 2000 iterations, tolerance 1e-13, ls 100 (`benchmarks/data/truth/`). Every engine runs natively (its own package, unmodified, default options; MuJoCo Warp keeps its internal tolerance floor of 1e-6). Genesis 1.4.5 runs the same model with its 502 contact pairs declared through contype / conaffinity (Genesis's MJCF importer reads no <pair>); its deviations from the problem are listed in the notes. Pooled rows concatenate the four corpora.

## 1. Forward: one control step against the float64 truth

qvel: relative error per state; qpos: tangent-space error without the base translation (mj_differentiatePos, free-joint rotation and hinges, rad, absolute). Statistics over the finite states, non-finite states counted. Model-ok: MuJoCo C and native MuJoCo Warp 3.17.0 see the same contacts, zones, limit rows and frames. All 16384 states (4 x 4096) for every engine, MJX included (no subset was needed).

| source | it | all: n | qvel median | p90 | p99 | max | > 1e-3 | non-finite | qpos (rad) median | p99 | max | base (m) median | p99 | model-ok: n | qvel median | p90 | p99 | max | > 1e-3 | qpos (rad) p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours (forward of both gradient modes) | 10 | 16384 | 8.5e-07 | 2.1e-06 | 1.2e-05 | 1.9e-01 | 0.57% | 0 | 1.8e-07 | 2.0e-06 | 2.6e-02 | 2.8e-08 | 9.4e-08 | 16237 | 8.5e-07 | 2.0e-06 | 5.1e-06 | 1.9e-01 | 0.06% | 6.0e-07 |
| MuJoCo Warp 3.17.0 (native) | 10 | 16384 | 1.3e-06 | 3.4e-06 | 1.7e-05 | 1.7e-01 | 0.60% | 0 | 2.4e-07 | 1.9e-06 | 2.6e-02 | 2.8e-08 | 9.8e-08 | 16237 | 1.3e-06 | 3.3e-06 | 9.5e-06 | 2.6e-03 | 0.02% | 1.1e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 10 | 16384 | 1.3e-06 | 3.5e-06 | 1.9e-05 | 1.7e-01 | 0.60% | 0 | 2.4e-07 | 2.6e-06 | 2.6e-02 | 2.8e-08 | 9.8e-08 | 16237 | 1.3e-06 | 3.4e-06 | 1.0e-05 | 4.4e-03 | 0.02% | 1.3e-06 |
| MuJoCo Warp + PR #1535 @4af093f | 10 | 16384 | 1.3e-06 | 3.5e-06 | 1.8e-05 | 1.7e-01 | 0.59% | 0 | 2.4e-07 | 2.6e-06 | 2.6e-02 | 2.8e-08 | 9.8e-08 | 16237 | 1.3e-06 | 3.3e-06 | 1.0e-05 | 4.4e-03 | 0.01% | 1.3e-06 |
| MJX 3.17.0 (float32, native) | 10 | 16384 | 2.9e-06 | 7.1e-05 | 1.6e-03 | 6.0e-01 | 1.43% | 0 | 4.1e-07 | 1.7e-04 | 5.6e-02 | 3.5e-08 | 1.5e-06 | 16237 | 2.9e-06 | 6.0e-05 | 9.1e-04 | 6.0e-01 | 0.86% | 1.0e-04 |
| Genesis 1.4.5 (float32, native) | 10 | 16384 | 5.5e-02 | 2.3e-01 | 5.0e-01 | 1.0e+00 | 92.82% | 0 | 8.1e-03 | 6.6e-02 | 2.5e-01 | 1.0e-04 | 1.4e-03 | 16237 | 5.5e-02 | 2.3e-01 | 5.0e-01 | 1.0e+00 | 92.78% | 6.6e-02 |
| MuJoCo C float64, 10 it (solver-truncation floor) | 10 | 16384 | 0 | 0 | 3.7e-06 | 3.1e-01 | 0.06% | 0 | 8.2e-18 | 5.5e-07 | 2.3e-02 | 0 | 4.1e-09 | 16237 | 0 | 0 | 3.0e-06 | 3.1e-01 | 0.06% | 3.8e-07 |
| floor: MuJoCo Warp 3.17.0 run to run (mf_s0, mf_s1; process 1, 2 vs 0) | 10 | 16384 | 3.1e-07 | 1.1e-06 | 3.3e-06 | 6.5e-05 | 0 | 0 | 6.6e-08 | 4.6e-07 | 8.0e-06 | 0 | 6.0e-08 | 16234 | 3.1e-07 | 1.1e-06 | 3.3e-06 | 6.5e-05 | 0 | 4.6e-07 |
| ours (forward of both gradient modes) | 50 | 16384 | 8.5e-07 | 2.0e-06 | 1.0e-05 | 1.9e-01 | 0.56% | 0 | 1.8e-07 | 1.7e-06 | 2.6e-02 | 2.8e-08 | 9.3e-08 | 16237 | 8.5e-07 | 2.0e-06 | 4.9e-06 | 1.9e-01 | 0.04% | 5.8e-07 |
| MuJoCo Warp 3.17.0 (native) | 50 | 16384 | 1.3e-06 | 3.4e-06 | 1.5e-05 | 1.7e-01 | 0.57% | 0 | 2.4e-07 | 1.7e-06 | 2.6e-02 | 2.8e-08 | 9.5e-08 | 16237 | 1.3e-06 | 3.3e-06 | 9.0e-06 | 4.4e-05 | 0 | 1.1e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 50 | 16384 | 1.3e-06 | 3.4e-06 | 1.7e-05 | 1.7e-01 | 0.58% | 0 | 2.4e-07 | 2.2e-06 | 2.6e-02 | 2.8e-08 | 9.5e-08 | 16237 | 1.3e-06 | 3.3e-06 | 9.9e-06 | 8.1e-05 | 0 | 1.2e-06 |
| MuJoCo Warp + PR #1535 @4af093f | 50 | 16384 | 1.3e-06 | 3.5e-06 | 1.6e-05 | 1.7e-01 | 0.58% | 0 | 2.4e-07 | 2.2e-06 | 2.6e-02 | 2.8e-08 | 9.6e-08 | 16237 | 1.3e-06 | 3.4e-06 | 9.8e-06 | 3.9e-03 | 0.01% | 1.3e-06 |
| MJX 3.17.0 (float32, native) | 50 | 16384 | 2.9e-06 | 7.0e-05 | 1.5e-03 | 6.0e-01 | 1.37% | 0 | 4.1e-07 | 1.6e-04 | 5.6e-02 | 3.5e-08 | 1.5e-06 | 16237 | 2.9e-06 | 5.9e-05 | 8.8e-04 | 6.0e-01 | 0.80% | 1.0e-04 |
| Genesis 1.4.5 (float32, native) | 50 | 16384 | 5.5e-02 | 2.3e-01 | 5.0e-01 | 1.0e+00 | 92.82% | 0 | 8.1e-03 | 6.6e-02 | 2.5e-01 | 1.0e-04 | 1.4e-03 | 16237 | 5.5e-02 | 2.3e-01 | 5.0e-01 | 1.0e+00 | 92.78% | 6.6e-02 |
| MuJoCo C float64, 50 it (solver-truncation floor) | 50 | 16384 | 0 | 0 | 2.8e-06 | 9.7e-05 | 0 | 0 | 8.0e-18 | 3.4e-07 | 9.5e-06 | 0 | 2.9e-09 | 16237 | 0 | 0 | 1.9e-06 | 9.7e-05 | 0 | 2.7e-07 |

Per group (model-ok states of each group; qvel relative error median / p99, qpos rad p99):

| source | it | G0 n | G0 qvel median | G0 p99 | G0 qpos p99 | G4 n | G4 qvel median | G4 p99 | G4 qpos p99 | G5 n | G5 qvel median | G5 p99 | G5 qpos p99 | G6 n | G6 qvel median | G6 p99 | G6 qpos p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours (forward of both gradient modes) | 10 | 13480 | 8.6e-07 | 4.7e-06 | 5.4e-07 | 476 | 4.6e-07 | 6.6e-06 | 1.8e-06 | 5645 | 8.4e-07 | 5.5e-06 | 6.5e-07 | 5668 | 9.4e-07 | 6.9e-06 | 1.0e-06 |
| MuJoCo Warp 3.17.0 (native) | 10 | 13480 | 1.3e-06 | 8.8e-06 | 1.1e-06 | 476 | 6.3e-07 | 7.2e-06 | 1.5e-06 | 5645 | 1.2e-06 | 6.8e-06 | 9.2e-07 | 5668 | 1.3e-06 | 8.8e-06 | 9.9e-07 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 10 | 13480 | 1.3e-06 | 9.3e-06 | 1.2e-06 | 476 | 6.4e-07 | 1.0e-05 | 3.4e-06 | 5645 | 1.2e-06 | 8.6e-06 | 1.2e-06 | 5668 | 1.4e-06 | 1.1e-05 | 1.4e-06 |
| MuJoCo Warp + PR #1535 @4af093f | 10 | 13480 | 1.3e-06 | 9.5e-06 | 1.2e-06 | 476 | 6.5e-07 | 1.0e-05 | 2.6e-06 | 5645 | 1.2e-06 | 8.5e-06 | 1.1e-06 | 5668 | 1.4e-06 | 1.1e-05 | 1.4e-06 |
| MJX 3.17.0 (float32, native) | 10 | 13480 | 2.7e-06 | 6.3e-04 | 7.2e-05 | 476 | 1.3e-06 | 4.2e-04 | 1.1e-04 | 5645 | 2.6e-06 | 5.1e-04 | 6.1e-05 | 5668 | 5.5e-06 | 6.6e-04 | 7.3e-05 |
| Genesis 1.4.5 (float32, native) | 10 | 13480 | 4.9e-02 | 4.8e-01 | 6.0e-02 | 476 | 5.7e-02 | 5.4e-01 | 1.4e-01 | 5645 | 4.9e-02 | 4.8e-01 | 6.1e-02 | 5668 | 1.0e-01 | 5.1e-01 | 7.1e-02 |
| MuJoCo C float64, 10 it (solver-truncation floor) | 10 | 13480 | 0 | 1.7e-15 | 2.6e-16 | 476 | 0 | 1.9e-15 | 5.8e-16 | 5645 | 0 | 1.5e-15 | 2.4e-16 | 5668 | 0 | 1.7e-15 | 2.6e-16 |
| floor: MuJoCo Warp 3.17.0 run to run (mf_s0, mf_s1; process 1, 2 vs 0) | 10 | 13450 | 3.1e-07 | 3.2e-06 | 4.3e-07 | 456 | 3.0e-07 | 2.2e-06 | 8.0e-07 | 5762 | 3.4e-07 | 3.3e-06 | 3.8e-07 | 5666 | 3.4e-07 | 3.5e-06 | 3.9e-07 |
| ours (forward of both gradient modes) | 50 | 13480 | 8.6e-07 | 4.6e-06 | 5.3e-07 | 476 | 4.6e-07 | 6.5e-06 | 1.5e-06 | 5645 | 8.4e-07 | 5.5e-06 | 6.5e-07 | 5668 | 9.4e-07 | 6.8e-06 | 1.0e-06 |
| MuJoCo Warp 3.17.0 (native) | 50 | 13480 | 1.3e-06 | 8.5e-06 | 1.1e-06 | 476 | 6.2e-07 | 5.7e-06 | 1.4e-06 | 5645 | 1.2e-06 | 6.9e-06 | 9.3e-07 | 5668 | 1.3e-06 | 8.3e-06 | 9.8e-07 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 50 | 13480 | 1.3e-06 | 9.4e-06 | 1.1e-06 | 476 | 6.6e-07 | 9.1e-06 | 2.1e-06 | 5645 | 1.2e-06 | 8.6e-06 | 1.1e-06 | 5668 | 1.4e-06 | 1.0e-05 | 1.4e-06 |
| MuJoCo Warp + PR #1535 @4af093f | 50 | 13480 | 1.3e-06 | 9.3e-06 | 1.2e-06 | 476 | 6.3e-07 | 9.2e-06 | 2.5e-06 | 5645 | 1.2e-06 | 8.5e-06 | 1.1e-06 | 5668 | 1.4e-06 | 1.0e-05 | 1.4e-06 |
| MJX 3.17.0 (float32, native) | 50 | 13480 | 2.7e-06 | 6.1e-04 | 7.1e-05 | 476 | 1.3e-06 | 3.1e-04 | 1.0e-04 | 5645 | 2.6e-06 | 5.1e-04 | 6.1e-05 | 5668 | 5.5e-06 | 6.5e-04 | 7.1e-05 |
| Genesis 1.4.5 (float32, native) | 50 | 13480 | 4.9e-02 | 4.8e-01 | 6.0e-02 | 476 | 5.7e-02 | 5.4e-01 | 1.4e-01 | 5645 | 4.9e-02 | 4.8e-01 | 6.1e-02 | 5668 | 1.0e-01 | 5.1e-01 | 7.1e-02 |
| MuJoCo C float64, 50 it (solver-truncation floor) | 50 | 13480 | 0 | 1.7e-15 | 2.5e-16 | 476 | 0 | 1.7e-15 | 5.4e-16 | 5645 | 0 | 1.5e-15 | 2.4e-16 | 5668 | 0 | 1.7e-15 | 2.5e-16 |

## 2. Gradients against the float64 finite-difference truth

VJPs of each corpus's 3 stored output cotangents at the truth's sampled states (up to 64 per group and corpus, all states of the group, not only model-ok ones), 10 iterations. Error per input block (qpos tangent / qvel / ctrl) on the valid columns: |g - g_fd| / |g_fd|, floored at the truth's own floor |fd(1e-7) - fd(1e-6)| / |fd(1e-6)| (no source can look closer than the truth is known); median and p99 over (state, cotangent) pairs, and the smallest cosine. n = usable states.

### G0 (pooled over the four corpora)

| source | n | qpos median | p99 | min cos | qvel median | p99 | min cos | ctrl median | p99 | min cos | model-ok n | qpos / qvel / ctrl median (model-ok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 256 | 3.6e-05 | 4.2e-04 | 0.9979 | 1.0e-05 | 1.6e-04 | 0.9892 | 1.9e-06 | 2.3e-05 | 0.9990 | 254 | 3.6e-05 / 9.9e-06 / 1.9e-06 |
| ours, interface defaults (clamp_kv=0, normal=3) | 256 | 3.6e-05 | 4.2e-04 | 0.9979 | 1.0e-05 | 1.6e-04 | 0.9892 | 1.9e-06 | 2.3e-05 | 0.9990 | 254 | 3.6e-05 / 9.9e-06 / 1.9e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 256 | 2.9e-01 | 3.5e+00 | -0.7854 | 3.4e-01 | 3.2e+00 | -0.7562 | 1.5e-02 | 1.9e-01 | 0.9767 | 254 | 2.9e-01 / 3.3e-01 / 1.4e-02 |
| MuJoCo Warp + PR #1535 @4af093f | 256 | 5.3e-05 | 8.7e-04 | 0.9979 | 1.0e-05 | 1.7e-04 | 0.9892 | 2.1e-06 | 1.4e-05 | 0.9990 | 254 | 5.2e-05 / 1.0e-05 / 2.1e-06 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 256 | 9.6e-05 | 4.3e-01 | -0.4366 | 2.1e-05 | 1.7e-01 | 0.8481 | 2.9e-06 | 6.3e-02 | 0.9930 | 254 | 9.4e-05 / 2.1e-05 / 2.9e-06 |
| Genesis 1.4.5, native reverse mode (initial qvel only) | 256 | not supported |  |  | 2.3e-01 | 1.7e+00 | 0.0282 | not supported |  |  | 254 | not supported / 2.3e-01 / not supported |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 256 | 6.7e-04 | 9.4e-03 | 0.9985 | 1.5e-04 | 3.3e-03 | 0.9999 | 7.8e-06 | 2.5e-04 | 1.0000 | 254 | 6.7e-04 / 1.5e-04 / 7.9e-06 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 256 | 2.3e-07 | 6.5e-03 | 0.9953 | 1.8e-07 | 5.4e-05 | 0.9999 | 2.2e-08 | 1.6e-06 | 1.0000 | 254 | 2.3e-07 / 1.8e-07 / 2.2e-08 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 256 | 6.6e-06 | 1.0e-04 | 0.9995 | 1.8e-06 | 2.0e-05 | 0.9999 | 5.9e-07 | 4.7e-06 | 1.0000 | 254 | 6.6e-06 / 1.8e-06 / 5.9e-07 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 256 | 1.8e-07 | 3.2e-05 | 0.9995 | 1.6e-07 | 6.4e-07 | 0.9999 | 3.0e-08 | 1.7e-07 | 1.0000 | 254 | 1.9e-07 / 1.6e-07 / 3.0e-08 |
| MuJoCo Warp 3.17.0 (native) | - | no backward (MuJoCo Warp has no gradient) | | | | | | | | | | |

### G4 (pooled over the four corpora)

| source | n | qpos median | p99 | min cos | qvel median | p99 | min cos | ctrl median | p99 | min cos | model-ok n | qpos / qvel / ctrl median (model-ok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 256 | 4.3e-05 | 2.7e-03 | 0.9746 | 1.3e-05 | 1.6e-02 | 0.9839 | 2.0e-06 | 2.0e-03 | 0.9995 | 254 | 4.3e-05 / 1.2e-05 / 1.9e-06 |
| ours, interface defaults (clamp_kv=0, normal=3) | 256 | not comparable: the truth has no clamp_kv=0 Jacobian | | | | | | | | | | |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 256 | 5.0e-01 | 3.2e+00 | -0.8966 | 4.8e-01 | 2.9e+00 | -0.8286 | 8.9e-02 | 5.9e-01 | 0.7238 | 254 | 5.0e-01 / 4.8e-01 / 8.8e-02 |
| MuJoCo Warp + PR #1535 @4af093f | 256 | not comparable: the truth has no clamp_kv=0 Jacobian | | | | | | | | | | |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 256 | 1.2e-04 | 5.4e-01 | -0.8250 | 2.3e-05 | 2.0e-01 | 0.9005 | 3.0e-06 | 4.5e-02 | 0.9953 | 254 | 1.1e-04 / 2.3e-05 / 2.9e-06 |
| Genesis 1.4.5, native reverse mode (initial qvel only) | 256 | not supported |  |  | 3.4e-01 | 2.2e+00 | -0.2922 | not supported |  |  | 254 | not supported / 3.4e-01 / not supported |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 256 | 7.1e-04 | 7.6e-03 | 0.6882 | 1.8e-04 | 2.6e-03 | 0.9637 | 7.2e-06 | 1.9e-04 | 0.9957 | 254 | 7.1e-04 / 1.8e-04 / 7.2e-06 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 256 | 2.7e-07 | 8.9e-03 | 0.7775 | 2.2e-07 | 2.2e-03 | 0.9663 | 2.3e-08 | 6.1e-05 | 0.9957 | 254 | 2.7e-07 / 2.2e-07 / 2.3e-08 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 256 | 7.8e-06 | 8.9e-05 | 0.8036 | 2.4e-06 | 2.8e-05 | 0.9662 | 7.1e-07 | 7.3e-06 | 0.9957 | 254 | 7.8e-06 / 2.4e-06 / 7.1e-07 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 256 | 2.2e-07 | 4.1e-05 | -0.0538 | 2.0e-07 | 1.1e-05 | 0.9110 | 3.8e-08 | 1.3e-06 | 0.9956 | 254 | 2.2e-07 / 2.0e-07 / 3.8e-08 |
| MuJoCo Warp 3.17.0 (native) | - | no backward (MuJoCo Warp has no gradient) | | | | | | | | | | |

### G5 (pooled over the four corpora)

| source | n | qpos median | p99 | min cos | qvel median | p99 | min cos | ctrl median | p99 | min cos | model-ok n | qpos / qvel / ctrl median (model-ok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 256 | 3.8e-05 | 3.2e-02 | 0.9877 | 1.1e-05 | 1.6e-02 | 0.9960 | 1.8e-06 | 4.2e-03 | 0.9999 | 250 | 3.7e-05 / 1.0e-05 / 1.8e-06 |
| ours, interface defaults (clamp_kv=0, normal=3) | 256 | 3.8e-05 | 3.2e-02 | 0.9877 | 1.1e-05 | 1.6e-02 | 0.9960 | 1.8e-06 | 4.2e-03 | 0.9999 | 250 | 3.7e-05 / 1.0e-05 / 1.8e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 256 | 3.1e-01 | 3.7e+00 | -0.8795 | 3.5e-01 | 3.2e+00 | -0.7101 | 1.8e-02 | 2.2e-01 | 0.9382 | 250 | 3.1e-01 / 3.5e-01 / 1.8e-02 |
| MuJoCo Warp + PR #1535 @4af093f | 256 | 5.1e-05 | 3.2e-02 | 0.9877 | 1.1e-05 | 1.6e-02 | 0.9960 | 1.9e-06 | 4.2e-03 | 0.9999 | 250 | 5.0e-05 / 1.1e-05 / 1.8e-06 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 256 | 9.4e-05 | 4.8e-01 | 0.6810 | 2.0e-05 | 1.9e-01 | 0.8197 | 3.1e-06 | 6.4e-02 | 0.9353 | 250 | 9.1e-05 / 1.9e-05 / 2.9e-06 |
| Genesis 1.4.5, native reverse mode (initial qvel only) | 256 | not supported |  |  | 2.7e-01 | 1.9e+00 | -0.4255 | not supported |  |  | 250 | not supported / 2.8e-01 / not supported |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 256 | 6.6e-04 | 1.2e-02 | 0.9258 | 1.5e-04 | 2.7e-03 | 0.9935 | 7.9e-06 | 1.7e-04 | 0.9994 | 250 | 6.7e-04 / 1.5e-04 / 8.0e-06 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 256 | 2.5e-07 | 6.1e-03 | 0.9263 | 1.7e-07 | 5.6e-04 | 0.9936 | 2.1e-08 | 5.2e-06 | 0.9994 | 250 | 2.6e-07 / 1.7e-07 / 2.1e-08 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 256 | 7.0e-06 | 5.7e-05 | 0.9263 | 1.9e-06 | 1.5e-05 | 0.9936 | 6.1e-07 | 3.6e-06 | 0.9994 | 250 | 7.0e-06 / 1.9e-06 / 6.1e-07 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 256 | 2.0e-07 | 6.6e-06 | 0.9263 | 1.6e-07 | 7.9e-07 | 0.9936 | 3.2e-08 | 2.0e-07 | 0.9994 | 250 | 2.1e-07 / 1.6e-07 / 3.2e-08 |
| MuJoCo Warp 3.17.0 (native) | - | no backward (MuJoCo Warp has no gradient) | | | | | | | | | | |

### G6 (pooled over the four corpora)

| source | n | qpos median | p99 | min cos | qvel median | p99 | min cos | ctrl median | p99 | min cos | model-ok n | qpos / qvel / ctrl median (model-ok) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 256 | 3.1e-05 | 3.5e-02 | 0.7352 | 1.0e-05 | 4.4e-02 | 0.6000 | 2.3e-06 | 7.4e-03 | 0.9933 | 253 | 3.1e-05 / 9.9e-06 / 2.3e-06 |
| ours, interface defaults (clamp_kv=0, normal=3) | 256 | 3.1e-05 | 3.5e-02 | 0.7352 | 1.0e-05 | 4.4e-02 | 0.6000 | 2.3e-06 | 7.4e-03 | 0.9933 | 253 | 3.1e-05 / 9.9e-06 / 2.3e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 256 | 3.5e-01 | 4.2e+00 | -0.8974 | 4.0e-01 | 4.0e+00 | -0.8866 | 1.9e-02 | 2.4e-01 | 0.8968 | 253 | 3.5e-01 / 3.9e-01 / 1.8e-02 |
| MuJoCo Warp + PR #1535 @4af093f | 256 | 4.5e-05 | 3.5e-02 | 0.7352 | 1.0e-05 | 4.4e-02 | 0.6000 | 2.3e-06 | 7.4e-03 | 0.9933 | 253 | 4.4e-05 / 9.8e-06 / 2.2e-06 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 256 | 1.0e-04 | 4.6e-01 | 0.5606 | 2.5e-05 | 2.9e-01 | 0.5115 | 4.8e-06 | 1.2e-01 | 0.9700 | 253 | 1.0e-04 / 2.5e-05 / 4.7e-06 |
| Genesis 1.4.5, native reverse mode (initial qvel only) | 256 | not supported |  |  | 3.4e-01 | 2.5e+00 | -0.7006 | not supported |  |  | 253 | not supported / 3.3e-01 / not supported |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 256 | 5.5e-04 | 7.3e-03 | -0.7917 | 1.3e-04 | 2.2e-03 | 0.7273 | 7.8e-06 | 4.3e-04 | 0.9969 | 253 | 5.5e-04 / 1.3e-04 / 7.8e-06 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 256 | 2.2e-07 | 2.8e-04 | -0.4624 | 1.8e-07 | 1.8e-04 | 0.9596 | 2.4e-08 | 5.2e-06 | 0.9969 | 253 | 2.2e-07 / 1.8e-07 / 2.4e-08 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 256 | 5.7e-06 | 7.9e-05 | 0.9873 | 1.7e-06 | 1.9e-05 | 0.9623 | 6.2e-07 | 4.4e-06 | 0.9969 | 253 | 5.7e-06 / 1.7e-06 / 6.2e-07 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 256 | 1.7e-07 | 4.4e-05 | 0.9873 | 1.6e-07 | 2.1e-06 | 0.9623 | 3.3e-08 | 2.3e-07 | 0.9969 | 253 | 1.7e-07 / 1.6e-07 / 3.4e-08 |
| MuJoCo Warp 3.17.0 (native) | - | no backward (MuJoCo Warp has no gradient) | | | | | | | | | | |

50 iterations (pooled, all sampled states; qpos / qvel / ctrl median floored error per group):

| source | G0 | G4 | G5 | G6 |
|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 3.6e-05 / 1.0e-05 / 1.9e-06 | 4.3e-05 / 1.3e-05 / 2.0e-06 | 3.8e-05 / 1.1e-05 / 1.8e-06 | 3.1e-05 / 1.0e-05 / 2.3e-06 |
| ours, interface defaults (clamp_kv=0, normal=3) | 3.6e-05 / 1.0e-05 / 1.9e-06 | n/c | 3.8e-05 / 1.1e-05 / 1.8e-06 | 3.1e-05 / 1.0e-05 / 2.3e-06 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 2.9e-01 / 3.4e-01 / 1.5e-02 | 5.0e-01 / 4.8e-01 / 8.9e-02 | 3.1e-01 / 3.5e-01 / 1.8e-02 | 3.5e-01 / 4.0e-01 / 1.9e-02 |
| MuJoCo Warp + PR #1535 @4af093f | 5.2e-05 / 9.7e-06 / 2.0e-06 | n/c | 5.1e-05 / 1.1e-05 / 1.9e-06 | 4.6e-05 / 1.0e-05 / 2.3e-06 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 1.0e-04 / 2.0e-05 / 3.0e-06 | 1.1e-04 / 2.4e-05 / 2.8e-06 | 9.3e-05 / 2.0e-05 / 3.0e-06 | 1.0e-04 / 2.5e-05 / 4.8e-06 |
| Genesis 1.4.5, native reverse mode (initial qvel only) | n/s / 2.3e-01 / n/s | n/s / 3.3e-01 / n/s | n/s / 2.7e-01 / n/s | n/s / 3.3e-01 / n/s |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 6.7e-04 / 1.5e-04 / 7.8e-06 | 7.1e-04 / 1.7e-04 / 7.2e-06 | 6.6e-04 / 1.5e-04 / 7.9e-06 | 5.5e-04 / 1.3e-04 / 7.5e-06 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 2.3e-07 / 1.8e-07 / 2.2e-08 | 2.7e-07 / 2.2e-07 / 2.3e-08 | 2.5e-07 / 1.7e-07 / 2.1e-08 | 2.1e-07 / 1.8e-07 / 2.4e-08 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 6.6e-06 / 1.8e-06 / 5.9e-07 | 7.7e-06 / 2.3e-06 / 7.0e-07 | 7.0e-06 / 1.9e-06 / 6.1e-07 | 5.7e-06 / 1.7e-06 / 6.2e-07 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 1.8e-07 / 1.6e-07 / 3.0e-08 | 2.2e-07 / 2.0e-07 / 3.8e-08 | 2.0e-07 / 1.6e-07 / 3.2e-08 | 1.7e-07 / 1.6e-07 / 3.3e-08 |

## 3. Non-finite outputs and gradients on every corpus state

All 4 x 4096 states with each corpus's 3 stored cotangents (process 0); a state counts when any of its outputs, or any entry of its 3 VJPs, is not finite.

| source | it | forward non-finite states | gradient non-finite states | gradient non-finite values |
|---|---|---|---|---|
| ours, exact mode (clamp_kv=1, normal=0) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| ours, exact mode (clamp_kv=1, normal=0) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| ours, interface defaults (clamp_kv=0, normal=3) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| ours, interface defaults (clamp_kv=0, normal=3) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo Warp 3.17.0 (native) | 10 | 0 of 16384 | no gradient | - |
| MuJoCo Warp 3.17.0 (native) | 50 | 0 of 16384 | no gradient | - |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo Warp + PR #1535 @357a75d (as published) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo Warp + PR #1535 @4af093f | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo Warp + PR #1535 @4af093f | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MJX 3.17.0, forward-mode Jacobian (cot^T J) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| Genesis 1.4.5, native reverse mode (initial qvel only) (qvel block only) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| Genesis 1.4.5, native reverse mode (initial qvel only) (qvel block only) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-6 | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-6 | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, forward, eps 1e-08 (chosen with truth) | 50 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 10 | 0 of 16384 | 0 of 16384 | 0 |
| MuJoCo C mjd_transitionFD chained, centered, eps 1e-07 (chosen with truth) | 50 | 0 of 16384 | 0 of 16384 | 0 |

## 4. MuJoCo C finite differences across step sizes (mf_s0, G0, 64 states)

Raw (unfloored) relative VJP error per block, median / p90, against the truth; floored medians are in the JSON. chain = mjd_transitionFD per substep chained over the control step (MuJoCo C's native derivative, the benchmark's MuJoCo C row); direct = finite differences of the whole control step (the truth's construction). Headline options: 10 iterations, tolerance 1e-8, ls 8. ms per state: wall time of the batch on 12 CPU threads.

| construction | eps | qpos median | p90 | qvel median | p90 | ctrl median | p90 | ms per state |
|---|---|---|---|---|---|---|---|---|
| chain forward | 0.001 | 8.7e+00 | 1.2e+02 | 2.2e+00 | 1.6e+01 | 1.9e-01 | 3.7e+00 | 3.36 |
| chain forward | 0.0001 | 8.5e-01 | 6.3e+01 | 4.9e-02 | 8.2e+00 | 4.3e-03 | 9.3e-01 | 2.69 |
| chain forward | 1e-05 | 6.7e-03 | 1.4e+01 | 1.5e-03 | 1.9e-02 | 8.5e-05 | 1.1e-03 | 2.66 |
| chain forward | 1e-06 | 5.7e-04 | 2.1e-03 | 1.4e-04 | 5.9e-04 | 7.1e-06 | 4.1e-05 | 2.68 |
| chain forward | 1e-07 | 5.6e-05 | 2.1e-04 | 1.4e-05 | 5.9e-05 | 7.1e-07 | 4.1e-06 | 2.64 |
| chain forward | 1e-08 (chosen) | 5.8e-06 | 2.2e-05 | 1.6e-06 | 5.9e-06 | 5.6e-07 | 1.4e-06 | 2.84 |
| chain forward | 1e-09 | 6.9e-06 | 2.0e-05 | 1.1e-05 | 2.6e-05 | 1.2e-05 | 3.9e-05 | 2.65 |
| chain centered | 0.001 | 5.4e+00 | 7.4e+01 | 1.3e+00 | 8.6e+00 | 1.5e-01 | 1.4e+00 | 5.15 |
| chain centered | 0.0001 | 2.3e+00 | 9.3e+01 | 3.2e-01 | 1.3e+01 | 2.1e-02 | 9.1e-01 | 5.16 |
| chain centered | 1e-05 | 5.8e-05 | 3.6e+01 | 1.2e-05 | 4.3e+00 | 4.4e-07 | 2.5e-01 | 5.11 |
| chain centered (semi-self-referential) | 1e-06 | 1.2e-07 | 8.3e-07 | 4.9e-08 | 3.7e-07 | 4.8e-09 | 1.8e-08 | 5.11 |
| chain centered (semi-self-referential) | 1e-07 (chosen) | 1.8e-07 | 9.0e-07 | 4.7e-08 | 9.2e-08 | 2.9e-08 | 6.5e-08 | 5.15 |
| chain centered | 1e-08 | 3.8e-07 | 1.2e-06 | 4.5e-07 | 8.0e-07 | 2.8e-07 | 6.9e-07 | 5.13 |
| chain centered | 1e-09 | 3.6e-06 | 1.1e-05 | 4.9e-06 | 1.1e-05 | 3.6e-06 | 1.2e-05 | 5.13 |
| direct forward | 0.001 | 3.3e+00 | 2.0e+01 | 4.8e-04 | 3.4e-03 | 3.3e-05 | 5.6e-04 | 4.51 |
| direct forward | 0.0001 | 6.4e-01 | 6.8e+01 | 4.5e-05 | 3.0e-04 | 2.9e-06 | 8.4e-06 | 4.06 |
| direct forward | 1e-05 | 3.9e-03 | 9.8e+00 | 4.5e-06 | 3.0e-05 | 2.9e-07 | 8.4e-07 | 3.83 |
| direct forward | 1e-06 | 3.7e-04 | 1.5e-03 | 4.5e-07 | 3.1e-06 | 3.0e-08 | 8.3e-08 | 3.87 |
| direct forward | 1e-07 | 3.7e-05 | 1.5e-04 | 2.8e-07 | 6.2e-07 | 3.8e-08 | 9.6e-08 | 4.12 |
| direct forward | 1e-08 | 3.7e-06 | 1.6e-05 | 2.6e-06 | 5.8e-06 | 3.8e-07 | 9.7e-07 | 3.82 |
| direct forward | 1e-09 | 1.5e-06 | 3.1e-06 | 2.5e-05 | 5.3e-05 | 3.9e-06 | 1.0e-05 | 3.82 |
| direct centered | 0.001 | 2.7e+00 | 1.3e+01 | 3.8e-06 | 6.6e-01 | 1.0e-07 | 3.0e-03 | 7.31 |
| direct centered | 0.0001 | 1.4e+00 | 4.6e+01 | 3.0e-08 | 1.3e-07 | 2.7e-09 | 6.7e-09 | 7.04 |
| direct centered | 1e-05 | 3.4e-05 | 3.0e+01 | 1.5e-08 | 3.1e-08 | 2.2e-09 | 5.2e-09 | 6.84 |
| direct centered (semi-self-referential) | 1e-06 | 3.0e-10 | 7.6e-10 | 0 | 2.0e-09 | 0 | 1.9e-10 | 6.84 |
| direct centered (semi-self-referential) | 1e-07 | 1.8e-07 | 9.1e-07 | 1.5e-07 | 2.9e-07 | 2.0e-08 | 4.9e-08 | 6.86 |
| direct centered | 1e-08 | 2.0e-07 | 9.2e-07 | 1.5e-06 | 3.0e-06 | 2.1e-07 | 4.6e-07 | 6.87 |
| direct centered | 1e-09 | 7.6e-07 | 1.6e-06 | 1.4e-05 | 2.8e-05 | 2.2e-06 | 5.2e-06 | 7.11 |

Tight options (2000 iterations, tolerance 1e-13, ls 100), direct centered at 1e-6 (self-reference): qpos / qvel / ctrl raw median 3.0e-10 / 0 / 0; every tight-option row is in the JSON (`fd_sweep.results`, keys `tight/...`).

## Reproduce

The harness jobs (one engine group per GPU; MuJoCo C on the CPU), the sweep and this rendering:

```
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_dk --out runs/bench/x2 --engines dk,dk_iface --add-engine dk_iface=dk
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_genesis --out runs/bench/x2 --engines genesis --add-engine genesis=mjx --corpora mf_s0 --fresh
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_genesis_s1 --out runs/bench/x2 --engines genesis --add-engine genesis=mjx --corpora mf_s1 --fresh
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_genesis_s2 --out runs/bench/x2 --engines genesis --add-engine genesis=mjx --corpora mf_s2 --fresh
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_genesis_s3 --out runs/bench/x2 --engines genesis --add-engine genesis=mjx --corpora mf_s3 --fresh
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --no-sampler --processes 1 --fresh --job acc_mjc --engines mjc --out runs/bench/x2 --variant mjc=default,centered
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --no-sampler --processes 1 --fresh --job acc_mjc_eps_ctr --engines mjc --out runs/bench/x2 --variant mjc=centered --opt mjc.eps=1e-7
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --no-sampler --processes 1 --fresh --job acc_mjc_eps_fwd --engines mjc --out runs/bench/x2 --opt mjc.eps=1e-8
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_mjw --out runs/bench/x2 --engines mjw
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_mjx --out runs/bench/x2 --engines mjx
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --processes 1 --job acc_mjx_s1 --corpora mf_s1 --engines mjx --out runs/bench/x2
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --processes 1 --job acc_mjx_s2 --corpora mf_s2 --engines mjx --out runs/bench/x2
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --processes 1 --job acc_mjx_s3 --corpora mf_s3 --engines mjx --out runs/bench/x2
python -m benchmarks.bench run --machine v100 --part accuracy --no-anchor --job acc_pr --out runs/bench/x2 --engines mjw_pr1535,mjw_pr1535_fix
python -m benchmarks.bench.accuracy.fd_sweep --corpora mf_s0 --groups G0 --threads 12 --out runs/bench/x2/fd_sweep_mf_s0_G0.json
python -m benchmarks.bench.report.paper_accuracy --jobs runs/bench/x2/acc_dk.json runs/bench/x2/acc_genesis.json runs/bench/x2/acc_genesis_s1.json runs/bench/x2/acc_genesis_s2.json runs/bench/x2/acc_genesis_s3.json runs/bench/x2/acc_mjc.json runs/bench/x2/acc_mjc_eps_ctr.json runs/bench/x2/acc_mjc_eps_fwd.json runs/bench/x2/acc_mjw.json runs/bench/x2/acc_mjx.json runs/bench/x2/acc_mjx_s1.json runs/bench/x2/acc_mjx_s2.json runs/bench/x2/acc_mjx_s3.json runs/bench/x2/acc_pr.json --fd-sweep runs/bench/x2/fd_sweep_mf_s0_G0.json --out-json docs/results/paper_accuracy_heldout_v100.json --out-md docs/results/paper_accuracy_heldout_v100.md
```

## Notes

- Platform: every number was produced on the V100 accuracy platform (GPU engines) and its host CPU (MuJoCo C); this file has no timing except the FD sweep's cost column.
- Subsets: every forward row covers all 4096 states of each corpus, MJX included; every gradient row covers all sampled truth states of each group (G0 / G4 / G5 / G6: 64 per corpus, 256 pooled). Model-ok columns restrict to states where MuJoCo C and native MuJoCo Warp 3.17.0 agree on the discrete structure.
- Floors: the forward solver-truncation floor is MuJoCo C float64 itself at 10 / 50 iterations (tolerance 1e-8, ls 8) against its converged solution; the MuJoCo Warp run-to-run floor compares processes 1 and 2 with process 0 of the same unit (mf_s0 and mf_s1, 10 iterations: the units run in three processes). The gradient floor is the truth's own step-size floor, applied to every source.
- The interface defaults (clamp_kv=0, normal=3) and the @4af093f tree (which keeps the actuator damping -kv on the qvel path of clamped dofs, i.e. clamp_kv=0) are not comparable on G4: the truth is the exact derivative and has no clamp_kv=0 Jacobian. Their G4 errors against the exact truth are kept, for information only, under `gradient_vs_exact`.
- MuJoCo C FD rows: chained mjd_transitionFD at eps 1e-6 (the default) and at the eps chosen with the truth, forward 1e-08 and centered 1e-07 (the eps of 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09 with the smallest mean over the qpos / qvel / ctrl blocks of log10(median floored relative error), MuJoCo C chained FD at the headline options, mf_s0 G0 sampled states (all): chosen with the truth). The centered rows at 1e-6 and 1e-7 are semi-self-referential (centered differences of MuJoCo C's own float64 map at the truth's steps).
- MJX's gradient is its native forward-mode Jacobian (jax.jvp through every solver iteration; MJX has no reverse mode), contracted with the same cotangents (cot^T J). MuJoCo Warp 3.17.0 has no backward.
- PR #1535 @357a75d is the published head of the unmerged upstream pull request; @4af093f is that head plus two follow-up fixes on the pull request's branch (pyramidal edge rule, clamp multiplier). Both trees run unmodified.
- Run to run (later processes against process 0, bitwise identical / compared): Genesis 1.4.5 (float32, native) 4 / 4, MuJoCo Warp 3.17.0 (native) 0 / 4, ours (forward of both gradient modes) 4 / 4, MuJoCo Warp + PR #1535 @357a75d (as published) 0 / 4, MuJoCo Warp + PR #1535 @4af093f 0 / 4.
- Our forward is bitwise identical in both gradient modes on 8 of 8 (corpus, iterations) dumps.
- The harness's MuJoCo C forward equals the truth file's floor run bitwise on 8 of 8 (corpus, iterations) dumps.
- The FD sweep's chained rows and the harness's MuJoCo C dumps agree on mf_s0 G0 (identical medians).
- Model-ok states with a qvel error above 1e-3 at 10 / 50 iterations (each listed under `forward.model_ok_above_1e_3` with the other sources' errors at the same state): MuJoCo C 3.17.0 float64 (the cell's iterations) 10 / 0; MuJoCo Warp 3.17.0 (native) 3 / 0; MJX 3.17.0 (float32, native) 140 / 130; ours (forward of both gradient modes) 9 / 6; MuJoCo Warp + PR #1535 @357a75d (as published) 4 / 0; MuJoCo Warp + PR #1535 @4af093f 2 / 1.
- Our 6 model-ok states above 1e-3 at 50 iterations (largest 1.9e-01) are above it at 10 iterations too (6 of 6, largest relative change of the error between 10 and 50 iterations 0.0e+00); at those states MuJoCo Warp 3.17.0 is within 1.9e-06 and MuJoCo C float64 at 50 iterations within 0.0e+00 of the truth. So this is not solver truncation. They are contact-activation flips at the distance threshold: on the one that differs already in the first substep (mf_s2 state 0), MuJoCo C (float64) activates a second plane-capsule contact of a foot at a distance of -3.2e-8 m, which our float32 step evaluates on the other side of zero; the foot moves into the floor, so that contact's damping changes the step. Any float32 engine has such flips. 'Model-ok' is defined by MuJoCo C and MuJoCo Warp agreeing, so it removes MuJoCo Warp's own flips but not ours; on all states the share above 1e-3 is about the same for both (see the 'all' columns), and that is the comparison to quote.
- Genesis 1.4.5 (genesis-world from PyPI, quadrants 1.3.3, unmodified): its MJCF importer reads no <pair> and imports contype = conaffinity = 0 geoms as visual only, so the adapter writes the problem's model as an MJCF with contype = conaffinity = 1 on the robot geoms and no <pair> block; Genesis's default collision filter (self collision on, parent-child and qpos0-touching pairs off) then gives 502 pairs, exactly the model's 502 explicit pairs. Options: Newton, the cell's iterations, ls 8, tolerance 1e-8, pyramidal cone, integrator implicitfast, 4 x 5 ms (SimOptions(dt=0.02, substeps=4)), PD targets held (control_dofs_position with the model's kp / kv and force ranges), constraint_timeconst = the model's solref time constant 0.02.
- Genesis's deviations from the problem (recorded by its adapter): model: robot geoms contype = conaffinity = 1 and no <pair> block (Genesis reads no <pair>); Genesis's default collision filter gives the same 502 pairs (provenance model.pair_check); condim: 154 condim-1 (frictionless) pairs are frictional in Genesis (no per-contact condim); solref: Genesis's constraint_timeconst = 0.02 (the model's solref) instead of a per-geom solref (Genesis drops an MJCF default solref in favour of its own time constant); no qacc warm start input (cold constraint solve at every control step); tolerance 1e-08 passed to Genesis as is (float32); contact generation: Genesis's (multi-contact, contact_pruning_tolerance 0.02; MPR in the forward scene, GJK in the differentiable one), inverse weights recomputed by Genesis; gradient: integrator approximate_implicitfast (the only one Genesis differentiates), w.r.t. the initial qvel only; DR: foot friction through per-world friction ratios on the floor and sole links (956 worlds above the floor friction, 1337 below the sole friction get scaled sole friction on their sole-robot contacts too); COM through set_links_COM; QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith (quadrants' dispatch override; set on sm_70, where the decomposed constraint-solver kernel does not compile).
- Genesis's DR: the floor-sole friction of every world is exact (per-world friction ratios on the floor and sole links; Genesis takes the larger of two geoms' friction x ratio); in 3744 of 16384 states (friction above the floor's 1) and 5306 (below the soles' 0.6) the soles' contacts with other robot geoms then use the scaled sole friction too. Torso COM through set_links_COM (Genesis recomputes its inverse weights).
- Genesis's gradient is its native reverse mode (scene.backward) with respect to the initial qvel: the only integrator Genesis differentiates is approximate_implicitfast (its forward rows use implicitfast), and it has no gradient path for the joint positions or the PD targets, so the qpos and ctrl blocks are not supported (Genesis's own messages in the dump cells' gradient_not_supported).
- Genesis's NaN-gradient guard (GenesisException 'Nan grad in qpos or dofs_vel found', which aborts the whole batch) stopped 9 of 288 backward batches of 512 worlds on the first attempt (every Genesis dump: 12 units, 3 cotangents each); the adapter retries a batch once, then bisects it to single states: 8 retries succeeded, 1 unit(s) needed bisection, and no state stayed aborted. The guard does not fire on the same batches in every process.
- Genesis's forward is bitwise identical from process to process, its gradient is not: over 4 later processes against process 0 (mf_s0, mf_s1, 10 iterations) the qvel-block VJPs differ on 3597-3707 of 4096 states, relative difference median 2.7e-06, p99 up to 1.3e+00, max 4.9e+00 (`checks.gradient_run_to_run`).
- Genesis's forward differs from MuJoCo's on most states (another contact model, frictional condim-1 pairs, its own solver): its model-ok states above 1e-3 are counted under `forward.model_ok_above_1e_3_count`, not listed.

