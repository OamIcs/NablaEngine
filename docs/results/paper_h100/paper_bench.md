# Paper timing kit: NVIDIA H100 80GB HBM3 (driver 595.91.07)

- started 2026-10-29T07:23:39Z, finished 2026-10-29T07:48:42Z, 1539.1 s in all; setup: ok
- NablaEngine 0.1.0 commit: a revision before this repository's history: not recorded; nvcc 12.8.93; arch sm_90
- CPU Intel(R) Xeon(R) Platinum 8462Y+: 128 allocated of 128 logical (affinity 0-127)
- warp-lang wheel of the native venvs: default (1.18.0)
- packages of the main environment: {"jax": "0.12.0.1", "jaxlib": "0.12.0.1", "mujoco": "3.17.0", "mujoco-mjx": "3.17.0", "warp-lang": "1.18.0+cu12", "numpy": "2.5.4"}
- corpora: mf_s0, mf_s1; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)
- MuJoCo C threads: 13 (PAPER_BENCH_MJC_THREADS); scheduler CPU counts null

What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of a differentiated rollout (the forward, then the adjoint). Jacobian rows: the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the per-world-rate rows is a linear extrapolation, the measured rows are labelled).

ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.

## Forward, 4096 worlds (mjlab G1)

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | 1.708 [1.708-1.710] | 1.724 [1.723-1.724] |  |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape (with tape) | 1.776 [1.775-1.776] | 1.790 [1.789-1.790] |  |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 8.680 [8.678-8.680] | 8.705 [8.703-8.705] | naconmax 94687 (nconmax 24) / njmax 128; the documented sizing rule (native headline) |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 8.800 [8.797-8.801] | 8.830 [8.828-8.833] | naconmax 262144 (nconmax 64) / njmax 192; our engine's capacity |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (default capacity) | NEFC overflow in 317 worlds | NEFC overflow in 378 worlds | naconmax 196608 (nconmax 48) / njmax 64; put_data's defaults; capacity overflow: not timed |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | 9.224 [9.222-9.224] | 9.252 [9.249-9.255] |  |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | 9.184 [9.182-9.185] | 9.212 [9.206-9.528] |  |
| MJX 3.17 (impl='jax', native) | forward (jit(vmap(scan))) | 215.981 [215.401-218.849] | 215.910 [215.854-216.166] |  |
| MJX 3.17 (impl='jax', native) | forward, XLA_FLAGS=--xla_gpu_triton_gemm_any=true | 215.470 [215.411-215.579] |  |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, Batch nthread 13 | 57.374 [57.233-59.161] | 58.169 [58.012-59.538] |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, 1 thread, 64 worlds (not 4096) | 11.913 [11.607-11.945] | 11.990 [11.795-12.011] |  |

## Gradient, 4096 worlds

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | bwd (exact) | 0.930 [0.930-0.932] | 0.929 [0.926-0.929] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (exact) | 2.734 [2.733-2.739] | 2.743 [2.741-2.747] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | bwd (interface) | 0.905 [0.905-0.906] | 0.904 [0.903-0.904] | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (interface) | 2.706 [2.703-2.707] | 2.712 [2.710-2.714] | interface defaults: clamp_kv=0, normal=3 |
| MuJoCo Warp 3.17.0 (PyPI, native) | backward | - | - | none in any release (no timing) |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph (its own taped forward recompute + tape.backward) | 25.112 [25.105-25.116] | 25.146 [25.146-25.148] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward + adjoint (sum of medians) | 34.336 | 34.398 | a differentiated rollout: the forward, then the adjoint graph |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph (its own taped forward recompute + tape.backward) | 25.127 [25.125-25.133] | 25.173 [25.162-25.177] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward + adjoint (sum of medians) | 34.310 | 34.385 | a differentiated rollout: the forward, then the adjoint graph |
| MJX 3.17 (impl='jax', native) | reverse mode |  |  | not supported natively: ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop. |

## Jacobian 71 x 99 (per world: us/world)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s0 | 4096 | 68.552 [68.481-68.586] | 16.73638 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s0 | 4096 | 66.697 [66.690-66.809] | 16.28351 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s0 | 4096 | 66.701 [66.646-66.892] | 16.28444 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s0 | 4096 | 64.886 [64.788-64.930] | 15.84128 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s1 | 4096 | 68.340 [68.308-68.387] | 16.68449 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s1 | 4096 | 66.574 [66.498-66.661] | 16.25341 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s1 | 4096 | 66.577 [66.459-66.584] | 16.2541 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s1 | 4096 | 64.824 [64.715-64.870] | 15.82617 | interface defaults: clamp_kv=0, normal=3 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 64 | 437.916 [436.938-438.737] | 6842.4 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 256 | 1564.775 [1557.767-1574.563] | 6112.4 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 1024 | 6124.921 [6124.821-6125.389] | 5981.4 | chunk 33 |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 4096 | 4743.230 [4709.556-4751.029] | 1158.02 | nthread 13; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 512 | 683.575 [683.314-702.993] | 1335.11 | nthread 13; per-world rate; at 4096 by linear extrapolation (not measured): 5468.611 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 64 | 929.228 [919.647-959.066] | 14519.19 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 4096 | 9168.478 [9098.293-9173.076] | 2238.4 | nthread 13; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 512 | 1293.429 [1285.082-1296.282] | 2526.23 | nthread 13; per-world rate; at 4096 by linear extrapolation (not measured): 10347.438 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 64 | 1807.162 [1806.462-1818.455] | 28236.9 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s1 | 512 | 686.214 [669.143-692.946] | 1340.26 | nthread 13; per-world rate; at 4096 by linear extrapolation (not measured): 5489.705 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s1 | 512 | 1242.043 [1237.152-1248.686] | 2425.86 | nthread 13; per-world rate; at 4096 by linear extrapolation (not measured): 9936.323 ms |

## MJX on Menagerie unitree_g1 (its native options)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| MJX 3.17 (impl='jax', native) | forward, Menagerie unitree_g1/scene_mjx.xml native options | native_states | 4096 | 58.380 [58.368-58.793] |  | 5 substeps of 0.004 s |

## nworld sweep (mf_s0)

| engine | op | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd | 0.762 [0.762-0.763] | 0.905 [0.904-0.905] | 1.705 [1.704-1.706] | 6.450 [6.447-6.455] |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | 0.964 [0.964-0.964] | 1.196 [1.195-1.196] | 2.736 [2.735-2.738] | 11.024 [11.020-11.031] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 4.709 [4.708-4.710] | 5.348 [5.348-5.349] | 8.680 [8.678-8.680] | 22.179 [22.175-22.181] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 4.738 [4.738-4.738] | 5.359 [5.358-5.360] | 8.800 [8.797-8.801] | 22.784 [22.780-22.786] |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s0 | 3 | 1.708, 1.700, 1.700 | 1.700 | 0.5% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s0 | 3 | 2.734, 2.728, 2.724 | 2.728 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s0 | 3 | 2.706, 2.690, 2.689 | 2.690 | 0.6% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s0 | 3 | 8.680, 8.652, 8.651 | 8.652 | 0.3% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s0 | 3 | 8.800, 8.769, 8.769 | 8.769 | 0.3% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s0 | 3 | 9.224, 9.190, 9.191 | 9.191 | 0.4% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s0 | 3 | 25.112, 25.039, 25.046 | 25.046 | 0.3% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s0 | 3 | 9.184, 9.190, 9.190 | 9.190 | 0.1% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s0 | 3 | 25.127, 25.149, 25.148 | 25.148 | 0.1% |
| MJX 3.17 (impl='jax', native) | forward | mf_s0 | 3 | 215.981, 216.364, 216.012 | 216.012 | 0.2% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s0 | 3 | 57.374, 57.173, 57.136 | 57.173 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s1 | 3 | 1.724, 1.712, 1.714 | 1.714 | 0.7% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s1 | 3 | 2.743, 2.731, 2.731 | 2.731 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s1 | 3 | 2.712, 2.697, 2.700 | 2.700 | 0.5% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s1 | 3 | 8.705, 8.676, 8.675 | 8.676 | 0.3% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s1 | 3 | 8.830, 8.805, 8.804 | 8.805 | 0.3% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s1 | 3 | 9.252, 9.217, 9.217 | 9.217 | 0.4% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s1 | 3 | 25.146, 25.085, 25.084 | 25.085 | 0.2% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s1 | 3 | 9.212, 9.215, 9.219 | 9.215 | 0.1% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s1 | 3 | 25.173, 25.188, 25.198 | 25.188 | 0.1% |
| MJX 3.17 (impl='jax', native) | forward | mf_s1 | 3 | 215.910, 216.823, 215.721 | 215.910 | 0.5% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s1 | 3 | 58.169, 58.225, 58.232 | 58.225 | 0.1% |

## Bare kernels (attribution; device time, CUDA events)

| op | ms |
|---|---|
| forward kernels, tape off (CUDA events) | 1.700 [1.699-1.701] |
| forward kernels, tape on (CUDA events) | 1.760 [1.759-1.760] |
| k_adjoint (exact, CUDA events) | 0.912 [0.909-0.916] |
| k_adjoint (interface, CUDA events) | 0.888 [0.887-0.890] |

jax_ffi.init: 9.253 s (built now: True), arch sm_90, flags {"forward": ["-arch=sm_90"], "adjoint": ["-arch=sm_90"]}, forward defines ['-DSPLIT_PRE_MINB=16', '-DSPLIT_POST_MAXNREG=128']

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| ours | ok | 143.2 | 1980.0 (1980.0) | 362.99 | False | 0 |  |
| ours_bare | ok | 47.9 | 1980.0 (1980.0) | 191.75 | False | 0 |  |
| mjw_native | ok | 61.0 | 1980.0 (1980.0) | 258.06 | False | 0 |  |
| mjw_native_nw512 | ok | 8.2 | 1980.0 (1980.0) | 160.01 | False | 0 |  |
| mjw_native_nw1024 | ok | 8.3 | 1980.0 (1980.0) | 180.76 | False | 0 |  |
| mjw_native_nw16384 | ok | 14.8 | 1980.0 (1980.0) | 337.88 | False | 0 |  |
| mjw_pr1535 | ok | 81.3 | 1980.0 (1980.0) | 285.54 | False | 0 |  |
| mjw_pr1535_fix | ok | 32.9 | 1980.0 (1980.0) | 287.09 | False | 0 |  |
| mjx | ok | 443.4 | 1980.0 (1980.0) | 490.07 | False | 0 |  |
| mjc | ok | 140.8 | 345.0 (345.0) | 122.47 | False | 0 |  |
| mjx_triton | ok | 37.9 | 1980.0 (1980.0) | 421.78 | False | 0 |  |
| rep2_ours | ok | 30.7 | 1980.0 (1980.0) | 337.99 | False | 0 |  |
| rep2_mjw_native | ok | 13.8 | 1980.0 (1980.0) | 259.01 | False | 0 |  |
| rep2_mjw_pr1535 | ok | 15.9 | 1980.0 (1980.0) | 285.15 | False | 0 |  |
| rep2_mjw_pr1535_fix | ok | 17.6 | 1980.0 (1980.0) | 286.17 | False | 0 |  |
| rep2_mjx | ok | 39.6 | 1980.0 (1980.0) | 422.62 | False | 0 |  |
| rep2_mjc | ok | 5.5 | 345.0 (345.0) | 123.09 | False | 0 |  |
| rep3_ours | ok | 30.6 | 1980.0 (1980.0) | 337.83 | False | 0 |  |
| rep3_mjw_native | ok | 13.8 | 1980.0 (1980.0) | 259.4 | False | 0 |  |
| rep3_mjw_pr1535 | ok | 44.2 | 1980.0 (1980.0) | 284.89 | False | 0 |  |
| rep3_mjw_pr1535_fix | ok | 16.1 | 1980.0 (1980.0) | 283.87 | False | 0 |  |
| rep3_mjx | ok | 40.4 | 1980.0 (1980.0) | 424.39 | False | 0 |  |
| rep3_mjc | ok | 5.6 | 1980.0 (1980.0) | 123.17 | False | 0 |  |
