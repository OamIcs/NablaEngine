# Paper timing kit: Tesla V100-SXM2-16GB (driver 550.90.07)

- started 2026-10-29T08:20:01Z, finished 2026-10-29T08:43:01Z, 1380.9 s in all; setup: ok
- NablaEngine 0.1.0 commit: a revision before this repository's history: not recorded; nvcc 12.4.131; arch sm_70
- CPU Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz: 48 allocated of 48 logical (affinity 0-47)
- warp-lang wheel of the native venvs: cu12 (warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl)
- packages of the main environment: {"jax": "0.12.0.1", "jaxlib": "0.12.0.1", "mujoco": "3.17.0", "mujoco-mjx": "3.17.0", "warp-lang": "1.18.0+cu12", "numpy": "2.5.4"}
- corpora: mf_s0, mf_s1; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)
- MuJoCo C threads: 48 (len(os.sched_getaffinity(0))); scheduler CPU counts null

What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of a differentiated rollout (the forward, then the adjoint). Jacobian rows: the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the per-world-rate rows is a linear extrapolation, the measured rows are labelled).

ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.

## Forward, 4096 worlds (mjlab G1)

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | 5.793 [5.787-5.797] | 5.844 [5.836-5.845] |  |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape (with tape) | 6.016 [6.014-6.017] | 6.075 [6.071-6.080] |  |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 18.420 [18.417-18.426] | 18.499 [18.494-18.505] | naconmax 94687 (nconmax 24) / njmax 128; the documented sizing rule (native headline) |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 18.876 [18.874-18.879] | 18.952 [18.950-18.955] | naconmax 262144 (nconmax 64) / njmax 192; our engine's capacity |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (default capacity) | NEFC overflow in 317 worlds | NEFC overflow in 378 worlds | naconmax 196608 (nconmax 48) / njmax 64; put_data's defaults; capacity overflow: not timed |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | 20.808 [20.802-20.814] | 20.967 [20.959-20.970] |  |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | 20.803 [20.800-20.805] | 20.973 [20.962-20.979] |  |
| MJX 3.17 (impl='jax', native) | forward (jit(vmap(scan))) | 709.392 [709.312-709.707] | 709.466 [709.291-709.903] |  |
| MJX 3.17 (impl='jax', native) | forward, XLA_FLAGS=--xla_gpu_triton_gemm_any=true | 711.028 [710.934-711.285] |  |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, Batch nthread 48 | 49.320 [49.198-49.475] | 50.269 [50.084-50.712] |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, 1 thread, 64 worlds (not 4096) | 20.854 [20.669-20.974] | 22.058 [21.933-22.446] |  |

## Gradient, 4096 worlds

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | bwd (exact) | 4.526 [4.524-4.535] | 4.553 [4.550-4.560] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (exact) | 10.577 [10.576-10.585] | 10.657 [10.653-10.659] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | bwd (interface) | 4.530 [4.528-4.535] | 4.560 [4.559-4.565] | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (interface) | 10.568 [10.560-10.575] | 10.672 [10.660-10.678] | interface defaults: clamp_kv=0, normal=3 |
| MuJoCo Warp 3.17.0 (PyPI, native) | backward | - | - | none in any release (no timing) |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph (its own taped forward recompute + tape.backward) | 63.225 [63.211-63.233] | 63.447 [63.393-63.462] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward + adjoint (sum of medians) | 84.033 | 84.414 | a differentiated rollout: the forward, then the adjoint graph |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph (its own taped forward recompute + tape.backward) | 63.447 [63.430-63.461] | 63.621 [63.605-63.623] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward + adjoint (sum of medians) | 84.250 | 84.594 | a differentiated rollout: the forward, then the adjoint graph |
| MJX 3.17 (impl='jax', native) | reverse mode |  |  | not supported natively: ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop. |

## Jacobian 71 x 99 (per world: us/world)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s0 | 4096 | 329.062 [328.965-329.197] | 80.3374 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s0 | 4096 | 322.994 [322.901-323.138] | 78.85588 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s0 | 4096 | 329.359 [329.115-329.658] | 80.40988 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s0 | 4096 | 323.274 [323.055-323.323] | 78.92432 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s1 | 4096 | 330.962 [330.822-331.064] | 80.80122 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s1 | 4096 | 324.781 [324.474-325.045] | 79.29217 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s1 | 4096 | 331.373 [331.208-331.431] | 80.90171 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s1 | 4096 | 325.627 [325.395-325.717] | 79.49873 | interface defaults: clamp_kv=0, normal=3 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 64 | 2013.360 [2012.304-2015.429] | 31458.7 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native) | mf_s0 | B256 | - |  | skipped: oom |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 4096 | 4148.805 [4146.760-4166.787] | 1012.89 | nthread 48; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 512 | 658.260 [656.370-658.492] | 1285.67 | nthread 48; per-world rate; at 4096 by linear extrapolation (not measured): 5266.104 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 64 | 1624.372 [1623.577-1624.665] | 25380.81 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 4096 | 7844.277 [7753.613-7963.254] | 1915.11 | nthread 48; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 512 | 1195.325 [1172.321-1219.882] | 2334.62 | nthread 48; per-world rate; at 4096 by linear extrapolation (not measured): 9562.604 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 64 | 3158.151 [3155.469-3166.209] | 49346.1 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s1 | 512 | 662.103 [644.263-666.496] | 1293.17 | nthread 48; per-world rate; at 4096 by linear extrapolation (not measured): 5296.824 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s1 | 512 | 1212.177 [1167.669-1215.911] | 2367.53 | nthread 48; per-world rate; at 4096 by linear extrapolation (not measured): 9697.403 ms |

## MJX on Menagerie unitree_g1 (its native options)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| MJX 3.17 (impl='jax', native) | forward, Menagerie unitree_g1/scene_mjx.xml native options | native_states | 4096 | 150.658 [150.459-151.716] |  | 5 substeps of 0.004 s |

## nworld sweep (mf_s0)

| engine | op | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd | 1.131 [1.128-1.131] | 1.848 [1.845-1.851] | 5.754 [5.750-5.761] | 24.318 [24.316-24.326] |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | 1.718 [1.717-1.719] | 2.982 [2.978-2.988] | 10.528 [10.522-10.543] | 43.685 [43.664-43.730] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 7.676 [7.676-7.677] | 9.408 [9.407-9.409] | 18.420 [18.417-18.426] | 57.529 [57.520-57.568] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 7.770 [7.768-7.771] | 9.493 [9.493-9.493] | 18.876 [18.874-18.879] | 59.611 [59.546-59.671] |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s0 | 3 | 5.793, 5.793, 5.770 | 5.793 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s0 | 3 | 10.577, 10.569, 10.535 | 10.569 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s0 | 3 | 10.568, 10.575, 10.538 | 10.568 | 0.4% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s0 | 3 | 18.420, 18.413, 18.419 | 18.419 | 0.0% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s0 | 3 | 18.876, 18.877, 18.875 | 18.876 | 0.0% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s0 | 3 | 20.808, 20.798, 20.797 | 20.798 | 0.1% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s0 | 3 | 63.225, 63.212, 63.210 | 63.212 | 0.0% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s0 | 3 | 20.803, 20.799, 20.799 | 20.799 | 0.0% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s0 | 3 | 63.447, 63.447, 63.433 | 63.447 | 0.0% |
| MJX 3.17 (impl='jax', native) | forward | mf_s0 | 3 | 709.392, 712.809, 710.445 | 710.445 | 0.5% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s0 | 3 | 49.320, 49.038, 49.141 | 49.141 | 0.6% |
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s1 | 3 | 5.844, 5.843, 5.822 | 5.843 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s1 | 3 | 10.657, 10.601, 10.606 | 10.606 | 0.5% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s1 | 3 | 10.672, 10.608, 10.602 | 10.608 | 0.7% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s1 | 3 | 18.499, 18.493, 18.495 | 18.495 | 0.0% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s1 | 3 | 18.952, 18.953, 18.953 | 18.953 | 0.0% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s1 | 3 | 20.967, 20.963, 20.959 | 20.963 | 0.0% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s1 | 3 | 63.447, 63.409, 63.403 | 63.409 | 0.1% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s1 | 3 | 20.973, 20.961, 20.965 | 20.965 | 0.1% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s1 | 3 | 63.621, 63.608, 63.624 | 63.621 | 0.0% |
| MJX 3.17 (impl='jax', native) | forward | mf_s1 | 3 | 709.466, 710.436, 710.157 | 710.157 | 0.1% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s1 | 3 | 50.269, 50.037, 50.253 | 50.253 | 0.5% |

## Bare kernels (attribution; device time, CUDA events)

| op | ms |
|---|---|
| forward kernels, tape off (CUDA events) | 5.770 [5.768-5.772] |
| forward kernels, tape on (CUDA events) | 5.982 [5.977-5.983] |
| k_adjoint (exact, CUDA events) | 4.496 [4.489-4.500] |
| k_adjoint (interface, CUDA events) | 4.498 [4.495-4.503] |

jax_ffi.init: 15.513 s (built now: True), arch sm_70, flags {"forward": ["-arch=sm_70"], "adjoint": ["-arch=sm_70"]}, forward defines []

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| ours | ok | 158.4 | 1530.0 (1312.0) | 176.72 | False | 0 |  |
| ours_bare | ok | 65.1 | 1530.0 (1312.0) | 173.28 | False | 0 |  |
| mjw_native | ok | 27.0 | 1530.0 (1530.0) | 163.46 | False | 0 |  |
| mjw_native_nw512 | ok | 8.9 | 1530.0 (1530.0) | 100.76 | False | 0 |  |
| mjw_native_nw1024 | ok | 8.9 | 1530.0 (1312.0) | 117.47 | False | 0 |  |
| mjw_native_nw16384 | ok | 11.9 | 1530.0 (1530.0) | 193.28 | False | 0 |  |
| mjw_pr1535 | ok | 31.3 | 1530.0 (1530.0) | 172.79 | False | 0 |  |
| mjw_pr1535_fix | ok | 30.7 | 1530.0 (1312.0) | 171.81 | False | 0 |  |
| mjx | ok | 399.9 | 1530.0 (1312.0) | 239.88 | False | 0 |  |
| mjc | ok | 143.3 | 135.0 (135.0) | 65.58 | False | 0 |  |
| mjx_triton | ok | 39.3 | 1530.0 (1312.0) | 212.27 | False | 0 |  |
| rep2_ours | ok | 32.6 | 1530.0 (1530.0) | 176.23 | False | 0 |  |
| rep2_mjw_native | ok | 15.1 | 1530.0 (1312.0) | 163.46 | False | 0 |  |
| rep2_mjw_pr1535 | ok | 18.8 | 1530.0 (1530.0) | 171.81 | False | 0 |  |
| rep2_mjw_pr1535_fix | ok | 18.7 | 1530.0 (1530.0) | 171.81 | False | 0 |  |
| rep2_mjx | ok | 46.8 | 1530.0 (1447.0) | 220.26 | False | 0 |  |
| rep2_mjc | ok | 5.6 | 1530.0 (1530.0) | 68.37 | False | 0 |  |
| rep3_ours | ok | 32.2 | 1530.0 (1530.0) | 177.7 | False | 0 |  |
| rep3_mjw_native | ok | 15.1 | 1530.0 (1312.0) | 163.95 | False | 0 |  |
| rep3_mjw_pr1535 | ok | 18.6 | 1530.0 (1530.0) | 173.77 | False | 0 |  |
| rep3_mjw_pr1535_fix | ok | 18.7 | 1530.0 (1530.0) | 173.28 | False | 0 |  |
| rep3_mjx | ok | 47.4 | 1530.0 (1530.0) | 219.63 | False | 0 |  |
| rep3_mjc | ok | 5.6 | 525.0 (135.0) | 68.37 | False | 0 |  |
