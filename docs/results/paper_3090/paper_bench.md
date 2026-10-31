# Paper timing kit: NVIDIA GeForce RTX 3090 (driver 565.57.01)

- started 2026-10-29T08:30:53Z, finished 2026-10-29T09:22:17Z, 3173.7 s in all; setup: ok
- NablaEngine 0.1.0 commit: a revision before this repository's history: not recorded; nvcc 12.6.85; arch sm_86
- CPU Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz: 88 allocated of 88 logical (affinity 0-87)
- warp-lang wheel of the native venvs: cu12 (warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl)
- packages of the main environment: {"jax": "0.12.0.1", "jaxlib": "0.12.0.1", "mujoco": "3.17.0", "mujoco-mjx": "3.17.0", "warp-lang": "1.18.0+cu12", "numpy": "2.5.4"}
- corpora: mf_s0, mf_s1; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)
- MuJoCo C threads: 32 (PAPER_BENCH_MJC_THREADS); scheduler CPU counts null

What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of a differentiated rollout (the forward, then the adjoint). Jacobian rows: the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the per-world-rate rows is a linear extrapolation, the measured rows are labelled).

ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.

## Forward, 4096 worlds (mjlab G1)

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | 3.151 [3.138-3.172] | 3.684 [3.662-3.691] |  |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape (with tape) | 3.397 [3.384-3.444] | 3.811 [3.769-3.815] |  |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 12.546 [12.358-12.866] | 13.205 [13.147-13.253] | naconmax 94687 (nconmax 24) / njmax 128; the documented sizing rule (native headline) |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 13.743 [13.685-13.788] | 13.947 [13.932-13.976] | naconmax 262144 (nconmax 64) / njmax 192; our engine's capacity |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (default capacity) | NEFC overflow in 317 worlds | NEFC overflow in 378 worlds | naconmax 196608 (nconmax 48) / njmax 64; put_data's defaults; capacity overflow: not timed |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | 13.886 [13.640-14.214] | 14.424 [14.375-14.486] |  |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | 14.283 [13.881-14.299] | 14.707 [14.668-14.735] |  |
| MJX 3.17 (impl='jax', native) | forward (jit(vmap(scan))) | 512.120 [509.896-515.738] | 520.582 [519.787-522.447] |  |
| MJX 3.17 (impl='jax', native) | forward, XLA_FLAGS=--xla_gpu_triton_gemm_any=true | 510.360 [507.869-511.579] |  |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, Batch nthread 32 | 68.081 [66.942-68.311] | 67.646 [66.907-69.439] |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, 1 thread, 64 worlds (not 4096) | 24.814 [24.573-24.902] | 25.124 [24.929-25.318] |  |

## Gradient, 4096 worlds

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | bwd (exact) | 1.458 [1.457-1.464] | 1.567 [1.563-1.575] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (exact) | 5.118 [5.001-5.123] | 5.402 [5.381-5.404] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | bwd (interface) | 1.500 [1.468-1.507] | 1.578 [1.554-1.579] | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (interface) | 5.216 [5.184-5.230] | 5.459 [5.432-5.461] | interface defaults: clamp_kv=0, normal=3 |
| MuJoCo Warp 3.17.0 (PyPI, native) | backward | - | - | none in any release (no timing) |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph (its own taped forward recompute + tape.backward) | 41.387 [41.255-42.123] | 44.091 [43.907-44.152] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward + adjoint (sum of medians) | 55.273 | 58.515 | a differentiated rollout: the forward, then the adjoint graph |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph (its own taped forward recompute + tape.backward) | 42.564 [41.858-43.263] | 44.841 [44.618-44.854] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward + adjoint (sum of medians) | 56.847 | 59.547 | a differentiated rollout: the forward, then the adjoint graph |
| MJX 3.17 (impl='jax', native) | reverse mode |  |  | not supported natively: ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop. |

## Jacobian 71 x 99 (per world: us/world)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s0 | 4096 | 110.831 [110.543-111.806] | 27.05826 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s0 | 4096 | 105.183 [104.638-105.934] | 25.67951 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s0 | 4096 | 114.779 [114.342-114.786] | 28.0223 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s0 | 4096 | 109.527 [109.273-109.868] | 26.74005 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s1 | 4096 | 116.168 [115.827-116.962] | 28.36133 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s1 | 4096 | 112.074 [111.617-112.244] | 27.36187 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s1 | 4096 | 117.447 [116.793-117.506] | 28.6735 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s1 | 4096 | 113.066 [112.805-113.320] | 27.60388 | interface defaults: clamp_kv=0, normal=3 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 64 | 1019.548 [1014.550-1020.184] | 15930.4 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 256 | 4203.009 [4176.497-4231.113] | 16418.0 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native) | mf_s0 | B1024 | - |  | skipped: oom |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 4096 | 5830.145 [5686.642-5861.686] | 1423.38 | nthread 32; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 512 | 937.434 [913.520-990.642] | 1830.92 | nthread 32; per-world rate; at 4096 by linear extrapolation (not measured): 7499.448 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 64 | 1922.136 [1920.957-1925.700] | 30033.37 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 4096 | 10912.659 [10644.745-11263.468] | 2664.22 | nthread 32; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 512 | 1574.737 [1570.311-1628.068] | 3075.66 | nthread 32; per-world rate; at 4096 by linear extrapolation (not measured): 12597.903 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 64 | 3692.382 [3667.274-3705.899] | 57693.48 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s1 | 512 | 948.701 [919.092-973.026] | 1852.93 | nthread 32; per-world rate; at 4096 by linear extrapolation (not measured): 7589.601 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s1 | 512 | 1661.672 [1576.696-1709.300] | 3245.45 | nthread 32; per-world rate; at 4096 by linear extrapolation (not measured): 13293.363 ms |

## MJX on Menagerie unitree_g1 (its native options)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| MJX 3.17 (impl='jax', native) | forward, Menagerie unitree_g1/scene_mjx.xml native options | native_states | 4096 | 129.154 [127.699-130.697] |  | 5 substeps of 0.004 s |

## nworld sweep (mf_s0)

| engine | op | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd | 0.913 [0.912-0.914] | 1.399 [1.379-1.400] | 3.754 [3.743-3.778] | 14.044 [13.975-14.223] |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | 1.222 [1.220-1.227] | 1.954 [1.937-1.958] | 5.534 [5.526-5.540] | 20.746 [20.742-20.755] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 4.672 [4.639-4.681] | 5.922 [5.828-6.035] | 12.546 [12.358-12.866] | 41.539 [41.143-42.110] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 4.810 [4.805-4.861] | 6.346 [6.268-6.374] | 13.743 [13.685-13.788] | 43.010 [42.659-43.453] |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s0 | 3 | 3.151, 3.340, 3.278 | 3.278 | 6.0% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s0 | 3 | 5.118, 5.257, 5.257 | 5.257 | 2.7% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s0 | 3 | 5.216, 5.337, 5.345 | 5.337 | 2.5% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s0 | 3 | 12.546, 12.143, 13.013 | 12.546 | 7.2% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s0 | 3 | 13.743, 13.128, 13.790 | 13.743 | 5.0% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s0 | 3 | 13.886, 13.625, 14.268 | 13.886 | 4.7% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s0 | 3 | 41.387, 43.008, 44.010 | 43.008 | 6.3% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s0 | 3 | 14.283, 14.026, 14.289 | 14.283 | 1.9% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s0 | 3 | 42.564, 43.810, 44.180 | 43.810 | 3.8% |
| MJX 3.17 (impl='jax', native) | forward | mf_s0 | 3 | 512.120, 509.620, 510.340 | 510.340 | 0.5% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s0 | 3 | 68.081, 67.773, 70.853 | 68.081 | 4.5% |
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s1 | 3 | 3.684, 3.700, 3.726 | 3.700 | 1.1% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s1 | 3 | 5.402, 5.406, 5.442 | 5.406 | 0.7% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s1 | 3 | 5.459, 5.441, 5.433 | 5.441 | 0.5% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s1 | 3 | 13.205, 12.629, 13.416 | 13.205 | 6.2% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s1 | 3 | 13.947, 13.350, 13.990 | 13.947 | 4.8% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s1 | 3 | 14.424, 14.534, 14.901 | 14.534 | 3.3% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s1 | 3 | 44.091, 44.177, 44.962 | 44.177 | 2.0% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s1 | 3 | 14.707, 14.758, 14.977 | 14.758 | 1.8% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s1 | 3 | 44.841, 44.773, 45.187 | 44.841 | 0.9% |
| MJX 3.17 (impl='jax', native) | forward | mf_s1 | 3 | 520.582, 520.660, 518.475 | 520.582 | 0.4% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s1 | 3 | 67.646, 73.404, 65.714 | 67.646 | 11.7% |

## Bare kernels (attribution; device time, CUDA events)

| op | ms |
|---|---|
| forward kernels, tape off (CUDA events) | 3.023 [3.002-3.119] |
| forward kernels, tape on (CUDA events) | 3.131 [3.110-3.139] |
| k_adjoint (exact, CUDA events) | 1.279 [1.271-1.284] |
| k_adjoint (interface, CUDA events) | 1.289 [1.276-1.290] |

jax_ffi.init: 0.953 s (built now: False), arch sm_86, flags {"forward": ["-arch=sm_86"], "adjoint": ["-arch=sm_86"]}, forward defines []

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| ours | ok | 139.3 | 1560.0 (1365.0) | 334.21 | False | 0 |  |
| ours_bare | ok | 134.1 | 1920.0 (1695.0) | 216.4 | False | 0 |  |
| mjw_native | ok | 113.1 | 1650.0 (1560.0) | 337.65 | False | 0 |  |
| mjw_native_nw512 | ok | 12.8 | 1875.0 (1695.0) | 268.1 | False | 0 |  |
| mjw_native_nw1024 | ok | 13.0 | 1800.0 (1695.0) | 299.33 | False | 0 |  |
| mjw_native_nw16384 | ok | 46.3 | 1695.0 (1635.0) | 340.41 | False | 0 |  |
| mjw_pr1535 | ok | 174.0 | 1695.0 (1560.0) | 340.62 | False | 0 |  |
| mjw_pr1535_fix | ok | 58.3 | 1650.0 (1530.0) | 337.24 | False | 0 |  |
| mjx | ok | 795.7 | 1695.0 (210.0) | 349.77 | False | 0 |  |
| mjc | ok | 191.3 | 210.0 (210.0) | 20.59 | False | 0 |  |
| mjx_triton | ok | 73.2 | 1740.0 (1575.0) | 346.17 | False | 0 |  |
| rep2_ours | ok | 31.1 | 1590.0 (1485.0) | 326.36 | False | 0 |  |
| rep2_mjw_native | ok | 20.2 | 1770.0 (1695.0) | 341.09 | False | 0 |  |
| rep2_mjw_pr1535 | ok | 27.0 | 1695.0 (1620.0) | 334.12 | False | 0 |  |
| rep2_mjw_pr1535_fix | ok | 26.7 | 1695.0 (1575.0) | 333.29 | False | 0 |  |
| rep2_mjx | ok | 76.2 | 1635.0 (210.0) | 348.1 | False | 0 |  |
| rep2_mjc | ok | 6.5 | 210.0 (210.0) | 155.78 | False | 0 |  |
| rep3_ours | ok | 31.1 | 1605.0 (1470.0) | 329.02 | False | 0 |  |
| rep3_mjw_native | ok | 20.0 | 1695.0 (1575.0) | 324.88 | False | 0 |  |
| rep3_mjw_pr1535 | ok | 26.9 | 1695.0 (1545.0) | 329.38 | False | 0 |  |
| rep3_mjw_pr1535_fix | ok | 26.8 | 1695.0 (1560.0) | 329.9 | False | 0 |  |
| rep3_mjx | ok | 75.3 | 1665.0 (1515.0) | 348.13 | False | 0 |  |
| rep3_mjc | ok | 6.5 | 225.0 (210.0) | 153.78 | False | 0 |  |
