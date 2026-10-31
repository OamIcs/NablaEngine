# Paper timing kit: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.58.03)

- started 2026-10-29T00:04:38Z, finished 2026-10-29T00:20:06Z, 957.7 s in all; setup: ok
- NablaEngine 0.1.0 commit: a revision before this repository's history: not recorded; nvcc 12.8.93; arch sm_120
- CPU AMD EPYC 9555 64-Core Processor: 16 allocated of 16 logical (affinity 0-15)
- warp-lang wheel of the native venvs: default (1.18.0)
- packages of the main environment: {"jax": "0.12.0.1", "jaxlib": "0.12.0.1", "mujoco": "3.17.0", "mujoco-mjx": "3.17.0", "warp-lang": "1.18.0+cu12", "numpy": "2.5.4"}
- corpora: mf_s0, mf_s1; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)
- MuJoCo C threads: 16 (len(os.sched_getaffinity(0))); scheduler CPU counts null

What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of a differentiated rollout (the forward, then the adjoint). Jacobian rows: the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the per-world-rate rows is a linear extrapolation, the measured rows are labelled).

ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.

## Forward, 4096 worlds (mjlab G1)

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | 1.272 [1.271-1.272] | 1.186 [1.185-1.186] |  |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape (with tape) | 1.297 [1.265-1.326] | 1.226 [1.225-1.227] |  |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 6.149 [6.148-6.150] | 6.186 [6.184-6.188] | naconmax 94688 (nconmax 24) / njmax 128; the documented sizing rule (native headline) |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 6.145 [6.123-6.213] | 6.088 [6.086-6.172] | naconmax 262144 (nconmax 64) / njmax 192; our engine's capacity |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (default capacity) | NEFC overflow in 317 worlds | NEFC overflow in 378 worlds | naconmax 196608 (nconmax 48) / njmax 64; put_data's defaults; capacity overflow: not timed |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | 6.533 [6.532-6.533] | 6.545 [6.544-6.547] |  |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | 6.390 [6.378-6.424] | 6.346 [6.345-6.347] |  |
| MJX 3.17 (impl='jax', native) | forward (jit(vmap(scan))) | 255.633 [255.565-256.006] | 254.304 [254.273-254.862] |  |
| MJX 3.17 (impl='jax', native) | forward, XLA_FLAGS=--xla_gpu_triton_gemm_any=true | 255.611 [255.584-255.653] |  |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, Batch nthread 16 | 34.746 [34.700-34.823] | 35.168 [35.161-35.175] |  |
| MuJoCo C 3.17 (CPU, float64, native) | forward, 1 thread, 64 worlds (not 4096) | 8.591 [8.585-8.601] | 8.784 [8.781-8.801] |  |

## Gradient, 4096 worlds

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | bwd (exact) | 0.563 [0.563-0.564] | 0.532 [0.531-0.532] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (exact) | 1.800 [1.800-1.804] | 1.753 [1.752-1.753] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | bwd (interface) | 0.537 [0.536-0.537] | 0.533 [0.533-0.533] | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (interface) | 1.783 [1.783-1.783] | 1.755 [1.755-1.758] | interface defaults: clamp_kv=0, normal=3 |
| MuJoCo Warp 3.17.0 (PyPI, native) | backward | - | - | none in any release (no timing) |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph (its own taped forward recompute + tape.backward) | 20.967 [20.963-20.974] | 20.992 [20.978-20.994] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward + adjoint (sum of medians) | 27.499 | 27.537 | a differentiated rollout: the forward, then the adjoint graph |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph (its own taped forward recompute + tape.backward) | 21.048 [21.040-21.056] | 20.581 [20.578-20.588] | the gradient from the input state; compare with ours vjp |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward + adjoint (sum of medians) | 27.438 | 26.928 | a differentiated rollout: the forward, then the adjoint graph |
| MJX 3.17 (impl='jax', native) | reverse mode |  |  | not supported natively: ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop. |

## Jacobian 71 x 99 (per world: us/world)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s0 | 4096 | 39.907 [39.747-39.919] | 9.74286 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s0 | 4096 | 38.608 [38.591-38.622] | 9.42569 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s0 | 4096 | 39.735 [39.699-39.743] | 9.70097 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s0 | 4096 | 38.422 [38.405-38.433] | 9.38027 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s1 | 4096 | 39.324 [39.318-39.330] | 9.60052 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s1 | 4096 | 38.035 [38.025-38.049] | 9.28583 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s1 | 4096 | 39.380 [39.368-39.394] | 9.61424 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s1 | 4096 | 38.113 [38.111-38.121] | 9.30503 | interface defaults: clamp_kv=0, normal=3 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 64 | 530.727 [530.690-531.335] | 8292.6 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 256 | 2100.410 [2099.471-2105.685] | 8204.7 | chunk 33 |
| MJX 3.17 (impl='jax', native) | forward-mode Jacobian (native), 99 jvp directions | mf_s0 | 1024 | 8840.144 [8840.012-8840.192] | 8633.0 | chunk 33 |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 4096 | 2750.441 [2741.137-2758.495] | 671.49 | nthread 16; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 512 | 387.159 [385.859-389.483] | 756.17 | nthread 16; per-world rate; at 4096 by linear extrapolation (not measured): 3097.272 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s0 | 64 | 650.957 [650.942-651.528] | 10171.21 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 4096 | 5252.529 [5238.075-5295.929] | 1282.36 | nthread 16; measured |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 512 | 740.834 [740.245-743.948] | 1446.94 | nthread 16; per-world rate; at 4096 by linear extrapolation (not measured): 5926.666 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s0 | 64 | 1275.747 [1275.639-1275.774] | 19933.55 | nthread 1; 1 thread |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, forward, eps 1e-06 | mf_s1 | 512 | 392.031 [390.913-392.848] | 765.69 | nthread 16; per-world rate; at 4096 by linear extrapolation (not measured): 3136.266 ms |
| MuJoCo C 3.17 (CPU, float64, native) | FD Jacobian, chained mjd_transitionFD, centered, eps 1e-06 | mf_s1 | 512 | 751.995 [747.912-752.681] | 1468.74 | nthread 16; per-world rate; at 4096 by linear extrapolation (not measured): 6015.959 ms |

## MJX on Menagerie unitree_g1 (its native options)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| MJX 3.17 (impl='jax', native) | forward, Menagerie unitree_g1/scene_mjx.xml native options | native_states | 4096 | 50.711 [50.698-50.728] |  | 5 substeps of 0.004 s |

## nworld sweep (mf_s0)

| engine | op | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd | 0.609 [0.609-0.609] | 0.723 [0.723-0.724] | 1.201 [1.200-1.201] | 3.962 [3.961-3.964] |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | 0.797 [0.797-0.797] | 0.943 [0.943-0.944] | 1.783 [1.782-1.784] | 5.938 [5.937-5.938] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 3.771 [3.771-3.772] | 4.095 [4.094-4.095] | 6.149 [6.148-6.150] | 15.965 [15.963-15.966] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 3.759 [3.759-3.760] | 4.096 [4.096-4.097] | 6.145 [6.123-6.213] | 16.228 [16.226-16.230] |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s0 | 3 | 1.272, 1.214, 1.202 | 1.214 | 5.8% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s0 | 3 | 1.800, 1.784, 1.783 | 1.784 | 0.9% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s0 | 3 | 1.783, 1.785, 1.785 | 1.785 | 0.1% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s0 | 3 | 6.149, 5.840, 5.839 | 5.840 | 5.3% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s0 | 3 | 6.145, 6.057, 5.932 | 6.057 | 3.6% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s0 | 3 | 6.533, 6.334, 6.335 | 6.335 | 3.1% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s0 | 3 | 20.967, 20.488, 20.491 | 20.491 | 2.3% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s0 | 3 | 6.390, 6.337, 6.338 | 6.338 | 0.8% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s0 | 3 | 21.048, 20.571, 20.569 | 20.571 | 2.3% |
| MJX 3.17 (impl='jax', native) | forward | mf_s0 | 3 | 255.633, 255.000, 254.335 | 255.000 | 0.5% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s0 | 3 | 34.746, 34.751, 34.491 | 34.746 | 0.8% |
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s1 | 3 | 1.186, 1.184, 1.185 | 1.185 | 0.1% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s1 | 3 | 1.753, 1.752, 1.751 | 1.752 | 0.1% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s1 | 3 | 1.755, 1.754, 1.754 | 1.754 | 0.1% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s1 | 3 | 6.186, 5.868, 5.868 | 5.868 | 5.4% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s1 | 3 | 6.088, 6.086, 5.959 | 6.086 | 2.2% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | forward graph | mf_s1 | 3 | 6.545, 6.345, 6.344 | 6.345 | 3.2% |
| MuJoCo Warp + unmerged PR #1535 @357a75d (as published) | adjoint graph | mf_s1 | 3 | 20.992, 20.501, 20.501 | 20.501 | 2.4% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | forward graph | mf_s1 | 3 | 6.346, 6.346, 6.348 | 6.346 | 0.0% |
| MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes) | adjoint graph | mf_s1 | 3 | 20.581, 20.581, 20.579 | 20.581 | 0.0% |
| MJX 3.17 (impl='jax', native) | forward | mf_s1 | 3 | 254.304, 254.281, 253.330 | 254.281 | 0.4% |
| MuJoCo C 3.17 (CPU, float64, native) | forward (Batch) | mf_s1 | 3 | 35.168, 35.149, 35.168 | 35.168 | 0.1% |

jax_ffi.init: 6.929 s (built now: True), arch sm_120, flags {"forward": ["-arch=compute_90", "-code=sm_120"], "adjoint": ["-arch=compute_90", "-code=sm_120"]}, forward defines ['-DSPLIT_PRE_MINB=16']

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| ours | ok | 124.3 | 2422.0 (2272.0) | 419.64 | False | 0 |  |
| mjw_native | ok | 46.4 | 2310.0 (2302.0) | 278.81 | False | 0 |  |
| mjw_native_nw512 | ok | 7.4 | 2310.0 (2310.0) | 158.0 | False | 0 |  |
| mjw_native_nw1024 | ok | 7.5 | 2310.0 (2302.0) | 183.29 | False | 0 |  |
| mjw_native_nw16384 | ok | 8.2 | 2415.0 (2287.0) | 336.11 | False | 0 |  |
| mjw_pr1535 | ok | 49.4 | 2310.0 (2295.0) | 289.91 | False | 0 |  |
| mjw_pr1535_fix | ok | 28.6 | 2422.0 (2295.0) | 290.65 | False | 0 |  |
| mjx | ok | 282.0 | 2422.0 (2295.0) | 383.34 | False | 0 |  |
| mjc | ok | 84.8 | 180.0 (172.0) | 39.9 | False | 0 |  |
| mjx_triton | ok | 24.1 | 2302.0 (180.0) | 363.86 | False | 0 |  |
| rep2_ours | ok | 27.1 | 2415.0 (2272.0) | 348.18 | False | 0 |  |
| rep2_mjw_native | ok | 12.8 | 2422.0 (2310.0) | 262.56 | False | 0 |  |
| rep2_mjw_pr1535 | ok | 14.1 | 2422.0 (2310.0) | 269.6 | False | 0 |  |
| rep2_mjw_pr1535_fix | ok | 14.1 | 2422.0 (2310.0) | 267.98 | False | 0 |  |
| rep2_mjx | ok | 26.2 | 2347.0 (2302.0) | 362.63 | False | 0 |  |
| rep2_mjc | ok | 4.9 | 180.0 (180.0) | 39.41 | False | 0 |  |
| rep3_ours | ok | 27.2 | 2415.0 (2280.0) | 347.94 | False | 0 |  |
| rep3_mjw_native | ok | 12.9 | 2422.0 (2310.0) | 252.92 | False | 0 |  |
| rep3_mjw_pr1535 | ok | 14.1 | 2422.0 (2310.0) | 268.98 | False | 0 |  |
| rep3_mjw_pr1535_fix | ok | 14.1 | 2422.0 (2310.0) | 271.46 | False | 0 |  |
| rep3_mjx | ok | 26.2 | 2310.0 (2302.0) | 365.02 | False | 0 |  |
| rep3_mjc | ok | 4.9 | 180.0 (180.0) | 38.43 | False | 0 |  |
