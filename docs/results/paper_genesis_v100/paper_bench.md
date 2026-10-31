# Paper timing kit: Tesla V100-SXM2-16GB (driver 550.90.07)

- started 2026-10-29T13:19:02Z, finished 2026-10-29T13:25:59Z, 417.0 s in all; setup: skipped
- NablaEngine 0.1.0 commit: a revision before this repository's history: not recorded; nvcc None; arch sm_70
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
| Genesis 1.4.5 (native) | forward (state load + scene.step(), 4 substeps) | 139.104 [139.086-139.128] | 143.610 [143.529-143.645] |  |
| Genesis 1.4.5 (native) | forward, gs.init(performance_mode=True) | 82.371 [82.348-82.410] | 86.418 [86.387-86.431] |  |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| Genesis 1.4.5 (native) | forward | mf_s0 | 3 | 139.104, 139.111, 139.116 | 139.111 | 0.0% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s0 | 3 | 82.371, 82.368, 82.349 | 82.368 | 0.0% |
| Genesis 1.4.5 (native) | forward | mf_s1 | 3 | 143.610, 143.533, 143.535 | 143.535 | 0.1% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s1 | 3 | 86.418, 86.420, 86.393 | 86.418 | 0.0% |

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| genesis | ok | 176.1 | 1530.0 (1312.0) | 113.96 | False | 0 |  |
| rep2_genesis | ok | 119.3 | 1530.0 (1312.0) | 114.45 | False | 0 |  |
| rep3_genesis | ok | 120.0 | 1530.0 (1312.0) | 114.94 | False | 0 |  |
