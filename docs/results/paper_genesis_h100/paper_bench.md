# Paper timing kit: NVIDIA H100 80GB HBM3 (driver 595.91.07)

- started 2026-10-29T14:43:23Z, finished 2026-10-29T14:49:30Z, 367.1 s in all; setup: skipped
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
| Genesis 1.4.5 (native) | forward (state load + scene.step(), 4 substeps) | 16.435 [16.427-16.437] | 16.477 [16.467-16.498] |  |
| Genesis 1.4.5 (native) | forward, gs.init(performance_mode=True) | 11.976 [11.963-12.406] | 12.107 [12.099-12.545] |  |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| Genesis 1.4.5 (native) | forward | mf_s0 | 3 | 16.435, 16.319, 16.364 | 16.364 | 0.7% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s0 | 3 | 11.976, 11.990, 12.025 | 11.990 | 0.4% |
| Genesis 1.4.5 (native) | forward | mf_s1 | 3 | 16.477, 16.476, 16.505 | 16.477 | 0.2% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s1 | 3 | 12.107, 12.086, 12.123 | 12.107 | 0.3% |

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| genesis | ok | 183.2 | 1980.0 (1980.0) | 248.08 | False | 0 |  |
| rep2_genesis | ok | 91.4 | 1980.0 (1980.0) | 248.61 | False | 0 |  |
| rep3_genesis | ok | 91.1 | 1980.0 (1980.0) | 246.91 | False | 0 |  |
