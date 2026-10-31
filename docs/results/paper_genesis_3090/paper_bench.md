# Paper timing kit: NVIDIA GeForce RTX 3090 (driver 565.57.01)

- started 2026-10-29T19:56:38Z, finished 2026-10-29T20:17:59Z, 1281.1 s in all; setup: skipped
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
| Genesis 1.4.5 (native) | forward (state load + scene.step(), 4 substeps) | 38.875 [38.705-40.626] | 37.233 [37.121-37.430] |  |
| Genesis 1.4.5 (native) | forward, gs.init(performance_mode=True) | 31.582 [31.465-33.245] | 32.481 [32.341-33.957] |  |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| Genesis 1.4.5 (native) | forward | mf_s0 | 3 | 38.875, 36.700, 36.732 | 36.732 | 5.9% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s0 | 3 | 31.582, 31.479, 30.764 | 31.479 | 2.7% |
| Genesis 1.4.5 (native) | forward | mf_s1 | 3 | 37.233, 37.267, 39.390 | 37.267 | 5.8% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s1 | 3 | 32.481, 31.324, 30.844 | 31.324 | 5.3% |

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| genesis | ok | 797.8 | 1890.0 (1695.0) | 301.37 | False | 0 |  |
| rep2_genesis | ok | 246.9 | 1890.0 (1695.0) | 305.77 | False | 0 |  |
| rep3_genesis | ok | 233.4 | 1890.0 (1695.0) | 309.02 | False | 0 |  |
