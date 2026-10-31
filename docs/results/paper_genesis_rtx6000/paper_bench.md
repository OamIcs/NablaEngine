# Paper timing kit: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.58.03)

- started 2026-10-29T22:36:48Z, finished 2026-10-29T22:41:19Z, 271.8 s in all; setup: skipped
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
| Genesis 1.4.5 (native) | forward (state load + scene.step(), 4 substeps) | 13.454 [13.450-13.460] | 13.566 [13.558-13.572] |  |
| Genesis 1.4.5 (native) | forward, gs.init(performance_mode=True) | 9.502 [9.498-10.082] | 9.623 [9.618-10.246] |  |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| Genesis 1.4.5 (native) | forward | mf_s0 | 3 | 13.454, 13.399, 13.452 | 13.452 | 0.4% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s0 | 3 | 9.502, 9.504, 9.530 | 9.504 | 0.3% |
| Genesis 1.4.5 (native) | forward | mf_s1 | 3 | 13.566, 13.591, 13.628 | 13.591 | 0.5% |
| Genesis 1.4.5 (native) | forward, performance_mode=True | mf_s1 | 3 | 9.623, 9.945, 9.921 | 9.921 | 3.3% |

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| genesis | ok | 139.1 | 2302.0 (180.0) | 243.95 | False | 0 |  |
| rep2_genesis | ok | 65.7 | 2302.0 (210.0) | 242.89 | False | 0 |  |
| rep3_genesis | ok | 66.0 | 2302.0 (637.0) | 242.89 | False | 0 |  |
