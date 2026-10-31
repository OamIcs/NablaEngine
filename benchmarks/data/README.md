# Benchmark corpora

Committed inputs of the benchmark: independent G1 states, each the input of one control step, with stored output
cotangents. All corpora here are **mocap-free**: they come from MuJoCo C rollouts of a PD-controlled G1 with random
setpoints and pushes (no motion capture, no learned controller). They were generated on 2026-10-28 before any engine
was scored on them and were never used for tuning.

| File | Track, model | States | Bytes | sha256 (file) | Content digest |
|---|---|---|---|---|---|
| `mf_s0.npz` | A, `fixtures/mjlab/g1_train.mjb` | 4096 | 5372366 | `d78930ce81c4d4ad3d0fd83ad93f7a393781e8227ed64e291749a85c45a45ef8` | `9bba992c0de22d4c47ee16c8608225476aa2c38599f7fe2ac6875c177aa0689c` |
| `mf_s1.npz` | A, same | 4096 | 5366865 | `83632a500acf6fb6d283e5283b4576879f310c2cb3be5bfb731b999625a1b960` | `58e672c5523e811126e9bcc189827a0a57d746a480cee3495ff4abe7403cc736` |
| `mf_s2.npz` | A, same | 4096 | 5369375 | `8f6c0a0eb0a183838062e6b63ff8831c7f92c6572574155a096a178c775c3cf3` | `8e9bcfe47c08848bb0889eef865372d4ee59912e9510d3a91e946bb02b443628` |
| `mf_s3.npz` | A, same | 4096 | 5371834 | `6a79d39e87b5523719d7ae3bfc53ae04169306b3412a6b566191cc256bdc109d` | `a6c3eaab6c3676a76d1c0d00ba0ecced7a0faea22d03dbedd24e7dcf722410ab` |
| `menagerie_s0.npz` | B, MuJoCo Menagerie `unitree_g1/scene_mjx.xml` @ f3b6013 | 4096 | 5308170 | `b1b440a1f0b81f3ecdb49ae65e8010eb56fd21dd9da6696b7d500b406d3433b9` | `92476fa3da1005327d7348fd40f69577786e4bb42c17e474d37ed6a2a44eb323` |

The content digest (`corpus_format.content_digest`) hashes the arrays themselves and does not depend on how the zip
was compressed. Also here:

- `corpora.json`: per corpus the file and content hashes, the generator command, its counts and where it ran
  (written by the generator). Its `model.meta_sha256`, also in each corpus's `metadata`, is that of
  `fixtures/mjlab/g1_meta.json` when the corpora were generated; only the file's free-text `source` entry has changed
  since, so the sha256 of the current file differs;
- `corpus_stats.json`: the distribution statistics below, with histograms (written by `corpus/stats.py`);
- `mjw_capacity.json`: the contact and constraint-row capacity MuJoCo Warp needs on these corpora (written by
  `bench/adapters/mjw_capacity.py`, read by the MuJoCo Warp adapters);
- `truth/`: the MuJoCo C float64 truth of every corpus (`truth/README.md`).

## Format

One `.npz` per corpus (`benchmarks/bench/corpus/format.py`), S = 4096 states, nq 36, nv 35, nu 29:

| Key | Shape, dtype | Content |
|---|---|---|
| `qpos`, `qvel`, `qacc_warmstart`, `ctrl` | (S, nq), (S, nv), (S, nv), (S, nu) float32 | inputs of one control step |
| `friction`, `com` | (S,), (S, 3) float32 | Track A only: foot-floor pair friction and torso COM offset (m) of the state |
| `cot_qpos`, `cot_qvel` | (S, 3, nq), (S, 3, nv) float32 | three N(0, 1) output cotangents per state, stored (not regenerated) |
| `metadata` | 0-d str | JSON: format `dk-bench-corpus/1`, generator script and version, seed, options, model (path or pinned repository file, sha256, sizes), physics options, DR, reset and termination constants, label descriptions, counts, mujoco and numpy versions |
| `label_*` | (S,) bool or int32 | per-state labels, below |

The arrays are stored in a fixed seeded permutation of the sampled states. Inputs are float32; float64 engines read
them exactly. Read with `np.load(path)` (no pickle), `benchmarks.bench.corpus.format.read(path)` or the harness's
`benchmarks.bench.problem.load_corpus(name)`.

Labels. From the rollout: `label_prefall` (within the last 25 control steps of an episode up to its fall),
`label_pick` (0 sampled as pre-fall, 1 for a self contact, 2 uniform), `label_rollout_step` (control steps since the
episode's reset), `label_rollout_index` (step * 4096 + world), `label_keyframe` (start pose). From one MuJoCo C float64
control step of the stored inputs at 10 iterations (`corpus/labels.py`): `label_ncon_max`, `label_nefc_max`,
`label_niter_max` (most contacts / constraint rows / solver iterations in any substep), `label_self_contact` (a contact
between two robot geoms), `label_clamped` (an actuated joint's force outside its actuator force range),
`label_limit` (a joint-limit row), `label_nonconv` (solver iterations at the limit), `label_over_capacity` (more than
64 contacts or 192 rows, NablaEngine's capacity), each "in some substep". These are engine-neutral
approximations of the groups G6, G4, G5 and G9; the gradient truth defines its own groups.

## How they were generated

`benchmarks/bench/corpus/gen_mf.py` (version 1; the module docstring has every detail):

- **Rollout.** 4096 worlds x 150 control steps in MuJoCo C float64, `mujoco.batch.Batch` (bit-identical at any
  thread count), Newton at 10 iterations, ls_iterations 8, tolerance 1e-8, sensors off. Track A: 4 x 5 ms substeps per
  control step; Track B: 5 x 4 ms (the scene's timestep).
- **Reset.** A keyframe (Track A: `home` or `knees_bent`, Track B: `home`) plus reset noise (base position,
  orientation, base velocity, joint positions within 90% of the range, qacc_warmstart 0). Track A also draws a
  per-episode domain randomization: foot-floor friction U[0.3, 1.2], torso COM offset U(+-(0.025, 0.05, 0.05)) m.
  Track B has no DR.
- **Setpoints.** The keyframe's joint pose plus a per-joint Ornstein-Uhlenbeck process (stationary standard deviation
  0.75 x action_scale, action_scale = 0.25 effort / kp; correlation time 0.4 s). About once per second a jump moves
  each joint with probability 0.2 to a setpoint uniform in 90% of its range, from where the process relaxes back.
  Setpoints stay within 90% of the joint range and the ctrlrange.
- **Pushes.** Every 0.25-1 s the base velocity gets U(+-1.5 x (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)) (m/s, rad/s).
- **Falls.** A termination against the episode's start keyframe (base height off by more than 0.25 m, or gravity-z in
  the base frame off by more than 0.8), or a MuJoCo bad-state warning; the world restarts with a new reset. Open
  loop, the robot falls within 0.73 s on average (about 14 500 falls per corpus; no bad states occurred).
- **Sampling.** Up to half of the corpus from the last 25 control steps before a fall, then states with a self contact
  until at least 15% of the corpus has one (the pre-fall half already had more, so none were added), the rest uniform
  over all 614 400 (world, step) pairs. No capacity filter. Inputs rounded to float32; labels computed from the
  rounded inputs.

The setpoint and push settings were chosen on small seed-0 runs, before any engine ran on these corpora, by comparing
the distribution statistics below with reference corpora of the same model that are not part of this repository
(`corpus/stats.py --ref`): with jumps that move every joint the
|qvel| p99 was 52-56, without jumps 9.7 with almost no clamped actuators.

Commands (from the repository root, in the `dk` environment):

```
python -m benchmarks.bench.models.fetch                                   # Track B model, verified against pins.json
for s in 0 1 2 3; do python -m benchmarks.bench.corpus.gen_mf --track A --seed $s --threads 24; done
python -m benchmarks.bench.corpus.gen_mf --track B --seed 0 --threads 24
python -m benchmarks.bench.corpus.stats --threads 16 --out benchmarks/data/corpus_stats.json --force --table
```

Each corpus takes about 20-30 s on 24 threads of an Intel Xeon E5-2690 v3 (mujoco 3.17.0, numpy 2.5.4, Python 3.12).
The generator refuses to overwrite an existing corpus without `--force`. Re-running it with 7, 11 or 13 threads,
another output directory or another models directory reproduced the files bit for bit (same sha256), and the
statistics are the same at 16 and 24 threads. Another CPU, libm or mujoco build may differ in the last bits, which 150
chaotic control steps amplify: then the generator reproduces the distribution (check with `corpus/stats.py`) and the
sha256 tells whether the bits matched too.

Environment variables (all optional, none changes the corpus bits): `DK_BENCH_DATA` (default `benchmarks/data`),
where the generator writes and the statistics and the harness read corpora; `DK_BENCH_MODELS` (default
`runs/bench/models`), where `benchmarks/bench/models/fetch.py` puts the Track B model.

## Distribution statistics

Every state of every corpus runs one control step in MuJoCo C float64 with its own DR (`corpus/labels.py`; the same
code for all corpora). "Per world" is the most in any substep; |qvel| is the norm of the input qvel
(`corpus_stats.json`, 2026-10-29).

| | `mf_s0` | `mf_s1` | `mf_s2` | `mf_s3` | `menagerie_s0` |
|---|---|---|---|---|---|
| kind | benchmark | benchmark | benchmark | benchmark | benchmark, Track B |
| states | 4096 | 4096 | 4096 | 4096 | 4096 |
| contacts / world, mean (it10) | 7.97 | 8.26 | 7.86 | 8.08 | 3.58 |
| contacts / world, p99 / max | 23 / 29 | 23 / 30 | 23 / 29 | 23 / 30 | 9 / 13 |
| contacts at the input state, mean | 6.83 | 7.03 | 6.80 | 6.89 | 2.94 |
| rows / world, mean | 30.7 | 31.9 | 30.3 | 31.3 | 42.1 |
| rows / world, p99 / max | 90 / 114 | 91 / 114 | 90 / 113 | 88 / 114 | 65 / 77 |
| solver iterations at 10, mean / p90 | 4.44 / 7 | 4.48 / 7 | 4.38 / 7 | 4.48 / 7 | 5.53 / 7 |
| at the limit of 10 | 0.3% | 0.4% | 0.2% | 0.6% | 0.5% |
| solver iterations at 50, mean / p99 / max | 4.45 / 9 / 12 | 4.48 / 9 / 12 | 4.38 / 8 / 10 | 4.48 / 9 / 12 | 5.53 / 9 / 12 |
| at the limit of 50 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| \|qvel\|, p50 / p99 / max | 9.03 / 30.43 / 57.66 | 8.91 / 29.12 / 53.97 | 8.96 / 30.76 / 48.43 | 8.99 / 30.84 / 51.26 | 7.81 / 43.35 / 70.98 |
| pre-fall | 79.2% | 78.0% | 78.0% | 79.1% | 78.7% |
| self contact | 44.3% | 44.7% | 44.6% | 43.7% | 40.0% |
| clamped actuator | 3.1% | 3.5% | 3.7% | 3.2% | 10.3% |
| joint-limit row | 41.7% | 42.6% | 39.4% | 41.1% | 26.2% |
| > 64 contacts or > 192 rows | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ctrl entries outside the ctrlrange | 0 | 0 | 0 | 0 | 0 |

Reading:

- The four Track A corpora are statistically alike (four seeds of one generator): about 79% of the states lie within
  25 control steps before a fall, about 44% have a self contact, 39-43% a joint-limit row and 3-4% a clamped actuator.
- No state of any corpus exceeds 64 contacts or 192 rows in MuJoCo C (at most 30 and 114), so the "no capacity
  filter" rule kept nothing that a filter would have dropped. MuJoCo Warp and NablaEngine count contacts by
  their own collision code; the harness reports their overflow.
- No ctrl lies outside the ctrlrange.
- `menagerie_s0` is another model (box feet, 49 pairs, frictionloss rows on every dof: more rows from fewer contacts).

## Models and licences

Track A uses the mjlab G1 of this repository (`fixtures/mjlab/g1_train.mjb`, sha256 `757b0997...02ef379`; see
`fixtures/mjlab/README.md` for its provenance and licence). Track B uses MuJoCo Menagerie's Unitree G1 MJX scene, fetched
(not vendored) at commit f3b6013629c7b94f8ee93fd977b161fa8350f9bc by `python -m benchmarks.bench.models.fetch`, which
checks the sha256 of all 38 files in `benchmarks/bench/models/pins.json` and that MuJoCo compiles the scene; the model
is under its own licence,
[unitree_g1/LICENSE](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1/LICENSE)
(BSD-3-Clause, Unitree Robotics). `pins.json` records why this commit rather than mujoco_playground's Menagerie pin
1b86ece. The corpora contain only simulated states, controls and random cotangents.
