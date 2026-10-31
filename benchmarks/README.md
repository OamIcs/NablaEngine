# NablaEngine benchmark

This directory holds the benchmark of NablaEngine against MuJoCo (C), MJX and MuJoCo Warp on one
physics problem: one control step of the Unitree G1 humanoid with per-world domain randomization, the step a
differentiable-simulation rollout repeats. It times the forward step and every derivative an engine offers natively,
and it scores forward outputs and gradients against an engine-independent MuJoCo C float64 truth. It is not part of
the `nablaengine` package; each engine runs in its own Python environment (`envs/`).

The timing tables of the paper come from `scripts/paper_bench` (one job per target GPU; see its README). The accuracy
tables of the paper come from this harness: the `accuracy` part's dumps (plus MuJoCo C FD dumps at the step chosen by
`bench/accuracy/fd_sweep.py`), scored against `data/truth/` with `bench/accuracy/` and written as one JSON with its
markdown tables by `python -m benchmarks.bench.report.paper_accuracy` (its docstring lists the sources and options);
`python -m benchmarks.bench report` renders the summary tables of any job.

## What is compared

| Engine id | Engine | Kind | Timed forward (`fwd`) | Derivative, as the engine provides it |
|---|---|---|---|---|
| `dk` | NablaEngine through its JAX interface (`nablaengine.jax_ffi`), float32, exact gradient mode (`clamp_kv=1, normal=0`) | ours | `jax_ffi.step`, AOT-compiled `jax.jit` | analytic adjoint: `fwd_train` (forward with tape), `bwd`, `vjp` |
| `dk_jac` | the same engine, Jacobian-equivalent | ours | - | `jac`: 71 VJPs of the unit output cotangents (the full 71 x 99 Jacobian) |
| `dk_iface` | the same engine in the gradient mode of `jax_ffi`'s interface defaults (`clamp_kv=0, normal=3`) | labelled | as `dk` | as `dk` |
| `mjw` | MuJoCo Warp 3.17.0 from PyPI, public API (`put_model`, `put_data`, `step`) | native | CUDA graph of [copy the inputs into `Data`; substeps x `mjw.step`] | none: no MuJoCo Warp release has a backward (its README: "Differentiability via Warp is not yet available", issue #500); `bwd`, `vjp`, `jac` are reported as not supported, and every unit records what 3.17.0 does under `wp.Tape` (no error, an exactly zero contribution) |
| `mjw_pr1535` | MuJoCo Warp with the unmerged upstream pull request google-deepmind/mujoco_warp#1535 ("hybrid analytic differentiability"): etaoxing/mujoco_warp at 357a75d, unmodified | unmerged-upstream | as `mjw`, after `mjw.enable_grad()` | the PR's adjoint, driven as its own `contrib/diffsim` rollout does: `bwd` recomputes the step under `wp.Tape` and runs `tape.backward`; `vjp` = `fwd` + `bwd`; no Jacobian API |
| `mjw_pr1535_fix` | 357a75d + etaoxing/mujoco_warp#1 (4af093f), unmodified: two public fixes of the PR's adjoint | labelled | as `mjw_pr1535` | as `mjw_pr1535` |
| `mjx` | MuJoCo MJX 3.17.0 (`mujoco-mjx`, `impl="jax"`) | native | `jax.jit(jax.vmap(...))` of a `lax.scan` over `mjx.step`, AOT-compiled | forward mode only: `jac` = `jax.jvp` along the 99 input directions; reverse mode (`bwd`, `vjp`) is not supported: `jax.vjp` of the step fails (the solver's `lax.while_loop` cannot be transposed) and the exact exception is recorded |
| `mjc` | MuJoCo C 3.17.0 (`mujoco` wheel), float64, CPU | native | `mujoco.batch.Batch.step` on the process's CPUs | finite differences: `jac` = `mjd_transitionFD` chained over the substeps; its VJP is `c^T J` |
| `genesis` | Genesis 1.4.5 (`genesis-world` from PyPI, quadrants 1.3.3), float32, its rigid solver; Track A only. Its MJCF importer reads no `<pair>`, so the model's contacts are declared through contype / conaffinity (Genesis's default filter then gives the same 502 pairs); every other deviation is recorded by the adapter (`adapters/genesis.py`) | native | `set_qpos`, `set_dofs_velocity`, `control_dofs_position`, `scene.step()` (4 substeps) | its native reverse mode (`requires_grad=True`, `scene.backward`), with respect to the initial qvel only and through the `approximate_implicitfast` integrator (the only one it differentiates): the accuracy dumps' qvel block; qpos and ctrl blocks and the ops `bwd`, `vjp`, `jac` are not supported |

Ops: `fwd` (one control step), `fwd_train` (the forward a backward needs), `bwd` (the backward alone, from that
forward's residuals), `vjp` (forward and backward of one cotangent), `jac` (the full Jacobian of the control step,
rows qpos' then qvel', columns qpos tangent, qvel, ctrl). An op an engine has no native way to do is recorded as
`not_supported` with the engine's own reason, never emulated. "labelled" rows are configurations other than an
engine's native or default one; they are marked and are never a headline row.

## Fairness statement

- **One problem for every engine.** The same model file, solver (Newton), iteration count (set explicitly by every
  cell: 10 for the headline, 50 for reference), line-search iterations (8), tolerance (1e-8), substeps (4 x 5 ms,
  control held), sensors off, the same corpus states as inputs and the same per-world foot friction and torso
  centre-of-mass offset (domain randomization) from the corpus. An engine's internal deviation from these options is
  recorded, not corrected (MuJoCo Warp raises the tolerance to at least 1e-6 internally).
- **Other engines run natively.** Official releases or published upstream code, unmodified; public APIs; documented
  settings; no monkeypatches; none of our optimisations. The only choices an adapter makes are the problem's and
  documented user settings, recorded with every cell: MuJoCo Warp's contact and constraint capacity (sized by a
  documented procedure from measurements on these corpora, verified overflow-free; overflow is counted, never
  hidden), MuJoCo C's thread count (the CPUs the process may use), MJX's XLA flags (default; the Triton variant is a
  labelled row). Each adapter refuses to run on a source tree other than its own (package version or source hash).
- **Derivatives as the engines provide them.** No backward is built for an engine that has none. Gradient accuracy is
  compared on the object each engine produces: VJPs of stored cotangents, or the full Jacobian (op `jac`: MuJoCo C's
  finite differences, MJX's forward mode and NablaEngine's 71 VJPs give the same 71 x 99 matrix).
- **An engine-independent truth.** Forward truth from MuJoCo C in float64 at 2000 iterations and tolerance 1e-13;
  gradient truth from central finite differences of the whole control step in MuJoCo C float64, with the truth's own
  floor; state groups from MuJoCo C alone; a model check against native MuJoCo Warp 3.17.0 marks the states on which
  the two contact models disagree. Nothing in the truth comes from NablaEngine or from a modified MuJoCo
  Warp (`data/truth/README.md`).
- **Held-out inputs.** The corpora (`data/`) are mocap-free MuJoCo C rollouts, generated before any engine was scored
  on them and never used for tuning.
- **Timing.** One fresh process per (engine, problem, nworld); setup, compilation and graph capture timed
  separately; inputs resident on the device and every call starting from the corpus states (no settling); warm-up,
  then rounds of back-to-back calls with one synchronisation (throughput) and of individually synchronised calls
  (latency), the first round dropped, medians reported with p10 / p90; headline cells in three processes; GPU clocks,
  power, temperature and throttle reasons sampled alongside, hardware throttling flagged; speed-ups only against the
  dk anchor cell of the same job, on the same machine. Results record the GPU, driver, CPU, package versions and wheel
  tags, and other compute processes on the GPU.

## Problems

| Track | Model | Control step | Engines |
|---|---|---|---|
| A (headline) | the mjlab G1, `fixtures/mjlab/g1_train.mjb` | 4 x 5 ms, ctrl held, Newton 10 iterations (50 reference), ls 8, tolerance 1e-8, sensors off, per-world foot friction and torso COM randomization | all |
| B | MuJoCo Menagerie `unitree_g1/scene_mjx.xml` at a pinned commit (fetched, not vendored) | 5 x 4 ms, the model's native options, no randomization | MJX, MuJoCo Warp, MuJoCo C |

`bench/problem.py` defines both, the randomization, the corpus loader and the seeded state selection (subsets of a
corpus are disjoint; small batches are averaged over several subsets). Data: `data/README.md` (the corpora
`mf_s0`..`mf_s3` for Track A and `menagerie_s0` for Track B) and `data/truth/README.md` (their truth files).

## Setup

From the repository root, in the NablaEngine environment (`scripts/setup_env.sh`, then `source <env>/activate.sh`):

```
benchmarks/envs/setup.sh                                   # mjw-native and mjw-pr environments (envs/README.md)
python -m benchmarks.bench fetch-models                    # Track B model (MuJoCo Menagerie at the pinned commit)
export CUDA_VISIBLE_DEVICES=<one GPU>                      # one exclusive GPU per driver
python -m benchmarks.bench env-check                       # machine, sampler, every engine's environment
```

| Environment | Engines |
|---|---|
| `dk`: nablaengine, mujoco 3.17.0, mujoco-mjx 3.17.0, JAX, warp-lang 1.18.0 | `dk`, `dk_iface`, `dk_jac`, `mjx`, `mjc` |
| `mjw-native`: PyPI mujoco-warp 3.17.0, mujoco 3.17.0, warp-lang 1.18.0 | `mjw` |
| `mjw-pr`: mujoco 3.17.0, warp-lang 1.18.0, the two unmodified pull-request trees on `PYTHONPATH` | `mjw_pr1535`, `mjw_pr1535_fix` |
| `genesis` (`setup.sh --envs genesis`): PyTorch 2.14.1, genesis-world 1.4.5, quadrants 1.3.3, mujoco 3.17.0 | `genesis` |

On CUDA 12 drivers and on Volta GPUs, `setup.sh` installs NVIDIA's `warp-lang 1.18.0+cu12` wheel (the default wheel
silently falls back to the CPU there); the installed tag is recorded in every result.

## Running

```
python -m benchmarks.bench list                            # engines, environments and the job parts
python -m benchmarks.bench run --machine <tag> --part gpu-fast,mjx,cpu,accuracy,trackB --out runs/bench/<tag>/
python -m benchmarks.bench.accuracy.job runs/bench/<tag>/<job>.json --write    # score the dumps against the truth
python -m benchmarks.bench report runs/bench/<tag>/ --out runs/bench/<tag>/    # report.md: timing and accuracy
```

- Parts (each at most about an hour on one GPU; `list` shows them): `gpu-fast` (dk, mjw, mjw_pr1535, mjw_pr1535_fix
  over nworld 64-16384, both iteration counts, all four Track A corpora; dk_jac), `mjx`, `cpu` (MuJoCo C), `accuracy`
  (raw outputs and gradients of every engine on every Track A corpus at 4096 worlds, 10 and 50 iterations), `trackB`.
- `--engines`, `--models`, `--corpora`, `--nworld`, `--iterations`, `--ops`, `--variant ENGINE=V1,V2`,
  `--opt ENGINE.KEY=VALUE` narrow or change a run; `--dry-run` prints the plan. `--add-engine E=REF` runs an
  additional adapter E wherever the parts run engine REF.
- `--quick` is a smoke test: nworld 256 and 4096, one corpus and the first iteration count per block, one process and
  one subset, 3 rounds, op `jac` only at 256 worlds, 600 s per unit. `python -m benchmarks.bench run --engines null
  --quick --out runs/bench/test_null/` exercises the whole driver without any engine.
- A job writes `<out>/<job>.json` after every unit; rerunning the same command resumes it (`--fresh` starts over,
  `--retry-errors` re-runs failed cells). `--out` never overwrites a file tracked by git. The exit status is 1 when a
  cell ended in an error or over its time budget.
- `python -m benchmarks.bench validate FILE...` checks result files against the schema (`bench/schema.py`, dk-bench/1;
  repository-relative paths only, no host or user names).
- Other commands: `truth` (make, verify and index the truth files), `gen-corpus` (the corpus generator),
  `fetch-models`; `python -m benchmarks.bench.adapters.mjw_capacity` remeasures MuJoCo Warp's capacity,
  `python -m benchmarks.bench.adapters.mjx_check` checks the MJX adapter against MuJoCo C.

## Timing protocol (summary; `bench/protocol.py` and `bench/worker.py` have the details)

One fresh process per (engine, model, nworld) unit; setup, compile and graph capture timed separately; warm-up of at
least 5 calls and 1 s; 5 rounds of N back-to-back calls with one sync (throughput) and of N individually synchronised
calls (latency), N chosen from the warm-up (at least 0.2 s and 20 calls per round on a GPU); round 0 dropped; 3
processes for the headline cells (mjlab, 4096 worlds, 10 iterations, mf_s0 and mf_s1); small batches averaged over
disjoint state subsets; engines interleaved round-robin; inputs resident on the device, every call from the corpus
state; GPU clocks, power, temperature and throttle reasons sampled every 100 ms by a separate `nvidia-smi` process; a
dk anchor cell in every job. Engines whose single call takes seconds (MuJoCo C's and MJX's Jacobians at large
batches) run one call per round and 3 rounds, recorded as the cell's timing hints.

## Layout

```
benchmarks/
  README.md            this file
  envs/                per-engine Python environments: envs.toml (engine -> interpreter), setup.sh, requirements
  data/                committed inputs: the corpora, their statistics, MuJoCo Warp's capacity file, truth/
  bench/               the harness (python -m benchmarks.bench)
    cli.py, __main__.py  command line
    protocol.py        the driver: plan, one fresh worker process per unit, timing protocol, clock sampler, resume
    worker.py          runs one unit inside the engine's interpreter
    schema.py          result schema dk-bench/1 (validated on write and read)
    prov.py            provenance: GPU, driver, CPU, packages and wheel tags, commit
    problem.py         physics problems, domain randomization, corpora and state selection
    adapters/          one module per engine; base.py documents the interface, null.py is a self-test stand-in
    truth/             the MuJoCo C float64 truth and the native MuJoCo Warp model check
    accuracy/          forward and gradient scorers against the truth, MuJoCo C finite-difference step sweep
    corpus/, models/   corpus generator, format and statistics; the Track B model fetcher (pins.json)
    report/            markdown tables from result files
```

## Writing an adapter

Add `benchmarks/bench/adapters/<name>.py` with `ADAPTERS = {"<engine id>": <class>}`; the interface and the rules
(native settings only, enqueue-only ops, NotSupported with the engine's own message) are in
`benchmarks/bench/adapters/base.py`. Map the engine to an environment in `benchmarks/envs/envs.toml` (or in another
`benchmarks/envs/*.toml`, merged over it), and add it to a run with `--engines` or `--add-engine`.

## Environment variables

None of these changes results; they locate files and interpreters.

| Variable | Default | Meaning |
|---|---|---|
| `DK_BENCH_ENV_PREFIX` | `runs/bench/envs` | where `envs/setup.sh` creates the environments and `envs.toml`'s `{prefix}` points |
| `DK_BENCH_ENVS` | `runs/bench/envs.local.toml` (if present) | a local, untracked TOML merged over `envs/envs.toml` |
| `DK_BENCH_PY_<ENV>` | unset | the interpreter of environment `<ENV>` (upper case, `-` as `_`, e.g. `DK_BENCH_PY_MJW_NATIVE`) |
| `DK_BENCH_DATA` | `benchmarks/data` | the benchmark corpora (and `truth/` below it) |
| `DK_BENCH_FIXTURE_DIR` | unset | a directory of corpora in the engine's fixture format (`<name>_fwd.npz`, `<name>_adj.npz`), looked up by name after `DK_BENCH_DATA` |
| `DK_BENCH_MODELS` | `runs/bench/models` | the fetched Track B models |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | set to `false` for the workers unless already set | JAX memory figures |
