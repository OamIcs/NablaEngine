# Paper timing kit

One command, one job on one GPU, at most 55 minutes: every timing of the paper's engine comparison on the mjlab
Unitree G1, with each other engine run as natively as it can be (official releases, public APIs, documented settings,
none of our optimisations; no backward is built for an engine that has none). It was written for the target cards
(RTX PRO 6000, sm_120; H100, sm_90) and runs on any CUDA GPU that the engine supports. The paper's accuracy tables
come from the benchmark harness instead (`benchmarks/README.md`, `benchmarks/bench/report/paper_accuracy.py`).

```
scripts/paper_bench/run.sh <out_dir>
```

writes `<out_dir>/paper_bench.json` (every part's status, time, GPU samples, its result embedded, the headline table and
the provenance), `<out_dir>/paper_bench.md` (the summary tables), `<out_dir>/parts/<part>.json`,
`<out_dir>/logs/<part>.log` and `<out_dir>/logs/<part>.smi.csv` (nvidia-smi samples). The exit status is 0 when every
part and the setup succeeded and 1 otherwise; a failing part is recorded with its error and the job goes on. The
JSON and markdown files have host paths scrubbed; `logs/` and `parts/raw/` are raw tool output (they can contain host
paths): keep them local and share only `paper_bench.json`, `paper_bench.md` and `parts/*.json`. An out-of-memory
failure is recorded as `skipped: oom`; at 4096 worlds or fewer (which fit on every target card) it also counts as a
failure of the part.

## What is measured

Common to every engine: the mjlab G1 (`fixtures/mjlab/g1_train.mjb`), one control step = 4 substeps of 5 ms with the
control held, Newton with 10 iterations (the model file stores 5), sensors disabled, per-world domain randomisation
(foot-floor pair friction, torso centre-of-mass offset), 4096 worlds unless stated. Inputs are the same corpus states
for every engine: the benchmark corpora `mf_s0` and `mf_s1` (`benchmarks/data/<name>.npz`, 4096 states each; a
missing one is flagged in the JSON and the summary; `PAPER_BENCH_CORPORA` selects among `mf_s0` .. `mf_s3`). The first
corpus is the corpus of the nworld sweep, the Jacobians and the adjoint split. No settling for any engine: every timed
call starts from the corpus states. Compile / capture / build times are recorded separately. Timing protocol
(`common.py`): warm-up of at least 5 calls and 1 s, then 6 rounds of N back-to-back calls and one synchronisation,
round 0 dropped; median and p10-p90 over the 5 kept rounds (ms per control step), plus the median latency of single
synchronised calls for the GPU engines where the rounds are not single calls already (N > 1). A separate
`nvidia-smi -lms 100` process samples SM clock, power, temperature and the clock-event (throttle) reasons during each
part; only the hardware slowdown reasons are flagged. Other compute processes on the GPU are recorded per part
(contention).

| part | engine and label | configuration |
|---|---|---|
| `ours` | ours: `nablaengine.jax_ffi` under `jax.jit` | `fwd` (step, no tape), `fwd_tape` (forward with tape), `bwd` (adjoint on the tape), `vjp` (`jax.vjp` + pullback, one jit); gradient modes *exact* (`clamp_kv=1, normal=0`, headline) and *interface* (`clamp_kv=0, normal=3`, the interface defaults); Jacobian-equivalent: 71 backward calls on one tape (one per output row of the 71 x 99 Jacobian; `lax.scan`, one jit), alone and with the forward (the qpos part comes out in qpos coordinates, 36 per world; its map to the 35 tangent directions, one small matrix product per world, is not included); `jax_ffi.init()` build info; nworld sweep of `fwd` / `vjp` |
| `ours_bare` | ours, bare kernels (attribution only; on the first corpus) | `scripts/engine_step_check.py` forward binary tape off / on, the adjoint check binary of `nablaengine.checks` (exact and interface variants); CUDA-event device time |
| `mjw_native`, `mjw_native_nw*` | "MuJoCo Warp 3.17.0" (PyPI `mujoco-warp==3.17.0`, `mujoco==3.17.0`, `warp-lang==1.18.0`) | CUDA graph of [input copies + 4 x `mjw.step`], `graph_conditional` as `put_model` sets it, ITERATIONS / LS_ITERATIONS overflow warnings masked as in upstream's own G1 benchmark, per-world DR via `put_model(batch_sizes=...)`; three capacities: *procedure* (the documented sizing rule of `benchmarks/`, `benchmarks/data/mjw_capacity.json`: naconmax = ceil(nworld mu 1.25) + c_max, njmax = the smallest valid size >= the most rows of any world over the corpora; 94687 / 128 at 4096 worlds, 94688 in the RTX PRO 6000 run of `docs/results/paper_rtx6000` from a measurement with one more contact in its most loaded world, the same per-world sizes 24 / 128; the native headline row), *matched* (nconmax 64 / njmax 192, our engine's capacity) and MuJoCo Warp's *default* (not timed when it overflows; the overflow is recorded). Backward: none in any release (not timed) |
| `mjw_pr1535` | "MuJoCo Warp + unmerged PR #1535 @357a75d" (google-deepmind/mujoco_warp#1535, the head commit as published) | forward graph and the PR's adjoint-graph pattern (`wp.Tape` over the 4 substeps under its `backward_context`, then `tape.backward`), through `nablaengine.reference.warp_physics._Sim` (nconmax 64, njmax 192, sensors disabled, all overflow warnings masked). The adjoint graph contains its own taped forward recompute: the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), forward + adjoint the pattern of a differentiated rollout |
| `mjw_pr1535_fix` | "MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f" (two public fixes) | as `mjw_pr1535` |
| `mjx` | "MJX 3.17 (impl='jax', native)" | `mujoco-mjx` 3.17, AOT `jit(vmap(lax.scan over substeps))`, all state outputs returned; mjlab G1 at 4096 per corpus; Menagerie `unitree_g1/scene_mjx.xml` (commit f3b6013, fetched by the setup) at its own options (4 ms, Newton 5, 5 substeps = 20 ms, sensors as declared) from MuJoCo C rollouts; derivative: "forward-mode Jacobian (native)", `jax.jvp` vmapped over the 99 input directions (qpos tangent 35, qvel 35, ctrl 29; chunks of 33), per-world time at B = 64, 256 and 1024 (each B that fits the memory and an estimated 4-minute budget); reverse mode: the exception text of `jax.vjp` + pullback at B = 2 ("not supported natively"); part `mjx_triton`: the mjlab forward on the first corpus again with `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` (the other XLA setting) |
| `mjc` | "MuJoCo C 3.17 (CPU, float64, native)" | `mujoco.batch.Batch`, nthread = the allocated cores (`len(os.sched_getaffinity(0))`, capped by `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE` when set; `PAPER_BENCH_MJC_THREADS` overrides; never `os.cpu_count()`), 4096 worlds; single-thread forward at B = 64; Jacobian by finite differences, `mjd_transitionFD` per substep chained over the 4 substeps (worlds split over Python threads; the binding releases the GIL inside the C calls), forward and centered, eps 1e-6: per-world rate at B = 512 on every corpus (its 4096 figure is a linear extrapolation, labelled so), and on the first corpus measured at 4096 worlds (one call per round, 3 kept rounds) and with one thread at B = 64; a MuJoCo version other than 3.17.0 is flagged |
| `genesis` | "Genesis 1.4.5 (native)" (PyPI `genesis-world==1.4.5`, quadrants 1.3.3; Track A forward only) | the scene of the benchmark harness's adapter (`benchmarks/bench/adapters/genesis.py`, which records every deviation: the 502 explicit pairs declared through contype / conaffinity because Genesis's importer reads no `<pair>`, condim-1 pairs frictional, `constraint_timeconst` = the model's 0.02, no qacc warm start, foot friction through per-world friction ratios, COM through `set_links_COM`; on sm_70 quadrants' monolith solver kernel); Newton 10, ls 8, tolerance 1e-8, implicitfast, 4 x 5 ms; timed call = `set_qpos` + `set_dofs_velocity` + `control_dofs_position` from the resident corpus inputs + `scene.step()`; one fresh process per corpus and configuration: `gs.init` defaults (the accuracy tables' configuration, the headline row) and `gs.init(performance_mode=True)` (labelled); skipped (environment missing) without a Genesis interpreter |

The nworld sweep (512, 1024, 4096, 16384; `PAPER_BENCH_SWEEP`) covers ours `fwd` / `vjp` and the MuJoCo Warp 3.17.0
forward (procedure and matched capacity) on the first corpus, one process per nworld for MuJoCo Warp; a size that does
not fit is recorded as `skipped: oom`.

Headline cells (4096 worlds, 10 iterations, `mf_s0` and `mf_s1`) are measured in P = 3 fresh processes per engine
(process-to-process variation): the main part plus the repeat parts `rep2_<part>` and `rep3_<part>` (only those
cells; ours without the Jacobian, MJX without Menagerie / Jacobian, MuJoCo C without FD); the summary lists the
per-process medians, their median and the spread.

Run order: ours first, then the MuJoCo Warp variants, MJX, MuJoCo C, then the repeat processes. Every part has its own
timeout and the job a budget (`PAPER_BENCH_BUDGET_MIN`, default 53 min, counted from the start of `run.sh`, the setup
included): a part that cannot start in the time left is recorded as skipped. `paper_bench.json` is rewritten after
every part and every worker rewrites its `parts/<part>.json` after every cell, so a job killed by the scheduler keeps
what it finished.

## Setup

`run.sh` first runs `setup.sh` (idempotent, a few minutes the first time, seconds afterwards), which creates under
`$PAPER_BENCH_PREFIX`: `venv_mjw` (PyPI MuJoCo Warp 3.17.0), `venv_pr` (mujoco 3.17.0 + warp-lang 1.18.0, no
mujoco_warp: the two PR trees go on `PYTHONPATH`), `src/mujoco_warp_357a75d`, `src/mujoco_warp_4af093f` (plain exports
of the two public commits, fetched from etaoxing/mujoco_warp, the fork that hosts both) and `menagerie/`, and records
versions and the warp wheel in `setup.json`. warp-lang: the default PyPI wheel (CUDA 13 build) when the driver is
>= 580, otherwise NVIDIA's `+cu12` wheel of the same release. Without network access a missing component fails with a
clear message; the parts that need it are then skipped and the rest still run. Run the setup once on a machine with
network access when the GPU machines have none.

Our engine, MJX and MuJoCo C run in the main environment: the interpreter `$PAPER_BENCH_DEV_PYTHON` (default
`python3`) must have NablaEngine's dependencies (`scripts/setup_env.sh`: JAX with CUDA, `mujoco==3.17.0`,
`mujoco-mjx==3.17.0`, `warp-lang`) and `nvcc` for the engine library (`$NVCC` or on `PATH`; the library is cached in
`$DK_CACHE_DIR`, default `~/.cache/nablaengine`). MuJoCo Warp itself is not needed there: its parts run in the
setup's venvs.

## Environment variables

None of them changes any result bitwise; `PAPER_BENCH_QUICK` changes only the number of rounds and calls.

| variable | default | meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | (unset: device 0) | the one GPU every part uses (its first entry; the kit sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` when unset, so the index is nvidia-smi's) |
| `PAPER_BENCH_PREFIX` | `runs/paper_bench_env` | where the setup installs the native environments and sources |
| `PAPER_BENCH_DEV_PYTHON` | `python3` | the main environment's interpreter |
| `PAPER_BENCH_PYTHON` | `$PAPER_BENCH_DEV_PYTHON` | the interpreter the native venvs are created from (>= 3.10) |
| `PAPER_BENCH_SETUP_TIMEOUT_MIN` | `15` | the setup's time limit (minutes) |
| `PAPER_BENCH_MJC_THREADS` | (allocated cores) | MuJoCo C's thread count |
| `PAPER_BENCH_JOB_START` | set by `run.sh` | the job's start (epoch seconds), the origin of the budget; not a user setting |
| `PAPER_BENCH_WARP_WHEEL` | `auto` | `default` (PyPI, CUDA 13) / `cu12`; auto: by driver >= 580 |
| `PAPER_BENCH_SKIP_SETUP` | `0` | `1`: do not run `setup.sh` (use what is installed) |
| `PAPER_BENCH_QUICK` | `0` | `1`: smoke test of the kit (2 kept rounds, shorter warm-up, MJX Jacobian at B = 16 only); not for publication |
| `PAPER_BENCH_PARTS` | all | comma-separated part names (`ours,mjw_native,...`; a name also selects its sweep parts `<name>_nw*` and its repeat processes `rep<r>_<name>`; `reps` selects every repeat) |
| `PAPER_BENCH_REPEATS` | `2` (QUICK: `1`) | extra processes per headline cell (`rep2_*`, `rep3_*`): P = 1 + this |
| `PAPER_BENCH_CORPORA` | `mf_s0,mf_s1` | comma-separated corpus names (`benchmarks/data/mf_s*.npz`) |
| `PAPER_BENCH_SWEEP` | `512,1024,4096,16384` | nworld sweep |
| `PAPER_BENCH_MJX_JAC_BATCHES` | `64,256,1024` | batch sizes of the MJX forward-mode Jacobian |
| `PAPER_BENCH_BUDGET_MIN` | `53` | whole-job budget (minutes) |
| `PAPER_BENCH_TIMEOUT_SCALE` | `1` | multiplies every per-part timeout |
| `PAPER_BENCH_GENESIS_PYTHON` | `$PAPER_BENCH_PREFIX/venv_genesis`, else `runs/bench/envs/genesis/venv` (`benchmarks/envs/setup.sh --envs genesis`) | the Genesis environment's interpreter (genesis-world 1.4.5, quadrants 1.3.3, mujoco 3.17.0, PyTorch with CUDA); the setup does not install it |
| `PAPER_BENCH_MJW_PYTHON`, `PAPER_BENCH_PR_PYTHON`, `PAPER_BENCH_PR_TREE_357A75D`, `PAPER_BENCH_PR_TREE_4AF093F` | under the prefix | use existing installs instead of the setup's (a tree is a directory that contains `mujoco_warp/`) |

## Expected run time

On a V100 (16 GB, 8 CPU cores, warm Warp kernel cache and engine library cache), `PAPER_BENCH_QUICK=1` takes about
12 minutes: ours under 1 min, ours_bare under 1 min, mjw_native with its sweep about 30 s, each PR part about 10 s,
mjx about 4 min, mjc about 3 min, mjx_triton under 1 min, one repeat set under 2 min. A full run takes about
25 minutes there (mjx about 7 min: the Jacobian at B = 256 does not fit in 16 GB; mjc about 7 min with 8 cores, of
which the finite differences at 4096 worlds take 3).

On an RTX PRO 6000 with 16 cores a full run with warm caches took 16 minutes (`docs/results/paper_rtx6000/`; that
result has no `ours_bare` part, which is attribution only and enters no table). A cold
machine adds the setup (about 2 min with network), the engine library build (about 0.5-1 min), Warp's kernel
compilation (about 1 min for MuJoCo Warp 3.17.0, about 1.4 min per PR #1535 tree) and JAX's first compilations: about
6 minutes. The budget (`PAPER_BENCH_BUDGET_MIN`, counted from the start of `run.sh`) skips, and records, whatever would
not fit, the repeat sets first (they run last).

## Running under a batch scheduler

One GPU, the CPU cores that come with it, 55 minutes, from the repository root, with the main environment active
(SLURM example):

```
#SBATCH --gres=gpu:1 --cpus-per-task=<cores> --time=00:55:00
export PAPER_BENCH_PREFIX=<a persistent directory, set up once with network access>
export PAPER_BENCH_DEV_PYTHON=$(command -v python)
export DK_CACHE_DIR=<a persistent directory for the engine library>     # optional
scripts/paper_bench/run.sh runs/paper_bench/<gpu label>
```

MuJoCo C uses the allocated cores (the affinity mask, capped by `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE`); the
JSON records the count and the rule. If the GPU machines have no network access, run `scripts/paper_bench/setup.sh`
once with the same `PAPER_BENCH_PREFIX` where there is network, and set `PAPER_BENCH_SKIP_SETUP=1` in the job (the
setup's warp wheel follows the driver of the machine it runs on, and is the default CUDA 13 wheel on a machine without
a GPU; set `PAPER_BENCH_WARP_WHEEL=default` or `cu12` to choose). Our engine's library is built with nvcc on first
use; where nvcc cannot build (for example missing host headers), build it once on a machine with the same GPU
architecture where it can (`python -c "from nablaengine import jax_ffi; jax_ffi.init()"` with the same
`DK_CACHE_DIR`); `ours_bare` builds standalone binaries and fails there (recorded, the job goes on).
