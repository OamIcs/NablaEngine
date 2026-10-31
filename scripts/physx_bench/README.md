# PhysX baseline: Isaac Lab 3.0 / Isaac Sim 6.2, Unitree G1, forward physics only

This directory times NVIDIA PhysX the way Isaac Lab users run it. That means Isaac Lab's own Unitree G1 task
(`Isaac-Velocity-Flat-G1`), Isaac Lab's own G1 asset, and Isaac Lab's default simulation settings, with the PhysX
backend of Isaac Sim (`physics=isaacsim_physx`). The kit installs the stack under one prefix and runs a fixed suite in
one command. It writes one JSON file and one markdown table.

Nothing here is tuned. The only knobs the suite changes are the number of parallel environments and, for one
variant, the robot asset and the self-collision flag.

## What is measured

**Physics only.** For each configuration, `bench_physics_only.py` builds the registered task with `gym.make` and
times the physics part of Isaac Lab's `ManagerBasedRLEnv.step`. One control step is:

```
action a ~ U[-1, 1]^37  ->  joint position targets = default pose + 0.5 a   (Isaac Lab's JointPositionAction)
4 x [ apply_action -> scene.write_data_to_sim -> sim.step(render=False) -> scene.update ]   (dt 5 ms, decimation 4)
```

- The targets are held for the whole control step, which covers 20 ms of simulated time.
- No rewards, observations, terminations, resets or policy run inside the timed loop.
- Each round starts with `env.reset()`, untimed, using the task's own reset events. The robots then flail, fall and hit
  the ground for 50 control steps (1 s).
- The headline number is the median over 8 timed rounds of wall time per control step. GPU work is synchronised at
  round boundaries. The p10-p90 range across rounds is reported next to it.
- Two diagnostics are also reported:
  - a per-control-step distribution, synchronised after every step;
  - a per-operation split, synchronised after every operation. This shows how much of the step is PhysX
    (`sim.step` = `simulate` + `fetch_results`) and how much is Isaac Lab's data path. Serialising inflates the sum.
- The time includes Isaac Lab's actuator and data path, because that is how Isaac Lab drives PhysX.

**Suite.** `run.sh` runs the configurations in this order:

| configuration | what |
|---|---|
| `physx_task_n{1024,4096,8192}` | PhysX, task defaults |
| `physx_full_selfcoll_n4096` | PhysX, full-collision asset `G1_CFG` plus self-collision |
| `newton_task_n{1024,4096}` | Newton with the MuJoCo Warp solver (`physics=newton_mjwarp`), task defaults. This is Isaac Lab 3.0's default backend for this task. Context only. |
| `official_runtime_physx_n4096` | Isaac Lab's own `isaaclab benchmark runtime`. Full `env.step` with random actions, MDP managers and resets. Secondary number. |
| `physx_task_n16384` | optional, skipped if the time budget would be exceeded |
| `physx_full_selfcoll_n8192` | optional, skipped if the time budget would be exceeded |

The result JSON also records the live USD stage that PhysX was given: scene, articulation and collider attributes,
and the joint drives as Isaac Lab reports them.

## This is not the same physics problem as an MJCF model

Report the PhysX numbers next to MuJoCo-family engines as a reference point. Do not present them as the same
simulation.

| aspect | Isaac Lab G1 task (this baseline) | an MJCF G1 such as the mjlab model of this repository |
|---|---|---|
| robot model | older G1 with 3-finger hands: 37 revolute DoF, 44 rigid bodies | 29-DoF G1, no hands |
| collision geometry | `G1_MINIMAL_CFG`: 3 box colliders (both feet and the torso); nothing else collides | capsules and spheres on every link, plus explicit contact pairs |
| self-collision | off by default (`enabled_self_collisions=False`) | usually on through explicit pairs |
| contact model | rigid contacts. TGS solver with 8 position / 4 velocity iterations, PCM collision, patch friction, PhysX's automatic contact offsets. GPU broadphase and dynamics. | soft constraints (solref / solimp), e.g. Newton solver, pyramidal cone |
| actuators | PhysX implicit joint drives. Legs: kp 150-200, kd 5, effort limit 300 N m. Ankles: kp 20, kd 2, limit 20. Arms: kp 40, kd 10. Armature 0.01. | MuJoCo position actuators with their own gains and armature |
| time step | 5 ms x 4 per 20 ms control step | same (by choice) |

**Variants.**
- The full-collision variant uses `G1_CFG` (`g1.usd`): 38 colliders (32 convex hulls, 4 cylinders, 2 boxes) with
  self-collision on. It brings the collision workload closer to a full-body MJCF model.
- The robot model and the contact model stay different.

**Why we do not load an MJCF model into PhysX.** Isaac Sim's MJCF importer (mujoco-usd-converter) has these gaps:
- It drops explicit `<pair>` contacts ("pairs are not supported").
- It gives geoms with `contype = conaffinity = 0` no collider at all. A pair-only model therefore has no robot
  colliders in PhysX.
- It keeps armature only as a Newton attribute, which PhysX does not read.
- It maps only actuator `forcerange`, not joint `actuatorfrcrange`, to the drive force limit.

Making such a model work in PhysX takes hand-tuning. The result would no longer be "PhysX as Isaac Lab ships it".

**Suggested labels**
- *PhysX 5 GPU (Isaac Lab 3.0.0 / Isaac Sim 6.2.0, `isaacsim_physx`), Isaac Lab G1 asset (37 DoF, 3 box colliders,
  self-collision off), Isaac Lab task defaults, physics only*
- *same, full-collision G1 asset (38 colliders) with self-collision*
- *Newton 1.6 / MuJoCo Warp 3.12 (`newton_mjwarp`, Isaac Lab 3.0's default backend for this task), Isaac Lab G1
  asset, physics only: context*
- *Isaac Lab runtime benchmark, PhysX, full `env.step` (random actions, MDP managers, resets)*

**Claims these numbers support.** The cost per 20 ms control step of each simulator running its own maintained G1
configuration, at the same batch size, on the same GPU.

**Claims they do not support.** Equal fidelity, an equal number of contacts, or a speed-up "on the same robot".

## Requirements

- Linux x86_64 with glibc >= 2.35 (Ubuntu 22.04 or newer). On older hosts, use a container.
- An RTX-class NVIDIA GPU. Isaac Sim starts its RTX/Vulkan stack even when headless, and it does not support GPUs
  without RT cores (for example A100, H100).
- NVIDIA driver >= 580.65.06. This is Isaac Lab 3.0's documented minimum, and its default PyTorch is built for
  CUDA 13.0. Isaac Sim 6.2 was tested with 595.58.03.
  - With an older driver, `install.sh` installs the CUDA 12.8 build of the same torch 2.11.0.
  - The results are then flagged as below the supported range (`driver_below_isaaclab_minimum_580_65`), so treat them
    as feasibility checks.
- Disk: 35-40 GB under the prefix (the venv is about 29 GB, of which Isaac Sim is about 17 GB; the rest is uv's
  download cache and Kit / asset caches). RAM: 32 GB or more;
  the process reaches about 20 GB at 4096 environments.
- Network at install time: pypi.org, pypi.nvidia.com, download.pytorch.org, astral.sh, raw.githubusercontent.com.
- Network at run time: Isaac Lab downloads the robot and ground USD assets from NVIDIA's public asset server on first
  use, and caches them under the prefix's TMPDIR. For nodes without internet, mirror the assets and set
  `ISAACSIM_ASSET_ROOT` (an Isaac Sim / Isaac Lab variable) to the mirror.

## License (EULA)

Running Isaac Sim requires accepting the NVIDIA Omniverse License Agreement:
<https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html>.

These scripts **never** accept it for you: they never set `OMNI_KIT_ACCEPT_EULA` and never write Isaac Sim's marker
file. `run.sh` checks whether the operator has accepted, using Isaac Sim's own rule. Without acceptance, it exits
with code 2 and prints this URL.

If you agree to the license, accept it yourself in one of two ways.

- **Once per installation, interactively.** Isaac Sim prints the notice, asks `Do you accept the EULA? (Yes/No)`, and
  writes `isaacsim/kit/EULA_ACCEPTED` inside the venv:
  ```
  PHYSX_BENCH_PREFIX=<prefix> bash -c 'source scripts/physx_bench/common.sh && python -c "import isaacsim"'
  ```
- **Per run.** Export `OMNI_KIT_ACCEPT_EULA=YES` in your own shell or job script before calling `run.sh`.

## Install

```
export PHYSX_BENCH_PREFIX=/path/to/node-local/physx_bench    # 35-40 GB
scripts/physx_bench/install.sh            # ~20 min at ~15 MB/s; idempotent (re-running is a no-op)
scripts/physx_bench/install.sh --dry-run  # print the commands only
```

The script runs Isaac Lab's documented pip route:
1. uv 0.13.6.
2. A Python 3.12 venv.
3. `uv pip install "isaaclab[isaacsim]==3.0.0"` with the v3.0.0 `uv-overrides.txt`, from `https://pypi.nvidia.com`.
   The `--torch-backend` is cu130, or cu128 with a driver below 580.65.
4. A sanity check that torch can use CUDA.

The script redirects `HOME`, the `XDG_*` directories, `TMPDIR`, and the uv / pip / torch / warp caches into the
prefix. Kit writes `~/.nvidia-omniverse` and `~/.cache/ov`, and also logs and settings inside the venv under
`site-packages/isaacsim/kit/`.

## Run

```
export PHYSX_BENCH_PREFIX=/path/to/node-local/physx_bench
scripts/physx_bench/run.sh --no-idle-check     # exclusive GPU: full suite, <= 55 min excluding install
scripts/physx_bench/run.sh --quick             # validation (N=1024, short rounds)
```

Measured reference (RTX 3090, driver 565, dual 22-core 2.2 GHz Xeon): `--quick` took 3 min. The full suite took 38 min,
and both optional configurations ran. The largest items were N=16384 (15 min, mostly environment creation) and the
full-collision N=8192 run (7 min). A faster CPU shortens the start-ups the most.

- **GPU selection.** The suite uses the first device in `CUDA_VISIBLE_DEVICES`, as set by Slurm or by you. If that
  variable is unset, it uses `PHYSX_BENCH_GPU`, which defaults to 0.
- **Idle check.** By default the suite stops (exit 3) if the GPU has another compute process or more than 1 GiB in
  use. On an exclusive card, pass `--no-idle-check`.
- **Time budget.** Every configuration starts Kit in a fresh process. Environment creation is CPU-bound and grows
  faster than linearly with N. On an older 2.2 GHz server CPU it took about 2 min at 4096 environments, 5 min at 8192
  and 13 min at 16384. When an optional configuration would not fit in the budget (`--budget-min`, default 55), it is
  skipped and listed in the output.

**Output** goes to `runs/physx_bench/<UTC time>_<mode>/` in the repository, which is git-ignored. You can set the
directory with `--out-dir`, but it must be new or empty.

| file | content |
|---|---|
| `physx_bench_results.json` | environment (GPU, driver, CPU, package versions); install choices and the license note; repository commit; for each configuration the command, medians with p10-p90, diagnostics, startup time, memory, contact proxy and live-stage settings |
| `physx_bench_results.md` | the summary table |
| `runs/`, `logs/` | per-configuration JSON and Isaac Lab's benchmark output; full logs |

Exit status:
- 0: every mandatory configuration produced a result.
- 1: a mandatory configuration failed or did not fit.
- 2: install or license missing.
- 3: GPU busy.

## Containers, Vulkan and clusters

- **Vulkan.** Kit initialises Vulkan and the RTX renderer even in headless runs. Two things must be visible: the
  NVIDIA Vulkan ICD (`nvidia_icd.json`) and the graphics user-space libraries (`libGLX_nvidia.so.0`,
  `libnvidia-glvkspirv.so`, ...). Compute-only driver installs fail. Check with `vulkaninfo --summary`.
- **Docker with the NVIDIA Container Toolkit.** `NVIDIA_DRIVER_CAPABILITIES` must include `graphics`, for example
  `docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...`. With the default `compute,utility`, Kit cannot
  create a Vulkan device.
- **Apptainer / Singularity.** `--nv` binds the driver libraries, but not always the ICD file. Bind it explicitly and
  point the Vulkan loader at it:
  ```
  apptainer exec --nv --bind /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \
      --env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json <image> scripts/physx_bench/run.sh ...
  ```
  The host's ICD may live in `/etc/vulkan/icd.d` instead. Older Vulkan loaders use `VK_ICD_FILENAMES`.
- **Slurm.** The suite honours `CUDA_VISIBLE_DEVICES`. Kit's RTX module logs a warning that this variable "can lead
  to undesired behavior"; with a single visible GPU it ran correctly.
- **Disk.** Put `PHYSX_BENCH_PREFIX` on node-local disk rather than a quota-limited shared file system. The first Kit
  start of a fresh install also fills extension and shader caches, which adds a few minutes.

## Environment variables

| variable | default | effect |
|---|---|---|
| `PHYSX_BENCH_PREFIX` | (required) | install prefix: uv, Python, venv, caches, and the redirected HOME / XDG / TMPDIR |
| `PHYSX_BENCH_TORCH_BACKEND` | automatic | `cu130` or `cu128`. Automatic means cu130 with driver >= 580.65, else cu128. Changes only the torch build, not physics settings. |
| `PHYSX_BENCH_SKIP_GPU_CHECK` | `0` | `1` skips install.sh's final CUDA check (install while the GPU is busy) |
| `PHYSX_BENCH_GPU` | `0` | GPU index, used only when `CUDA_VISIBLE_DEVICES` is unset |
| `PHYSX_BENCH_IDLE_CHECK` | `1` | `0` disables the idle-GPU check (same as `--no-idle-check`) |
| `PHYSX_BENCH_BUDGET_MIN` | `55` | suite time budget in minutes (same as `--budget-min`) |
| `UV_HTTP_TIMEOUT` | `900` | uv download timeout in seconds; pypi.nvidia.com can be slow |
| `OMNI_KIT_ACCEPT_EULA` | not set | Isaac Sim's own acceptance switch. Only the operator sets it; the scripts read it. |
| `ISAACSIM_ASSET_ROOT` | NVIDIA's asset server | Isaac Sim / Isaac Lab asset root, e.g. a local mirror for offline nodes |

None of these change the physics configuration.

## Files

| file | role |
|---|---|
| `common.sh` | environment confinement and the license check |
| `install.sh` | installation |
| `run.sh` | entry point for the suite |
| `run_suite.py` | plan, idle check, time budget, aggregation into JSON and the table |
| `bench_physics_only.py` | one configuration; also dumps the live stage |
