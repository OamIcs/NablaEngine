# Benchmark truth files

One file per corpus, `<corpus>.npz`, written by `python -m benchmarks.bench truth make` (code:
`benchmarks/bench/truth/`). Every number in them comes from MuJoCo C in float64; the only other engine involved is
native MuJoCo Warp 3.17.0, for the model check. Nothing comes from NablaEngine, from any other MuJoCo Warp
build. `truth.json` is the index: for each file it gives the sha256, size, counts, the
command that makes it and where it ran (date, Python, CPU); `python -m benchmarks.bench truth index --corpora ...`
rebuilds it from the files.

| corpus | kind | gradient states (sampled G0/G4/G5/G6/G1) | model-ok of 4096 | group population G0/G4/G5/G6/G1 | bytes |
|---|---|---|---|---|---|
| mf_s0 | benchmark | 252 (64/64/64/64/0) | 4059 | 3368/111/1441/1416/0 | 3166023 |
| mf_s1 | benchmark | 257 (64/64/64/64/2) | 4058 | 3400/120/1468/1427/2 | 3190027 |
| mf_s2 | benchmark | 252 (64/64/64/64/0) | 4061 | 3389/135/1360/1438/0 | 3166304 |
| mf_s3 | benchmark | 253 (64/64/64/64/0) | 4059 | 3417/115/1435/1419/0 | 3171131 |
| menagerie_s0 | benchmark (Track B) | none: forward truth and groups only | 3994 | 3385/383/854/1243/0 | 110958 |

## What a file holds

The full key list and dtypes are in the docstring of `benchmarks/bench/truth/store.py`.

- **Per-state labels**, for all 4096 states. These are MuJoCo C-only label bits, the model-check flags, solver
  iterations and the continuous labels (axis distance, distance to contact, kink measure, clamp margin).
- **Check values**: the forward truth of 64 fixed states.
- **Gradient truth**, for Track A corpora with stored cotangents. It covers the sampled states and holds:
  - the VJPs of the corpus's 3 stored cotangents through the finite-difference Jacobian at h = 1e-6 (the truth) and at
    h = 1e-7 (the truth floor);
  - the column-validity bits;
  - the group memberships;
  - the unperturbed outputs;
  - for the G0 sample (`--jacobian-groups`), the full Jacobian (71 x 99, float32) and its per-column floor.

The forward truth of every state is not stored, because it would break the size budget. It is regenerated on the CPU,
which takes about 2 s for 4096 states on 24 cores, and cached under `runs/bench/truth/`. The cached copy is checked
against the file's check values: a relative difference of 1e-9 at most, bitwise on the machine that wrote the file.

**Truth options.** MuJoCo C 3.17.0, float64, Newton, 2000 iterations, tolerance 1e-13, ls 100, sensors off. The problem's
substeps run with ctrl held and qfrc_applied 0. Track A applies per-world foot friction and torso COM offset without
mj_setConst. This is the semantics of `benchmarks/bench/truth/fdlib.py` `model_job`.

**FD construction.** The gradient truth uses `fdlib.fd_job` (`benchmarks/bench/truth/fdlib.py`). It takes central
differences of the whole control step, one input column at a time. qpos is perturbed along the tangent with
mj_integratePos, so the quaternion stays normalised.

**Validity bits.** These come from MuJoCo C alone. A column is marked invalid for any of these reasons:
- step: the two step sizes disagree by more than 1e-4 relative plus 1e-6 absolute;
- contacts: a contact set changes;
- rows: constraint rows or their zones change, with "active" meaning `efc_force > 0`;
- clamp: the actuator clamp mask changes;
- ctrlrange: the perturbation crosses ctrlrange;
- solver: the iteration limit is reached.

**Groups.** These also come from MuJoCo C alone and are defined in the `benchmarks/bench/truth/mjc.py` docstring:
- G0 = normal and not clamped;
- G4 = normal and clamped;
- G5 = normal, joint-limit row, not clamped;
- G6 = normal, self contact, not clamped;
- G1 = axis distance below 5 mm (report only).

"Normal" means none of these labels apply:
- axis below 5 mm;
- a contact within 1e-6 m of switching;
- a row within 1e-4 of its kink;
- a self contact at mju_makeFrame's |n_y| = 0.5 basis switch;
- the 50-iteration run hitting its limit.

**Sampling.** Up to 64 states per group, in a fixed seeded order (`truth/mjc.py` `sample`), over all states of the group.
The sample is not restricted to model-ok states, so scores are reported both on all sampled states and on the model-ok
ones.

**Model check.** A state is model-ok when MuJoCo C's truth run and native MuJoCo Warp 3.17.0 agree on per-substep
contacts, edge zones, limit rows and condim-3 frames. The MuJoCo Warp run uses 50 iterations, MuJoCo Warp's own
effective tolerance of 1e-6, and capacity 64 contacts / 192 rows. The rules are in `truth/mjw_labels.py` and
`truth/mjc.py` (`model_check`).

## Reproduce and check

```
source <dk env>/activate.sh                       # mujoco 3.17.0; the mjw-native environment for the model check
export CUDA_VISIBLE_DEVICES=<one GPU>             # only the native MuJoCo Warp labelling uses it
python -m benchmarks.bench truth make --corpora mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0 --jobs 24
python -m benchmarks.bench truth verify --corpora mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0
```

`make` takes about 15 s per corpus on 24 cores of a Xeon E5-2690 v3:
- forward truth and labels: 1-3 s;
- native MuJoCo Warp labels: 5 s;
- FD truth of about 255 states: 5 s, about 100 CPU s.

**Reproducibility.** The files are written deterministically: fixed zip timestamps, and no date or commit inside.
With the same MuJoCo build, a fresh `make --recompute` reproduces every array bitwise except `model_flags`. MuJoCo
Warp's atomics leave 0 to 2 borderline states per corpus whose model check flips between runs (seen on the V100).
3.17.0's `opt.deterministic` did not make its outputs bitwise reproducible either. The committed file is the record.

**Track B floor.** On menagerie_s0, MuJoCo C's own float64 result at the model's ls_iterations of 8 is far from the
converged solution: over all 4096 states the qvel median relative error is 1.0e-3 at 10 or 50 iterations and 1.5e-3 at
the native 5. On 200 states, raising ls_iterations to 100 at 50 iterations brings it to 0 (max 7e-6), while tolerance
1e-13 at ls 8 does not help, so the line search limits the solve. Track B's friction-loss rows (frictionloss 0.1 on
every joint; Track A has none) are the likely reason. The truth itself is converged: tolerance 1e-15 with ls 200
changes it by at most 3e-14. Track A's floors are tiny by comparison: at 10 iterations the median is 0 and the p99 is
3e-6.

**Environment.** No environment variable of its own. The files live in `$DK_BENCH_DATA/truth`, where `DK_BENCH_DATA`
defaults to `benchmarks/data` (see `benchmarks/bench/problem.py`). Caches go to `--cache-dir`, default
`runs/bench/truth`.
