# The G1 model (`fixtures/mjlab`)

`g1_train.mjb` is the model the engine is written for: the Unitree G1 humanoid with a floating base and 29 hinge
joints, each driven by a position (PD) actuator, standing on a flat floor. It is built from the Unitree G1 MJX
description in [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), with feet, contacts, joint and
actuator parameters changed after the G1 configuration of [mjlab](https://github.com/mujocolab/mjlab).

This directory is the default fixtures directory of `nablaengine.fixtures` and of the scripts (`DK_FIXTURES`
overrides it). `nablaengine.fixtures.load_model()` loads the model and checks that the engine supports it.

| File | Content |
|---|---|
| `g1_train.mjb` | The compiled model in MuJoCo's binary format, 176356 bytes, sha256 `757b0997d992073b645732a1daaeab92587f00e29691b93aa76094a3f02ef379`. |
| `g1_meta.json` | Metadata of the model (below). |
| `LICENSE` | The licence of the Unitree G1 description (BSD 3-Clause), copied verbatim from MuJoCo Menagerie. |

The other files are reference data for the checks, all computed with this model: `mjw317_mf_s0.npz` and
`mjw317_mf_s1.npz`, the outputs of one control step of native MuJoCo Warp 3.17.0 and its contacts at the input states
of the benchmark corpora `mf_s0` / `mf_s1` (`scripts/make_mjw_reference.py`), and `engine_digests.json`, the sha256
digests of the engine's forward outputs (`nablaengine.checks`).

## The model

| | |
|---|---|
| Degrees of freedom | `nq` 36, `nv` 35: a free root joint and 29 hinge joints. 31 bodies (the world included), 29 actuators. |
| Collision geometry | 34 geoms: the floor plane, 2 spheres (pelvis, head) and 31 capsules, 7 of them on each sole. No meshes, boxes or other geom types. |
| Contacts | 502 explicit contact pairs and nothing else: `contype` / `conaffinity` are 0 on every robot geom. |
| Options | Timestep 5 ms, Newton solver, pyramidal friction cones, `implicitfast` integrator, Euler damping disabled. The engine sets its own Newton iteration count. |
| Control step | 4 substeps (20 ms), the convention recorded in `g1_meta.json`. |
| Other content | 3 materials, 1 light, 1 camera, 6 sites, 14 sensors and the keyframes `home` and `knees_bent`, as in Menagerie. |

The binary format ties the file to the MuJoCo version that wrote it, `mujoco==3.17.0`:
`mujoco.MjModel.from_binary_path("fixtures/mjlab/g1_train.mjb")`.

`g1_meta.json` holds:
- `mujoco_version`: the MuJoCo version that wrote the model;
- `foot_pairs`: the indices of the 14 sole-floor contact pairs (`left_foot1_floor` ... `right_foot7_floor`), whose
  friction can be set per world;
- `torso_body`: the body whose centre-of-mass offset can be set per world;
- the sizes `nq`, `nv`, `nu`, `nbody` and `npair`;
- `timestep`, `substeps_per_control_step`, `solver`, `cone` and `integrator`;
- `robot`: the name of the model variant, `"mjlab"`;
- `source`: a one-line provenance.

## How it was derived

The source is MuJoCo Menagerie at commit
[`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_g1),
file `unitree_g1/scene_mjx.xml` (the flat-floor scene, which includes `unitree_g1/g1_mjx.xml`). The steps below were
applied with MuJoCo's `MjSpec` in MuJoCo 3.17.0, and the result was written with `mujoco.mj_saveModel` (not as MJCF,
whose writer rounds numbers). Applied in this order to the Menagerie files at that commit, they reproduce
`g1_train.mjb` byte for byte (the same sha256).

1. **Sole capsules.** The foot collision geoms of the MJX model are removed: per foot, three capsules
   (`<side>_foot1_collision` ... `<side>_foot3_collision`) and a box (`<side>_foot_box_collision`). In their place,
   seven capsules per sole are added to each ankle-roll link, as in mjlab's G1 (left foot first, then the right):
   radius 0.01 m, `contype` / `conaffinity` 0, group 3, `condim` 3, friction (0.6, 0.005, 0.0001), named
   `<side>_foot<k>_collision`. Half-lengths, positions and orientations in the ankle-roll link frame:

   | k | left: half-length, position, quaternion (w, x, y, z) | right: half-length, position, quaternion (w, x, y, z) |
   |---|---|---|
   | 1 | 0.025005, (0.075, -0.0265, -0.025), (0.707107, -0.014139, 0.706965, 0) | 0.025, (0.075, -0.026, -0.025), (0.707107, 0, 0.707107, 0) |
   | 2 | 0.0835, (0.0395, -0.018, -0.025), (0.707107, 0, -0.707107, 0) | 0.0835, (0.0395, -0.018, -0.025), (0.707107, 0, -0.707107, 0) |
   | 3 | 0.091, (0.039, -0.01, -0.025), (0.707107, 0, -0.707107, 0) | 0.091, (0.039, -0.01, -0.025), (0.707107, 0, -0.707107, 0) |
   | 4 | 0.093, (0.039, 0, -0.025), (0.707107, 0, -0.707107, 0) | 0.093, (0.039, 0, -0.025), (0.707107, 0, -0.707107, 0) |
   | 5 | 0.091, (0.039, 0.01, -0.025), (0.707107, 0, -0.707107, 0) | 0.091, (0.039, 0.01, -0.025), (0.707107, 0, -0.707107, 0) |
   | 6 | 0.0835, (0.0395, 0.018, -0.025), (0.707107, 0, -0.707107, 0) | 0.0835, (0.0395, 0.018, -0.025), (0.707107, 0, -0.707107, 0) |
   | 7 | 0.025, (0.075, 0.026, -0.025), (0.707107, 0, 0.707107, 0) | 0.025, (0.075, 0.026, -0.025), (0.707107, 0, 0.707107, 0) |

2. **Contact pairs.** Menagerie's 49 explicit pairs are removed. The new pairs give mjlab's contact set of the G1, in
   which every collision geom collides with the floor and with every collision geom of another body, except on
   parent-child bodies and on four excluded body pairs (the pelvis with each hip-roll link, each elbow link with the
   wrist-pitch link of the same arm). It is written as 502 explicit pairs, in this order: each robot collision geom
   (in geom order) with the floor, named `<side>_foot<k>_floor` for the sole capsules and `<geom>__floor` otherwise;
   then every allowed pair of robot collision geoms in geom order, named `<geom1>__<geom2>`. Pair parameters:
   - `condim` 3 for the floor pairs and every pair with a sole capsule, 1 for the other pairs;
   - sliding friction 0.6 between two sole capsules and 1.0 otherwise, torsional 0.005, rolling 0.0001;
   - on every pair, MuJoCo's default `solref` (0.02, 1) and `solimp` (0.9, 0.95, 0.001, 0.5, 2). Menagerie's
     foot-floor pairs have `solref` (0.008, 1).
3. **Visual content.** The mesh geoms (which do not collide), the mesh assets and the textures are removed. This
   changes no dynamics, collision or solver quantity of the model.
4. **Options.** The timestep is 5 ms (Menagerie's MJX model: 4 ms). The other options stay as in Menagerie's MJX
   model.
5. **Actuators.** mjlab's PD gains, from the armature of each joint (Menagerie's values, as are the torque limits):
   kp = armature (2 pi 10 Hz)^2 and kd = 2 * 2 * armature (2 pi 10 Hz), a natural frequency of 10 Hz and a damping
   ratio of 2 (`gainprm[0]` = kp, `biasprm[1:3]` = (-kp, -kd)).
6. **Joint and control parameters**, set on the compiled model:
   - no dof friction loss (Menagerie: 0.1 N m);
   - joint-limit `solref` / `solimp` at MuJoCo's defaults as above (Menagerie: (0.008, 1) and (0, 0.99, 0.01));
   - mjlab's setpoint range: each actuator's `ctrlrange` is its joint's range widened on each side by
     tau_max / kp, where tau_max is the joint's actuator force limit.

## Licences

- **Unitree G1 description** (MuJoCo Menagerie `unitree_g1`): Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY
  CO.,LTD. ("Unitree Robotics"), BSD 3-Clause License. Its full text is `LICENSE` in this directory, and it applies to
  `g1_train.mjb`, a modified binary form of that description. The Menagerie repository as a whole is under
  Apache-2.0, but the licence of the model directory governs this model.
- **mjlab**: Apache License 2.0, Copyright 2025, The mjlab Developers. The model takes over values of mjlab's G1
  configuration (the sole capsules, the contact set, the contact and joint parameters, the PD gains and the setpoint
  range). It contains no mjlab code.
- The other files of this directory are covered by the repository's licence (Apache-2.0, `LICENSE` at the repository
  root). `THIRD_PARTY_NOTICES.md` at the root lists all third-party material.

Unitree Robotics, Google DeepMind and the mjlab developers do not endorse this project. Their names are used only to
identify the sources.
