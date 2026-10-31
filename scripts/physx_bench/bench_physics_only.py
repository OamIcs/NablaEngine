"""Physics-only timing of Isaac Lab's own Unitree G1 task with the backend Isaac Lab runs it on (one configuration).

Called by run_suite.py (one process per configuration: Kit cannot be restarted inside a process). It builds the task
exactly as Isaac Lab registers it (``gym.make`` of ``Isaac-Velocity-Flat-G1``: Isaac Lab's G1 USD asset, actuators,
sim and physics settings), then times only the physics part of ``ManagerBasedRLEnv.step``:

    per control step:  action_manager.process_action(a),  a ~ U[-1, 1]^37   (target = default pose + 0.5 a, held)
                       decimation x [ apply_action -> scene.write_data_to_sim -> sim.step(render=False) -> scene.update ]

No rewards, observations, terminations, resets or policy inside the timed loop. ``env.reset()`` (the task's own reset
events, untimed) starts every round, so each round of ``--steps_per_round`` control steps (default 50 = 1 s simulated)
covers robots that flail, fall and hit the ground. Timing: wall clock with ``torch.cuda.synchronize()`` and
``wp.synchronize()`` at round boundaries; the headline is the median over timed rounds. Two diagnostic passes follow
(not part of the headline): one round synchronised after every control step (per-step distribution) and one
synchronised after every operation (where the time goes; serialising inflates the sum).

The live USD stage is dumped into the result (physics scene, articulation root and collider attributes of env 0,
per-joint drive values as Isaac Lab reports them), so the JSON records what PhysX was actually given.

usage: python bench_physics_only.py --out r.json --label x --num_envs 4096 physics=isaacsim_physx
       [--robot task|full] [--self_collision default|on|off] [--rounds 8] [--steps_per_round 50]
Hydra preset tokens (``physics=...``) are passed through exactly as Isaac Lab's own scripts take them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time

T_START = time.perf_counter()


def parse():
    from isaaclab.app import add_launcher_args
    from isaaclab_tasks.utils import setup_preset_cli

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaac-Velocity-Flat-G1")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=8, help="timed rounds")
    p.add_argument("--warmup_rounds", type=int, default=1)
    p.add_argument("--steps_per_round", type=int, default=50, help="control steps per round (50 = 1 s simulated)")
    p.add_argument("--self_collision", choices=["default", "on", "off"], default="default")
    p.add_argument("--robot", choices=["task", "full"], default="task",
                   help="task = the task's asset (G1_MINIMAL_CFG); full = G1_CFG (g1.usd, full collision meshes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--baseline_mem_mib", type=float, default=None, help="device memory in use before the run")
    add_launcher_args(p)
    args, remaining = setup_preset_cli(p, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def pct(xs, q):
    """Linear-interpolated percentile (q in [0, 100])."""
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def smi(query: str) -> str:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", dev],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def process_mem_mib(pid: int) -> float | None:
    """This process's GPU memory per nvidia-smi (None inside PID namespaces, e.g. some containers)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        tot = sum(float(b) for a, b in (ln.split(",") for ln in out.strip().splitlines() if ln.strip()) if int(a) == pid)
        return tot or None
    except Exception:
        return None


def cpu_model() -> str:
    with contextlib.suppress(Exception):
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    return platform.processor()


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ["isaaclab", "isaacsim", "isaacsim-kernel", "torch", "warp-lang", "newton", "mujoco", "mujoco-warp",
              "ovphysx", "usd-exchange", "numpy"]:
        with contextlib.suppress(Exception):
            out[p] = md.version(p)
    return out


def jsonable(v):
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (int, float)) for x in v):
        return [jsonable(x) for x in v]
    return str(v)


def to_torch(x):
    import torch
    import warp as wp
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return torch.as_tensor(x)


def dump_stage(u) -> dict:
    """PhysX-relevant attributes of the live stage for env 0 (empty sections when a backend has no USD stage)."""
    out: dict = {}
    try:
        from pxr import Usd, UsdPhysics
        stage = getattr(u.sim, "stage", None)
        if stage is None:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
    except Exception as e:  # pragma: no cover
        return {"error": f"no stage: {e}"}
    scenes = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            scenes[str(prim.GetPath())] = {a.GetName(): jsonable(a.Get()) for a in prim.GetAttributes()
                                           if a.GetName().startswith(("physxScene", "physics:", "newton:", "mjc:"))}
    out["physics_scenes"] = scenes
    root = "/World/envs/env_0/Robot"
    rp = stage.GetPrimAtPath(root)
    if not rp:
        return out
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    art, colliders, types = {}, [], {}
    for prim in Usd.PrimRange(rp, pred):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art[str(prim.GetPath())[len(root):] or "/"] = {a.GetName(): jsonable(a.Get()) for a in prim.GetAttributes()
                                                           if a.GetName().startswith(("physxArticulation", "physxRigidBody"))}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            t = prim.GetTypeName()
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                t += f"[{UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}]"
            types[t] = types.get(t, 0) + 1
            rec = {"path": str(prim.GetPath())[len(root):], "type": t}
            rec.update({a.GetName(): jsonable(a.Get()) for a in prim.GetAttributes()
                        if a.GetName().startswith(("physxCollision", "physxConvexHullCollision"))})
            colliders.append(rec)
    out["articulation_root"] = art
    out["robot_collider_count"] = len(colliders)
    out["robot_collider_types"] = types
    out["robot_colliders"] = colliders
    robot = u.scene["robot"]
    d = robot.data
    joints = {"names": list(robot.joint_names)}
    for key in ["joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff", "joint_effort_limits",
                "joint_vel_limits", "joint_pos_limits", "default_joint_pos"]:
        with contextlib.suppress(Exception):
            joints[key] = [jsonable(x) for x in to_torch(getattr(d, key))[0].tolist()]
    out["joints_env0"] = joints
    return out


def main():
    args, remaining = parse()
    t_imports0 = time.perf_counter()
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, None)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.robot == "full":
        from isaaclab_assets import G1_CFG

        from isaaclab.utils import replace
        env_cfg.scene.robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    if args.self_collision != "default":
        on = args.self_collision == "on"
        props = env_cfg.scene.robot.spawn.articulation_props
        for pr in props if isinstance(props, (list, tuple)) else [props]:
            if hasattr(pr, "enabled_self_collisions"):  # PhysX
                pr.enabled_self_collisions = on
            if hasattr(pr, "self_collision_enabled"):  # Newton
                pr.self_collision_enabled = on

    rec: dict = {"label": args.label, "task": args.task, "num_envs": args.num_envs, "hydra_overrides": remaining,
                 "robot_variant": args.robot, "self_collision_override": args.self_collision,
                 "rounds": args.rounds, "warmup_rounds": args.warmup_rounds, "steps_per_round": args.steps_per_round,
                 "seed": args.seed}
    t_app0 = time.perf_counter()
    with launch_simulation(env_cfg, args):
        t_app1 = time.perf_counter()
        import warp as wp

        t_env0 = time.perf_counter()
        with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
            t_env1 = time.perf_counter()
            u = env.unwrapped
            sim, scene = u.sim, u.scene
            dec = u.cfg.decimation
            steps_per_call = dec if u._physics_handles_decimation else 1
            physics_dt = u.physics_dt
            n_act = u.action_manager.total_action_dim
            pm, pcfg = sim.physics_manager, sim.cfg.physics
            rec["backend"] = {"physics_manager": getattr(pm, "__name__", type(pm).__name__),
                              "physics_cfg_type": type(pcfg).__name__,
                              "physics_handles_decimation": bool(u._physics_handles_decimation)}
            rec["physics_cfg"] = {k: jsonable(v) for k, v in vars(pcfg).items() if not k.startswith("_")}
            rec["sim"] = {"dt": float(sim.cfg.dt), "decimation": int(dec), "control_dt": float(dec * sim.cfg.dt),
                          "device": str(sim.cfg.device), "action_dim": int(n_act)}
            robot = scene["robot"]
            rec["robot"] = {"num_joints": int(robot.num_joints), "num_bodies": int(robot.num_bodies)}
            with contextlib.suppress(Exception):
                rec["live_stage"] = dump_stage(u)
            cs = scene.sensors.get("contact_forces") if hasattr(scene, "sensors") else None

            def sync():
                torch.cuda.synchronize()
                wp.synchronize()

            def control_step(a):
                u.action_manager.process_action(a)
                for _ in range(dec // steps_per_call):
                    u.action_manager.apply_action()
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(dt=physics_dt * steps_per_call)

            def contact_stats():
                if cs is None:
                    return None
                f = to_torch(cs.data.net_forces_w)
                if f.dim() == 3 and f.shape[-1] != 3:
                    f = f.view(f.shape[0], -1, 3)
                c = (f.norm(dim=-1) > 1e-3).float()
                return {"bodies_in_contact_per_env": float(c.sum(-1).mean()),
                        "envs_with_contact_frac": float((c.sum(-1) > 0).float().mean())}

            gen = torch.Generator(device=u.device)
            gen.manual_seed(args.seed)
            S = args.steps_per_round

            def new_actions():
                return 2.0 * torch.rand(S, u.num_envs, n_act, device=u.device, generator=gen) - 1.0

            rounds, first_ms = [], None
            with torch.inference_mode():
                for r in range(args.warmup_rounds + args.rounds):
                    u.reset()
                    acts = new_actions()
                    sync()
                    k0 = 0
                    if first_ms is None:  # first control step after creation (includes lazy init / JIT)
                        t0 = time.perf_counter()
                        control_step(acts[0])
                        sync()
                        first_ms, k0 = 1e3 * (time.perf_counter() - t0), 1
                    t0 = time.perf_counter()
                    for k in range(k0, S):
                        control_step(acts[k])
                    sync()
                    ms = 1e3 * (time.perf_counter() - t0) / (S - k0)
                    z = to_torch(robot.data.root_pos_w)[:, 2]
                    rounds.append({"warmup": r < args.warmup_rounds, "ms_per_control_step": ms,
                                   "contacts_end": contact_stats(), "base_z_mean_end": float(z.mean()),
                                   "frac_base_below_0p4m_end": float((z < 0.4).float().mean())})
                    print(f"[bench] {args.label} round {r}{' (warm-up)' if r < args.warmup_rounds else ''}: "
                          f"{ms:.3f} ms/control step, base z {float(z.mean()):.3f}", flush=True)
                # diagnostic 1: synchronised after every control step
                u.reset()
                acts = new_actions()
                per = []
                for k in range(S):
                    sync()
                    t0 = time.perf_counter()
                    control_step(acts[k])
                    sync()
                    per.append(1e3 * (time.perf_counter() - t0))
                # diagnostic 2: synchronised after every operation
                u.reset()
                acts = new_actions()
                ops = dict.fromkeys(["process_action", "apply_action", "write_data_to_sim", "sim_step", "scene_update"], 0.0)
                nb = min(S, 20)
                for k in range(nb):
                    sync()
                    t0 = time.perf_counter()
                    u.action_manager.process_action(acts[k])
                    sync()
                    ops["process_action"] += time.perf_counter() - t0
                    for _ in range(dec // steps_per_call):
                        t0 = time.perf_counter(); u.action_manager.apply_action(); sync(); t1 = time.perf_counter()
                        scene.write_data_to_sim(); sync(); t2 = time.perf_counter()
                        sim.step(render=False); sync(); t3 = time.perf_counter()
                        scene.update(dt=physics_dt * steps_per_call); sync(); t4 = time.perf_counter()
                        ops["apply_action"] += t1 - t0
                        ops["write_data_to_sim"] += t2 - t1
                        ops["sim_step"] += t3 - t2
                        ops["scene_update"] += t4 - t3
            timed = [x["ms_per_control_step"] for x in rounds if not x["warmup"]]
            med = pct(timed, 50)
            cont = [x["contacts_end"] for x in rounds if not x["warmup"] and x["contacts_end"]]
            rec["timing"] = {
                "ms_per_control_step_median": med,
                "ms_per_control_step_p10_rounds": pct(timed, 10), "ms_per_control_step_p90_rounds": pct(timed, 90),
                "ms_per_control_step_min_rounds": min(timed), "ms_per_control_step_max_rounds": max(timed),
                "env_steps_per_s_at_median": args.num_envs / (med * 1e-3),
                "physics_steps_per_control_step": int(dec),
                "rounds": rounds,
                "synced_each_control_step_ms": {"p10": pct(per, 10), "median": pct(per, 50), "p90": pct(per, 90)},
                "synced_each_op_ms_per_control_step": {k: 1e3 * v / nb for k, v in ops.items()},
                "first_control_step_ms": first_ms,
            }
            rec["contacts"] = {
                "bodies_in_contact_per_env_mean": sum(c["bodies_in_contact_per_env"] for c in cont) / len(cont) if cont else None,
                "envs_with_contact_frac_mean": sum(c["envs_with_contact_frac"] for c in cont) / len(cont) if cont else None,
                "frac_base_below_0p4m_end_mean": sum(x["frac_base_below_0p4m_end"] for x in rounds if not x["warmup"]) / len(timed),
                "note": "proxy from Isaac Lab's ContactSensor: bodies with net contact force > 1e-3 N, at round ends",
            }
            rec["startup_s"] = {"imports_and_cfg": t_app0 - t_imports0, "app_launch": t_app1 - t_app0,
                                "env_creation": t_env1 - t_env0, "process_start_to_env_ready": t_env1 - T_START}
            dev_used = smi("memory.used")
            rec["memory"] = {
                "gpu_process_mib_nvidia_smi": process_mem_mib(os.getpid()),
                "gpu_device_used_mib_end": float(dev_used) if dev_used else None,
                "gpu_device_used_delta_mib": (float(dev_used) - args.baseline_mem_mib)
                if dev_used and args.baseline_mem_mib is not None else None,
                "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            gpu = smi("name,driver_version,memory.total")
            rec["environment"] = {"gpu": gpu.split(", ")[0] if gpu else None,
                                  "driver": gpu.split(", ")[1] if gpu else None,
                                  "gpu_memory_total_mib": gpu.split(", ")[2] if gpu else None,
                                  "cpu": cpu_model(), "logical_cpus": os.cpu_count(),
                                  "python": platform.python_version(), "versions": versions()}
            # write before leaving the launch context: Kit may end the process on shutdown
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=1, default=str)
                f.flush()
                os.fsync(f.fileno())
            print(f"[bench] RESULT {args.label}: N={args.num_envs} median {med:.3f} ms/control step "
                  f"({args.num_envs / (med * 1e-3):.0f} env-steps/s)", flush=True)


if __name__ == "__main__":
    main()
