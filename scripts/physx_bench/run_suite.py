"""Run the PhysX baseline suite (called by run.sh; one subprocess per configuration) and write one JSON + one table.

Plan (``--mode full``), in this order; the last two are optional and skipped when the time budget would be exceeded
(estimated from the measured wall time of the N=8192 / N=4096 run of the same variant, x2.6):

  1  physx_task_n1024             physics=isaacsim_physx, Isaac Lab task defaults (G1_MINIMAL_CFG, self-collision off)
  2  physx_task_n4096
  3  physx_task_n8192
  4  physx_full_selfcoll_n4096    G1_CFG (full collision meshes) + self-collision on
  5  newton_task_n1024            physics=newton_mjwarp (Isaac Lab 3.0's default backend for this task), context
  6  newton_task_n4096
  7  official_runtime_physx_n4096 Isaac Lab's own `isaaclab benchmark runtime` (full env.step, random actions)
  8  physx_task_n16384            optional
  9  physx_full_selfcoll_n8192    optional

``--mode quick`` (validation): items 1, 4 and 5 at N=1024 with 2 short rounds, and the official benchmark at N=1024.

Before every configuration the GPU is checked to be idle (no compute processes, < 1 GiB in use) unless
``--no-idle-check``; a busy GPU stops the suite (exit 3). Outputs (in --out-dir, which must not exist yet or be empty):
  physx_bench_results.json   environment, versions, install choices, commands, per-configuration results
  physx_bench_results.md     the summary table
  runs/<label>.json          per-configuration result of bench_physics_only.py (incl. the live-stage settings dump)
  runs/official_runtime_*/   Isaac Lab's own benchmark output
  logs/<label>.log           full stdout/stderr of each configuration
Exit status: 0 when every mandatory configuration produced a result, 1 otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TASK = "Isaac-Velocity-Flat-G1"
LICENSE_NOTE = ("NVIDIA Isaac Sim / Omniverse license accepted by the operator before the run; "
                "these scripts never accept it")


def rel(p: str) -> str:
    return os.path.relpath(p, REPO)


def gpu_dev() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def smi(args: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60).stdout.strip()


def gpu_state() -> tuple[int, float]:
    dev = gpu_dev()
    uuid = smi(["--query-gpu=uuid", "--format=csv,noheader", "-i", dev])
    apps = smi(["--query-compute-apps=gpu_uuid", "--format=csv,noheader"])
    n = sum(1 for ln in apps.splitlines() if ln.strip() == uuid)
    used = float(smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", dev]) or 0)
    return n, used


def run_proc(cmd: list[str], log: str, cwd: str, timeout_s: float) -> tuple[int | None, float]:
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)
        try:
            rc = p.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = None
    return rc, time.time() - t0


def plan(mode: str) -> list[dict]:
    phx, nwt = "physics=isaacsim_physx", "physics=newton_mjwarp"
    if mode == "quick":
        r = ["--rounds", "2", "--steps_per_round", "20"]
        return [
            dict(label="physx_task_n1024", n=1024, args=[*r, phx]),
            dict(label="physx_full_selfcoll_n1024", n=1024, args=[*r, "--robot", "full", "--self_collision", "on", phx]),
            dict(label="newton_task_n1024", n=1024, args=[*r, nwt]),
            dict(label="official_runtime_physx_n1024", n=1024, official=True, steps=50, warmup=10, args=[phx]),
        ]
    full = ["--robot", "full", "--self_collision", "on"]
    return [
        dict(label="physx_task_n1024", n=1024, args=[phx]),
        dict(label="physx_task_n4096", n=4096, args=[phx]),
        dict(label="physx_task_n8192", n=8192, args=[phx]),
        dict(label="physx_full_selfcoll_n4096", n=4096, args=[*full, phx]),
        dict(label="newton_task_n1024", n=1024, args=[nwt]),
        dict(label="newton_task_n4096", n=4096, args=[nwt]),
        dict(label="official_runtime_physx_n4096", n=4096, official=True, steps=300, warmup=50, args=[phx]),
        dict(label="physx_task_n16384", n=16384, args=[phx], optional=True, estimate_from="physx_task_n8192"),
        dict(label="physx_full_selfcoll_n8192", n=8192, args=[*full, phx], optional=True,
             estimate_from="physx_full_selfcoll_n4096"),
    ]


def describe(item: dict) -> dict:
    a = item["args"]
    backend = "PhysX (isaacsim_physx)" if "physics=isaacsim_physx" in a else "Newton / MuJoCo Warp (newton_mjwarp)"
    full = "--robot" in a and a[a.index("--robot") + 1] == "full"
    selfc = "--self_collision" in a and a[a.index("--self_collision") + 1] == "on"
    return {"backend": backend,
            "asset": "G1_CFG (g1.usd, 38 colliders: 32 convex hulls, 4 cylinders, 2 boxes)" if full
            else "G1_MINIMAL_CFG (g1_minimal.usd, 3 box colliders: both feet, torso)",
            "self_collision": "on" if selfc else "off (task default)"}


def parse_official(out_dir: str) -> dict:
    files = glob.glob(os.path.join(out_dir, "*_json.json"))
    if not files:
        return {}
    data = json.load(open(files[0]))
    single = {}
    for ph in data:
        for m in ph.get("measurements", []):
            if m.get("type") == "single":
                single[m["name"].replace("benchmark_runtime ", "")] = {"value": m.get("value"), "unit": m.get("unit")}
    fps = next((v["value"] for k, v in single.items() if k.startswith("runtime Mean") and k.endswith("Total FPS")), None)
    it = next((v["value"] for k, v in single.items() if k.startswith("runtime Mean") and "Iteration Time" in k), None)
    phys = single.get("frametime Mean Physics Frametime", {}).get("value")
    return {"env_steps_per_s_mean": fps, "env_step_ms_mean": it, "physics_frametime_ms_mean": phys,
            "measurements": single, "files": [rel(f) for f in files]}


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(res: dict) -> str:
    env = res.get("environment") or {}
    lines = [
        "# PhysX baseline (Isaac Lab 3.0 / Isaac Sim 6.2), Unitree G1, physics only",
        "",
        f"GPU {env.get('gpu')} (driver {env.get('driver')}), CPU {env.get('cpu')}; "
        f"isaaclab {env.get('versions', {}).get('isaaclab')}, isaacsim {env.get('versions', {}).get('isaacsim')}, "
        f"torch {env.get('versions', {}).get('torch')}; mode {res['mode']}, {res['created_utc']}",
        "",
        "Control step = 4 physics steps of 5 ms (20 ms simulated); random PD targets held per control step; robots "
        "reset at the start of each round and fall. NOT the same physics problem as an MJCF model: see README.",
        "",
        "| configuration | backend | asset | self-collision | N | ms / control step: median (p10-p90 of rounds) "
        "| env-steps/s | sim.step ms (synced per op) | GPU MiB | startup s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"]:
        d = r.get("describe", {})
        if r.get("status") != "ok":
            lines.append(f"| {r['label']} | {d.get('backend', '')} | | | {r.get('n')} | {r.get('status')} | | | | |")
            continue
        if r.get("official"):
            o = r["official"]
            lines.append(f"| {r['label']} (Isaac Lab runtime benchmark, full env.step) | {d.get('backend')} | "
                         f"{d.get('asset', '').split(' ')[0]} | {d.get('self_collision')} | {r['n']} | "
                         f"{fmt(o.get('env_step_ms_mean'))} (mean env.step) | {fmt(o.get('env_steps_per_s_mean'), 0)} | "
                         f"{fmt(o.get('physics_frametime_ms_mean'), 2)} per physics step | | |")
            continue
        t = r["result"]["timing"]
        m = r["result"].get("memory", {})
        mem = m.get("gpu_process_mib_nvidia_smi") or m.get("gpu_device_used_delta_mib")
        lines.append(
            f"| {r['label']} | {d.get('backend')} | {d.get('asset', '').split(' ')[0]} | {d.get('self_collision')} | "
            f"{r['n']} | {fmt(t['ms_per_control_step_median'], 2)} ({fmt(t['ms_per_control_step_p10_rounds'], 2)}-"
            f"{fmt(t['ms_per_control_step_p90_rounds'], 2)}) | {fmt(t['env_steps_per_s_at_median'], 0)} | "
            f"{fmt(t['synced_each_op_ms_per_control_step'].get('sim_step'), 2)} | {fmt(mem, 0)} | "
            f"{fmt(r['result']['startup_s']['process_start_to_env_ready'], 0)} |")
    if res.get("skipped"):
        lines += ["", "Skipped (time budget): " + ", ".join(res["skipped"])]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "quick"], default="full")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget-min", type=float, default=float(os.environ.get("PHYSX_BENCH_BUDGET_MIN", 55)))
    ap.add_argument("--no-idle-check", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out_dir)
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"run_suite: {out} exists and is not empty; choose a new --out-dir")
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)

    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", rel(HERE)],
                                capture_output=True, text=True).stdout.strip())
    res = {
        "schema": "physx_bench/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": a.mode, "budget_min": a.budget_min, "task": TASK,
        "provenance": {"repo_commit": commit, "scripts_dirty": dirty, "scripts": rel(HERE)},
        "install": {"torch_backend": os.environ.get("PHYSX_BENCH_TORCH_BACKEND_USED"),
                    "driver_below_isaaclab_minimum_580_65": os.environ.get("PHYSX_BENCH_DRIVER_BELOW_MIN") == "1"},
        "license": LICENSE_NOTE + f" ({os.environ.get('PHYSX_BENCH_EULA_HOW', 'unknown')})",
        "method": ("bench_physics_only.py: gym.make of the registered task; per control step one action a~U[-1,1]^37 "
                   "(JointPositionAction: target = default pose + 0.5 a, held), then decimation x (apply_action, "
                   "scene.write_data_to_sim, sim.step(render=False), scene.update); no MDP managers or resets inside "
                   "the timed loop; env.reset() before each round; median over timed rounds, sync at round ends"),
        "idle_check": not a.no_idle_check,
        "runs": [], "skipped": [], "environment": None,
    }
    t_suite = time.time()
    walls: dict[str, float] = {}
    failed = False

    def save():
        with open(os.path.join(out, "physx_bench_results.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        with open(os.path.join(out, "physx_bench_results.md"), "w") as f:
            f.write(table(res))

    for item in plan(a.mode):
        label = item["label"]
        elapsed = time.time() - t_suite
        remaining = a.budget_min * 60 - elapsed
        if item.get("optional"):
            base = walls.get(item.get("estimate_from", ""))
            est = 2.6 * base if base else None
            if est is None or est > remaining:
                print(f"[suite] skip {label}: estimated {fmt(est and est / 60)} min > remaining {remaining / 60:.1f} min",
                      flush=True)
                res["skipped"].append(label)
                continue
        if remaining <= 60:
            print(f"[suite] skip {label}: time budget used up", flush=True)
            res["skipped"].append(label)
            failed = failed or not item.get("optional")
            continue
        if not a.no_idle_check:
            n, used = gpu_state()
            if n or used > 1024:
                print(f"[suite] GPU {gpu_dev()} is busy ({n} processes, {used:.0f} MiB in use); stopping. "
                      f"Use --no-idle-check on an exclusive GPU.", flush=True)
                res["stopped_reason"] = "gpu busy"
                save()
                sys.exit(3)
        _, base_mem = gpu_state()
        log = os.path.join(out, "logs", f"{label}.log")
        entry = {"label": label, "n": item["n"], "describe": describe(item), "official": item.get("official", False)}
        if item.get("official"):
            odir = os.path.join(out, "runs", label)
            os.makedirs(odir, exist_ok=True)
            cmd = ["isaaclab", "benchmark", "runtime", "--task", TASK, "--num_envs", str(item["n"]),
                   "--num_steps", str(item["steps"]), "--warmup_steps", str(item["warmup"]),
                   "--benchmark_formatter", "json,summary", "--output_path", ".", *item["args"]]
            entry["command"] = " ".join(cmd)
            print(f"[suite] {label}: {entry['command']}", flush=True)
            rc, wall = run_proc(cmd, log, odir, remaining)
            o = parse_official(odir)
            entry.update(status="ok" if rc == 0 and o else ("timeout" if rc is None else f"failed rc={rc}"),
                         wall_s=wall, official=o or True)
        else:
            jpath = os.path.join(out, "runs", f"{label}.json")
            cmd = [sys.executable, os.path.join(HERE, "bench_physics_only.py"), "--task", TASK, "--num_envs",
                   str(item["n"]), "--label", label, "--out", jpath, "--baseline_mem_mib", str(base_mem), *item["args"]]
            entry["command"] = " ".join(["python", rel(cmd[1]), *cmd[2:]]).replace(jpath, f"runs/{label}.json")
            print(f"[suite] {label}: {entry['command']}", flush=True)
            rc, wall = run_proc(cmd, log, out, remaining)
            ok = rc == 0 and os.path.isfile(jpath)
            entry.update(status="ok" if ok else ("timeout" if rc is None else f"failed rc={rc}"), wall_s=wall)
            if ok:
                entry["result"] = json.load(open(jpath))
                res["environment"] = res["environment"] or entry["result"].get("environment")
                t = entry["result"]["timing"]
                print(f"[suite] {label}: {t['ms_per_control_step_median']:.2f} ms/control step "
                      f"({t['env_steps_per_s_at_median']:.0f} env-steps/s), wall {wall / 60:.1f} min", flush=True)
        if entry["status"] == "ok":
            walls[label] = entry["wall_s"]
        else:
            print(f"[suite] {label}: {entry['status']} (see {rel(log) if log.startswith(REPO) else log})", flush=True)
            failed = failed or not item.get("optional")
        res["runs"].append(entry)
        save()
    res["suite_wall_min"] = (time.time() - t_suite) / 60
    save()
    print(table(res))
    print(f"[suite] wrote {os.path.join(out, 'physx_bench_results.json')} ({res['suite_wall_min']:.1f} min)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
