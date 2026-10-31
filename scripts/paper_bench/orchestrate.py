"""The paper timing kit's driver (scripts/paper_bench/README.md): runs every part as its own process in its own
environment, with a per-part timeout and a whole-job time budget, samples the GPU with a separate nvidia-smi process
during each part, and writes <out>/paper_bench.json (every part's status, time, log, clocks, its result JSON embedded,
the headline table) plus <out>/parts/<part>.json and <out>/logs/<part>.log. A failing part is recorded with its error
and the job goes on; the exit status is 1 if any part failed (2 if nothing ran).

Run by run.sh in the main environment (NablaEngine's dependencies, JAX with CUDA and mujoco-mjx; src/ of this checkout
on PYTHONPATH):
  python scripts/paper_bench/orchestrate.py --out <out_dir>

A local_parts.py next to this file (common.LOCAL), when present, may add parts: its extend(ctx) is called with the
part list and the environments before the part selection (PAPER_BENCH_PARTS); this kit has no such file.
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

REPO = C.REPO
ENV_DOC = ("PAPER_BENCH_PREFIX", "PAPER_BENCH_QUICK", "PAPER_BENCH_PARTS", "PAPER_BENCH_CORPORA", "PAPER_BENCH_SWEEP",
           "PAPER_BENCH_BUDGET_MIN", "PAPER_BENCH_TIMEOUT_SCALE", "PAPER_BENCH_SKIP_SETUP")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Sampler:
    """nvidia-smi --query-gpu ... -lms 100 in the background for one part."""
    BASE = ["timestamp", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu", "memory.used"]
    REASONS = ("active", "hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown",
               "sw_power_cap")

    def __init__(self, idx):
        self.idx, self.proc, self.fields = idx, None, list(self.BASE)
        for pfx in ("clocks_event_reasons", "clocks_throttle_reasons"):
            f = [f"{pfx}.{r}" for r in self.REASONS]
            try:
                r = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=" + ",".join(f), "--format=csv,noheader"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                return
            if r.returncode == 0 and "Field" not in r.stdout and "not a valid" not in r.stdout:
                self.fields += f
                return

    def start(self, path):
        self.path = Path(path)
        try:
            self.fh = open(self.path, "w")
            self.proc = subprocess.Popen(["nvidia-smi", "-i", self.idx, "--query-gpu=" + ",".join(self.fields),
                                          "--format=csv,noheader,nounits", "-lms", "100"], stdout=self.fh,
                                         stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None

    def stop(self):
        if not self.proc:
            return None
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.fh.close()
        return self.summary()

    def summary(self):
        rows = []
        for line in self.path.read_text().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) != len(self.fields):
                continue
            rows.append(dict(zip(self.fields, p)))
        if not rows:
            return None

        def num(r, k):
            try:
                return float(r[k])
            except (KeyError, ValueError):
                return None
        busy = [r for r in rows if (num(r, "utilization.gpu") or 0) > 0] or rows
        sm = sorted(x for x in (num(r, "clocks.sm") for r in busy) if x is not None)
        pw = [x for x in (num(r, "power.draw") for r in rows) if x is not None]
        tc = [x for x in (num(r, "temperature.gpu") for r in rows) if x is not None]
        mem = [x for x in (num(r, "memory.used") for r in rows) if x is not None]
        flags = {}
        for f in self.fields:
            if f.endswith(("hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown",
                           "sw_power_cap")):
                n = sum(1 for r in rows if r.get(f, "").lower() == "active")
                if n:
                    flags[f.split(".", 1)[1]] = n
        flagged = {k: v for k, v in flags.items() if k.startswith("hw_")}
        return dict(samples=len(rows), busy_samples=len(busy), sm_mhz_median=sm[len(sm) // 2] if sm else None,
                    sm_mhz_min=sm[0] if sm else None, sm_mhz_max=sm[-1] if sm else None,
                    power_w_max=max(pw) if pw else None, temp_c_max=max(tc) if tc else None,
                    memory_used_mib_max=max(mem) if mem else None, reason_samples=flags,
                    hardware_slowdown_flagged=bool(flagged), csv=C.rel(self.path))


def other_gpu_processes(idx, session):
    """Compute processes on the GPU outside the worker's session (contention), from nvidia-smi; [] when none."""
    try:
        r = subprocess.run(["nvidia-smi", "-i", idx, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    out = []
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if not (p and p[0].isdigit()):
            continue
        try:
            ours = os.getsid(int(p[0])) == session
        except OSError:
            ours = False
        if not ours:
            out.append(dict(used_memory=p[1] if len(p) > 1 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--setup-status", default="ok", help="run.sh: ok, skipped, or failed:<code>")
    a = ap.parse_args()
    t_job = time.perf_counter()
    try:  # the job started before the setup (run.sh): the budget covers the setup too
        t_job -= max(0.0, time.time() - float(os.environ.get("PAPER_BENCH_JOB_START") or time.time()))
    except ValueError:
        pass
    out = Path(a.out).absolute()
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    prefix = Path(os.environ.get("PAPER_BENCH_PREFIX") or REPO / "runs" / "paper_bench_env").absolute()
    quick = C.QUICK
    budget_s = 60 * float(os.environ.get("PAPER_BENCH_BUDGET_MIN") or 53)
    tscale = float(os.environ.get("PAPER_BENCH_TIMEOUT_SCALE") or 1)
    sweep = [int(x) for x in (os.environ.get("PAPER_BENCH_SWEEP") or "512,1024,4096,16384").split(",") if x.strip()]
    corp = os.environ.get("PAPER_BENCH_CORPORA") or ",".join(C.corpora())
    first = corp.split(",")[0]
    gidx = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
    if "," in (os.environ.get("CUDA_VISIBLE_DEVICES") or ""):
        print("paper_bench: more than one GPU visible; every part uses the first", file=sys.stderr)

    dev_py = sys.executable
    mjw_py = os.environ.get("PAPER_BENCH_MJW_PYTHON") or str(prefix / "venv_mjw" / "bin" / "python")
    pr_py = os.environ.get("PAPER_BENCH_PR_PYTHON") or str(prefix / "venv_pr" / "bin" / "python")
    # Genesis (part genesis): its own environment (benchmarks/envs/setup.sh --envs genesis); the part is skipped
    # and recorded as not set up when no interpreter is found
    gen_py = os.environ.get("PAPER_BENCH_GENESIS_PYTHON") or next(
        (str(x) for x in (prefix / "venv_genesis" / "bin" / "python",
                          Path(os.environ.get("DK_BENCH_ENV_PREFIX") or REPO / "runs" / "bench" / "envs")
                          / "genesis" / "venv" / "bin" / "python") if x.exists()),
        str(prefix / "venv_genesis" / "bin" / "python"))
    trees = {c: os.environ.get(f"PAPER_BENCH_PR_TREE_{c.upper()}") or str(prefix / "src" / f"mujoco_warp_{c}")
             for c in ("357a75d", "4af093f")}
    base_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    base_env.update(PYTHONNOUSERSITE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", DK_FIXTURES=str(C.FIXTURES),
                    PAPER_BENCH_PREFIX=str(prefix))
    base_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # CUDA device i = nvidia-smi -i i (the GPU sampler)
    src = str(REPO / "src")
    envs = {"dev": dict(base_env, PYTHONPATH=src), "mjw": dict(base_env),
            "triton": dict(base_env, PYTHONPATH=src, XLA_FLAGS=" ".join(
                x for x in (base_env.get("XLA_FLAGS"), "--xla_gpu_triton_gemm_any=true") if x)),
            "pr357": dict(base_env, PYTHONPATH=f"{trees['357a75d']}{os.pathsep}{src}"),
            "pr4af": dict(base_env, PYTHONPATH=f"{trees['4af093f']}{os.pathsep}{src}"),
            "genesis": dict(base_env, PYTHONPATH=str(REPO))}
    pys = {"dev": dev_py, "triton": dev_py, "mjw": mjw_py, "pr357": pr_py, "pr4af": pr_py, "genesis": gen_py}
    # one Warp kernel cache per interpreter: two builds of the same warp-lang release (NVIDIA's +cu12 wheel and the
    # default CUDA 13 wheel) share a cache key, and a module cached by one can fail to link in the other
    wcache = Path(base_env.get("WARP_CACHE_PATH") or prefix / "warp_cache")

    def warp_cache(py):
        return str(wcache / {dev_py: "dev", mjw_py: "venv_mjw", pr_py: "venv_pr"}.get(py, "other"))
    for env, py in pys.items():
        envs[env]["WARP_CACHE_PATH"] = warp_cache(py)
    need = {"mjw": [mjw_py], "pr357": [pr_py, trees["357a75d"]], "pr4af": [pr_py, trees["4af093f"]], "dev": [],
            "triton": [], "genesis": [gen_py]}
    S = lambda name: str(HERE / name)  # noqa: E731
    P = lambda part: str(out / "parts" / f"{part}.json")  # noqa: E731
    cflag = ["--corpora", corp]
    sw_other = [n for n in sweep if n != 4096]
    parts = []  # (name, env, argv, timeout_s, minimum_s)
    parts.append(("ours", "dev", [S("ours.py"), "--out", P("ours"), *cflag, "--sweep", ",".join(map(str, sweep))],
                  900, 60))
    parts.append(("ours_bare", "dev", [S("ours_bare.py"), "--out", P("ours_bare"), "--corpus", first], 600, 60))
    parts.append(("mjw_native", "mjw", [S("mjw_native.py"), "--out", P("mjw_native"), *cflag], 600, 60))
    for n in sw_other:
        parts.append((f"mjw_native_nw{n}", "mjw", [S("mjw_native.py"), "--out", P(f"mjw_native_nw{n}"), "--corpora",
                                                   first, "--nworld", str(n), "--capacity", "procedure,matched"],
                      300, 30))
    common_adj = ["--iterations", "10", "--nconmax", "64", "--njmax", "192", *cflag]
    # PR #1535: its forward and its adjoint-graph pattern, as published (357a75d) and with the two public fixes
    parts.append(("mjw_pr1535", "pr357", [S("mjw_adjoint.py"), "--label", "mjw_pr1535", "--out", P("mjw_pr1535"),
                                          *common_adj, "--split"], 900, 60))
    parts.append(("mjw_pr1535_fix", "pr4af", [S("mjw_adjoint.py"), "--label", "mjw_pr1535_fix", "--out",
                                              P("mjw_pr1535_fix"), *common_adj, "--split"], 900, 60))
    jb = "16" if quick else (os.environ.get("PAPER_BENCH_MJX_JAC_BATCHES") or "64,256,1024")
    parts.append(("mjx", "dev", [S("mjx_native.py"), "--out", P("mjx"), *cflag, "--jac-batches", jb], 1500, 120))
    parts.append(("mjc", "dev", [S("mjc.py"), "--out", P("mjc"), *cflag], 900, 60))
    parts.append(("genesis", "genesis", [S("genesis_native.py"), "--out", P("genesis"), *cflag], 900, 60))
    # MJX with XLA's Triton GEMM for every matmul (MJX's forward under the other XLA GEMM setting): forward only
    parts.append(("mjx_triton", "triton", [S("mjx_native.py"), "--out", P("mjx_triton"), "--corpora", first,
                                           "--parts", "mjlab"], 600, 60))
    # P = 3 processes for the headline cells (4096 worlds, 10 iterations, the headline corpora): two more processes per
    # engine with only those cells (process-to-process variation), last in the job (the budget skips them first)
    head = [c for c in corp.split(",") if c in C.HEADLINE] or [first]
    hflag = ["--corpora", ",".join(head)]
    for r in (2, 3):
        parts += [
            (f"rep{r}_ours", "dev", [S("ours.py"), "--out", P(f"rep{r}_ours"), *hflag, "--no-jac", "--sweep", ""],
             600, 30),
            (f"rep{r}_mjw_native", "mjw", [S("mjw_native.py"), "--out", P(f"rep{r}_mjw_native"), *hflag, "--capacity",
                                           "procedure,matched"], 300, 30),
            (f"rep{r}_mjw_pr1535", "pr357", [S("mjw_adjoint.py"), "--label", "mjw_pr1535", "--out",
                                             P(f"rep{r}_mjw_pr1535"), "--iterations", "10", "--nconmax", "64",
                                             "--njmax", "192", *hflag], 400, 30),
            (f"rep{r}_mjw_pr1535_fix", "pr4af", [S("mjw_adjoint.py"), "--label", "mjw_pr1535_fix", "--out",
                                                 P(f"rep{r}_mjw_pr1535_fix"), "--iterations", "10", "--nconmax", "64",
                                                 "--njmax", "192", *hflag], 400, 30),
            (f"rep{r}_mjx", "dev", [S("mjx_native.py"), "--out", P(f"rep{r}_mjx"), *hflag, "--parts", "mjlab"], 600, 60),
            (f"rep{r}_mjc", "dev", [S("mjc.py"), "--out", P(f"rep{r}_mjc"), *hflag, "--no-fd"], 300, 30),
            (f"rep{r}_genesis", "genesis", [S("genesis_native.py"), "--out", P(f"rep{r}_genesis"), *hflag], 600, 60)]
    job_extra = {}
    if C.LOCAL is not None and hasattr(C.LOCAL, "extend"):  # local additions (module docstring)
        C.LOCAL.extend(dict(parts=parts, envs=envs, pys=pys, need=need, base_env=base_env, src=src, prefix=prefix,
                            warp_cache=warp_cache, script=S, result=P, corpora=corp, first=first, sweep=sweep,
                            headline=head, quick=quick, job=job_extra))
    sel = os.environ.get("PAPER_BENCH_PARTS")
    if sel:
        want = sel.split(",")
        parts = [p for p in parts if p[0] in want or p[0].split("_nw")[0] in want
                 or (p[0].startswith("rep") and ("reps" in want or p[0].split("_", 1)[1] in want))]
    nrep = int(os.environ.get("PAPER_BENCH_REPEATS") or (1 if quick else 2))  # extra processes per headline cell
    parts = [p for p in parts if not p[0].startswith("rep") or int(p[0][3:].split("_")[0]) <= nrep + 1]

    job = dict(schema="paper-bench/1", kit="scripts/paper_bench", started=now(), quick=quick, out_dir=C.rel(out),
               setup_status=a.setup_status, budget_min=budget_s / 60, corpora=corp.split(","), sweep=sweep,
               corpus_kind={c: C.corpus_kind(c) for c in corp.split(",")},
               headline_missing=[c for c in C.HEADLINE if c not in C.corpora()], **job_extra,
               job_start_includes_setup=bool(os.environ.get("PAPER_BENCH_JOB_START")),
               gpu_index_for_nvidia_smi="first CUDA_VISIBLE_DEVICES entry (CUDA_DEVICE_ORDER=PCI_BUS_ID)", parts={},
               environment={k: (C.rel(v) if "/" in v else v) for k, v in os.environ.items()
                            if k in ENV_DOC or k.startswith("PAPER_BENCH_")})
    try:
        from nablaengine import provenance
        job["provenance"] = provenance.provenance()
    except Exception as e:  # noqa: BLE001
        job["provenance"] = dict(error=repr(e)[:300])
    job["system"] = C.basic_env(("jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang", "mujoco-warp", "numpy"))
    sj = prefix / "setup.json"
    job["setup"] = json.loads(sj.read_text()) if sj.exists() else None
    try:  # the release rule: the default (CUDA 13) warp wheel where the driver supports it
        drv = int(str((job["system"].get("gpu") or {}).get("driver", "0")).split(".")[0])
        if job["setup"] and job["setup"].get("warp_wheel") == "cu12" and drv >= 580:
            job["warp_wheel_note"] = (f"the native venvs use the +cu12 warp wheel on driver {drv} (>= 580: the "
                                      "default wheel is the rule; rerun setup.sh with PAPER_BENCH_WARP_WHEEL=default)")
    except ValueError:
        pass
    for k in ("357a75d", "4af093f"):
        tp = Path(trees[k])
        ok = tp / ".paper_bench_ok"
        commit = ok.read_text().strip() if ok.exists() else None
        if commit is None and (tp / ".git").exists():
            r = subprocess.run(["git", "-C", str(tp), "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, universal_newlines=True)
            commit = r.stdout.strip() or None
        job.setdefault("trees", {})[k] = dict(path=C.rel(tp), commit=commit)
        try:
            from nablaengine import provenance as pv
            job["trees"][k]["tree_sha256"] = pv.tree_sha256(tp / "mujoco_warp") if (tp / "mujoco_warp").is_dir() else None
        except Exception:  # noqa: BLE001
            pass
    path = out / "paper_bench.json"

    def save():
        job["seconds"] = round(time.perf_counter() - t_job, 1)
        try:
            import summarize
            job["table"] = summarize.table(job)
        except Exception as e:  # noqa: BLE001
            job["table_error"] = repr(e)[:500]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(C.scrub_text(json.dumps(job, indent=1, default=str)))
        os.replace(tmp, path)

    if job["headline_missing"]:
        print(f"paper_bench: WARNING headline corpora missing: {job['headline_missing']} (benchmarks/data); the "
              f"timings use {corp}", file=sys.stderr)
    save()
    sampler = Sampler(gidx)
    for name, env, argv, tmo, tmin in parts:
        rec = dict(env=env, python=C.rel(pys[env]), argv=[C.rel(x) if "/" in x else x for x in argv],
                   started=now(), log=C.rel(out / "logs" / f"{name}.log"))
        job["parts"][name] = rec
        left = budget_s - (time.perf_counter() - t_job)
        missing = [C.rel(x) for x in need[env] if not Path(x).exists()]
        if missing:
            rec.update(status="skipped", reason=f"environment missing: {missing} (setup {a.setup_status})")
            save()
            continue
        if left < tmin:
            rec.update(status="skipped", reason=f"time budget: {left:.0f} s left, part needs >= {tmin} s")
            save()
            continue
        timeout = min(tmo * tscale * (2 if quick else 1), left)
        rec["timeout_s"] = round(timeout)
        save()
        print(f"[paper_bench {datetime.datetime.now():%H:%M:%S}] {name} (timeout {timeout:.0f} s)", flush=True)
        sampler.start(out / "logs" / f"{name}.smi.csv")
        t0 = time.perf_counter()
        with open(out / "logs" / f"{name}.log", "w") as log:
            p = subprocess.Popen([pys[env], *argv], cwd=str(REPO), env=envs[env], stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:  # after a few seconds (the worker has its CUDA context): who else uses the GPU
                rc = p.wait(timeout=min(15.0, timeout))
            except subprocess.TimeoutExpired:
                rc = None
                rec["other_gpu_processes"] = other_gpu_processes(gidx, p.pid)
            try:
                if rc is None:
                    rc = p.wait(timeout=max(1.0, timeout - (time.perf_counter() - t0)))
                rec["status"] = "ok" if rc == 0 else "failed"
                rec["returncode"] = rc
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rec["status"] = "timeout"
                rec["returncode"] = None
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        # and right after the part (short parts end before the first look): contention evidence for every part
        rec["other_gpu_processes_after"] = other_gpu_processes(gidx, p.pid)
        if "other_gpu_processes" not in rec:
            rec["other_gpu_processes"] = rec["other_gpu_processes_after"]
        rec["gpu_samples"] = sampler.stop()
        pj = Path(P(name))
        if pj.exists():
            try:
                rec["result"] = json.loads(pj.read_text())
                rec["result_file"] = C.rel(pj)
                if rec["status"] == "failed" and rec["result"].get("errors"):
                    rec["error"] = {k: v.get("message") for k, v in rec["result"]["errors"].items()}
            except json.JSONDecodeError as e:
                rec["result_error"] = repr(e)
        if rec["status"] != "ok" and "error" not in rec:
            try:
                rec["error_tail"] = (out / "logs" / f"{name}.log").read_text(errors="replace")[-2000:]
            except OSError:
                pass
        print(f"[paper_bench {datetime.datetime.now():%H:%M:%S}] {name}: {rec['status']} in {rec['seconds']} s",
              flush=True)
        save()
    job["finished"] = now()
    save()
    st = [r.get("status") for r in job["parts"].values()]
    if not any(s == "ok" for s in st):
        return 2
    return 0 if all(s == "ok" for s in st) and a.setup_status in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
