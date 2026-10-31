"""Shared helpers of the paper timing kit (scripts/paper_bench): the timing protocol, corpus inputs, small provenance.

Standard library + numpy only, so every engine's environment can import it (the native MuJoCo Warp venv has no
nablaengine). Workers import it from their own directory (scripts/paper_bench is on sys.path when a worker runs).

Timing protocol: compile / capture / build timed separately by the worker; warm-up of at least WARM_CALLS calls and WARM_S seconds; then ROUNDS + 1 rounds of N back-to-back calls and one
synchronization each, round 0 dropped; per-call time of each round = round time / N; reported: median, p10, p90, min,
max over the kept rounds (ms), plus a latency mode (median of LAT single synchronized calls) where cheap and where
the rounds are not single calls already (n > 1).
PAPER_BENCH_QUICK=1 reduces rounds and calls (kit smoke test; the numbers are then not for publication).
"""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "mjlab"
MF_DIR = REPO / "benchmarks" / "data"
QUICK = os.environ.get("PAPER_BENCH_QUICK", "") not in ("", "0")
ROUNDS = 2 if QUICK else 5  # kept rounds (round 0 is extra and dropped)
WARM_CALLS, WARM_S = (2, 0.2) if QUICK else (5, 1.0)
STATE_KEYS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")


def rel(path):
    """A path relative to the repository when inside it, else '.../<name>' (no host paths in results)."""
    if path is None:
        return None
    p = Path(path).absolute()
    for q in (p, p.resolve()):
        try:
            return str(q.relative_to(REPO))
        except ValueError:
            continue
    return f".../{p.name}"


def scrub_text(text):
    """Host paths in free text (tracebacks, log tails, messages) made generic: the checkout -> '', $HOME -> '~',
    $PAPER_BENCH_PREFIX -> '$PAPER_BENCH_PREFIX', the temporary directory -> '$TMPDIR'."""
    import tempfile
    subs = []
    pref = os.environ.get("PAPER_BENCH_PREFIX")
    if pref:
        subs.append((str(Path(pref).absolute()), "$PAPER_BENCH_PREFIX"))
    subs += [(str(REPO) + "/", ""), (str(REPO), "."), (tempfile.gettempdir().rstrip("/") + "/", "$TMPDIR/"),
             (str(Path.home()) + "/", "~/")]
    for a, b in subs:
        if len(a) > 1:
            text = text.replace(a, b)
    return text


HEADLINE = ("mf_s0", "mf_s1")  # the corpora timed by default (benchmarks/data), in this order


def _local():
    """local_parts.py next to this file, when present: additions of a local installation (extra corpora, parts and
    table rows) that are not part of this kit; None otherwise."""
    try:
        import local_parts
    except ImportError:
        return None
    return local_parts


LOCAL = _local()


def corpora(names=None):
    """{name: path} of the timing corpora. Default: HEADLINE (benchmarks/data/<name>.npz) that exist, in that order;
    the first one is the "first corpus" of the sweeps, Jacobians and splits. names (comma-separated or a list) selects
    (and orders) among every benchmarks/data/mf_s*.npz (and the corpora local_parts.py adds)."""
    allc = {p.stem: p for p in sorted(MF_DIR.glob("mf_s*.npz"))}
    extra = dict(getattr(LOCAL, "CORPORA", {}) or {})
    allc.update({n: Path(v) for n, v in extra.items()})
    if names:
        want = names.split(",") if isinstance(names, str) else list(names)
        missing = [n for n in want if n not in allc]
        if missing:
            raise FileNotFoundError(f"corpora not available: {missing} (have {sorted(allc)})")
        return {n: allc[n] for n in want}
    default = {n: allc[n] for n in HEADLINE if n in allc}
    default.update({n: allc[n] for n in getattr(LOCAL, "DEFAULT_CORPORA", ()) if n in allc})
    return default


def corpus_kind(name):
    """'benchmark' for the corpora of benchmarks/data, else the kind local_parts.py gives."""
    kinds = getattr(LOCAL, "CORPUS_KIND", {}) or {}
    return kinds.get(name, "benchmark")


def load_states(name, path=None, nworld=None, dtype=np.float32):
    """Inputs of one control step: qpos, qvel, qacc_warmstart, ctrl (B, .), friction (B,), com (B, 3), tiled to nworld
    (the first nworld states when the corpus has more); plus cot_qpos / cot_qvel (the corpus's stored cotangent 0:
    cot_*[:, 0] of a benchmark corpus, cot0_* of <name>_adj.npz for a corpus in the fixture format; None if absent)."""
    path = Path(path or corpora([name])[name])
    z = np.load(path)
    out = {k: np.asarray(z[k]) for k in STATE_KEYS}
    cq = cv = None
    if "cot_qpos" in z.files:
        cq, cv = np.asarray(z["cot_qpos"])[:, 0], np.asarray(z["cot_qvel"])[:, 0]
    else:
        adj = path.with_name(path.name.replace("_fwd.npz", "_adj.npz"))
        if adj.exists():
            za = np.load(adj)
            cq, cv = np.asarray(za["cot0_qpos"]), np.asarray(za["cot0_qvel"])
    out["cot_qpos"], out["cot_qvel"] = cq, cv
    n0 = len(out["qpos"])
    n = nworld or n0
    idx = np.arange(n) % n0
    res = {k: (None if v is None else np.ascontiguousarray(v[idx].astype(dtype))) for k, v in out.items()}
    res["_source"] = dict(name=name, path=rel(path), states=n0, nworld=n, tiled=n > n0)
    return res


def meta():
    return json.loads((FIXTURES / "g1_meta.json").read_text())


def is_oom(exc):
    """An out-of-memory failure (XLA RESOURCE_EXHAUSTED, CUDA / Warp out of memory): recorded as skipped: oom."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(k in msg for k in ("resource_exhausted", "out of memory", "failed to allocate", "outofmemory"))


HEADLINE_NWORLD = 4096


def record_oom(res, key, exc, nworld, **cell):
    """An out-of-memory cell: recorded as skipped: oom; at the headline size or below (nworld <= 4096, which fits on
    every target card) it is also an error, so the part and the job report the missing headline number."""
    res.cell(key, dict(cell, skipped="oom", message=str(exc)[:300]))
    if nworld <= HEADLINE_NWORLD:
        res.error(key, RuntimeError(f"out of memory at nworld {nworld} (another process on the GPU?): {str(exc)[:500]}"))


def stats_ms(per_call_s):
    a = 1e3 * np.asarray(per_call_s, np.float64)
    return dict(median=round(float(np.median(a)), 5), p10=round(float(np.percentile(a, 10)), 5),
                p90=round(float(np.percentile(a, 90)), 5), min=round(float(a.min()), 5), max=round(float(a.max()), 5),
                rounds=[round(float(x), 5) for x in a])


def timeit(call, sync, n=None, rounds=None, warm_calls=None, warm_s=None, latency=0, target_s=0.25):
    """The protocol above for call() (dispatch only) and sync() (wait for the device): returns a dict with ms per call
    (stats_ms over the kept rounds), n, the warm-up done and, with latency > 0, the median of `latency` single
    synchronized calls (ms). n None: chosen from the warm-up's time per call so a round lasts about target_s."""
    rounds = ROUNDS if rounds is None else rounds
    warm_calls = WARM_CALLS if warm_calls is None else warm_calls
    warm_s = WARM_S if warm_s is None else warm_s
    t0, k = time.perf_counter(), 0
    while k < warm_calls or time.perf_counter() - t0 < warm_s:
        call()
        sync()
        k += 1
    warm = time.perf_counter() - t0
    if n is None:
        n = calls_for(warm / k, target_s)
    per = []
    for _ in range(rounds + 1):
        t = time.perf_counter()
        for _ in range(n):
            call()
        sync()
        per.append((time.perf_counter() - t) / n)
    res = dict(ms=stats_ms(per[1:]), n=n, kept_rounds=rounds, warmup_calls=k, warmup_s=round(warm, 3))
    if latency and n == 1:
        res["latency"] = "n = 1: every round is one synchronized call (no separate latency mode)"
    elif latency:
        lat = []
        for _ in range(latency):
            t = time.perf_counter()
            call()
            sync()
            lat.append(time.perf_counter() - t)
        res["latency_ms_median"] = round(1e3 * float(np.median(lat)), 5)
    return res


def calls_for(seconds_per_call, target_s=0.25, lo=1, hi=200):
    """N back-to-back calls so one round lasts about target_s (QUICK: a third of it)."""
    t = target_s / (3 if QUICK else 1)
    return int(min(hi, max(lo, round(t / max(seconds_per_call, 1e-6)))))


def packages(names):
    import importlib.metadata as md
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


SLURM_COUNTS = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE")


def cpu_threads():
    """(threads, rule) for the CPU engine: PAPER_BENCH_MJC_THREADS when set; else the allocated cores =
    len(os.sched_getaffinity(0)), capped by the scheduler's CPU count (SLURM_CPUS_PER_TASK, else SLURM_CPUS_ON_NODE)
    when one is set (an allocation without CPU binding shows every core of the node in the affinity mask). Never
    os.cpu_count(), which oversubscribes a shared node."""
    env = os.environ.get("PAPER_BENCH_MJC_THREADS")
    if env:
        return int(env), "PAPER_BENCH_MJC_THREADS"
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for k in SLURM_COUNTS[:2]:
        v = os.environ.get(k, "")
        if v.isdigit() and int(v) > 0:
            if int(v) < aff:
                return int(v), f"min(len(sched_getaffinity) = {aff}, {k} = {v})"
            return aff, f"len(sched_getaffinity) ({k} = {v})"
    return aff, "len(os.sched_getaffinity(0))"


def cpu_info():
    model = None
    try:
        model = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        model = platform.processor() or None
    aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return dict(model=model, logical_cpus=os.cpu_count(), allocated=len(aff) if aff else None,
                affinity=_ranges(aff) if aff else None,
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                scheduler_cpu_counts={k: os.environ[k] for k in SLURM_COUNTS if k in os.environ} or None,
                cpu_engine_threads=dict(zip(("threads", "rule"), cpu_threads())))


def _ranges(xs):
    out, start, prev = [], None, None
    for x in xs:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = x
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


def gpu_info():
    """name, compute capability, driver, memory of the first visible GPU (nvidia-smi; CUDA_VISIBLE_DEVICES[0])."""
    q = ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"]
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    for cmd in ([*q, "-i", idx] if idx else None, q):
        if cmd is None:
            continue
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                               timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            p = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
            return dict(name=p[0], compute_cap=p[1], driver=p[2], memory=p[3])
    return None


def basic_env(pkgs=()):
    """What every worker records: python, packages, CPU, GPU, the kit's environment variables."""
    env = {k: v for k, v in os.environ.items() if k.startswith(("PAPER_BENCH_", "XLA_", "JAX_", "MJW_", "DK_"))
           and "/" not in v}
    env.update({k: rel(v) for k, v in os.environ.items() if k.startswith(("PAPER_BENCH_", "DK_")) and "/" in v})
    return dict(python=platform.python_version(), executable=rel(sys.executable), packages=packages(pkgs),
                cpu=cpu_info(), gpu=gpu_info(), env=env, quick=QUICK)


class Result:
    """A worker's result JSON, rewritten after every cell (a killed worker keeps what it finished)."""

    def __init__(self, path, **head):
        self.path = Path(path)
        self.data = dict(head, cells={}, errors={})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save()

    def cell(self, key, value):
        self.data["cells"][key] = value
        self.save()
        print(json.dumps({key: _brief(value)}), flush=True)

    def error(self, key, exc):
        self.data["errors"][key] = dict(type=type(exc).__name__, message=str(exc)[:2000],
                                        traceback=traceback.format_exc()[-3000:])
        self.save()
        print(f"ERROR {key}: {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr, flush=True)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(scrub_text(json.dumps(self.data, indent=1, default=_default)))
        os.replace(tmp, self.path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _brief(v):
    if isinstance(v, dict):
        if "ms" in v and isinstance(v["ms"], dict):
            return dict(ms_median=v["ms"].get("median"), p10=v["ms"].get("p10"), p90=v["ms"].get("p90"))
        return {k: _brief(x) for k, x in list(v.items())[:12]}
    return v
