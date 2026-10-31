"""The benchmark driver: plans the units of a job, runs each in a fresh worker process in its engine's interpreter,
samples the GPU's clocks alongside, and rewrites the job's result JSON (schema dk-bench/1) after every unit so a run
resumes where it stopped.

  python -m benchmarks.bench run --machine rtx_pro_6000 --part gpu-fast --out runs/bench/rtx_pro_6000/
  python -m benchmarks.bench run --engines null --quick --out runs/bench/test_null/
  python -m benchmarks.bench env-check

Units and processes. A unit is (engine, variant, problem, corpus, nworld, iterations, process) with its ops, timing
modes and state subsets; each runs in its own fresh worker process (worker.py), so there is at least one process per
(engine, model, nworld). Headline units (mjlab, 4096 worlds, 10 iterations, corpora
HEADLINE_CORPORA) run in processes_headline (3) processes, the others in 1. nworld <= 256 runs SUBSETS_SMALL disjoint
state subsets (<= 1024: 2, else 1) one after the other in the unit's process (a fresh adapter each), so small batches
are averaged over several subsets. Units are ordered round-robin over the engines (A B C A B C ...), process index
outermost. Every job gets a dk anchor unit (mjlab, 4096 worlds, 10 iterations, fwd + vjp; speed-ups only against
the same job's anchor) when the dk adapter exists (--no-anchor: none). The parts (PARTS) name the engines of the
public comparison; an adapter outside it joins a part with --add-engine E=REF (engine E wherever the part runs REF).

Timing protocol (worker.py has the details): warm-up >= warmup_calls (5) calls and >= warmup_s (1 s); rounds (5)
rounds of N back-to-back calls + one sync (throughput) and of N individually synchronised calls (latency); round 0
dropped; N from the warm-up (>= round_min_s per round, >= min_reps_gpu (20) / min_reps_cpu (5) calls, <= round_max_s);
compile / capture / first call timed separately; per-unit timeout timeout_s (skipped:time, the finished ops kept).
--quick: nworld 256 and 4096, one corpus, the first iteration count of each block, P = 1, one subset, 3 rounds,
warm-up >= 0.5 s, op jac only at 256 worlds (QUICK_JAC_NWORLD), timeout 600 s per unit.

GPU and CPU state. A separate `nvidia-smi --query-gpu=... -lms <clock_sample_ms>` process samples SM / memory clock,
power, temperature, utilisation and the clock-event (throttle) reasons of the first visible GPU while a unit runs; each
timed cell gets the summary of the samples inside its kept rounds. Only hardware slowdown, hardware thermal slowdown
and power brake are flags (status flagged:<reasons>); nothing is re-run automatically. The job records the CPU
(model, sockets, cores, threads, affinity, governor, NUMA) and every unit the load average before and after it and
the number of other compute processes on the GPU when it started.

Environments. benchmarks/envs/envs.toml maps each engine to a Python environment ([engine.<id>] env = ...) and each
environment to its interpreter, extra PYTHONPATH entries and variables ([env.<name>]); interpreter candidates may use
{driver} (this interpreter), {prefix} ($DK_BENCH_ENV_PREFIX, default runs/bench/envs), {repo} and {env:VAR}; the first
existing one wins. Any other benchmarks/envs/*.toml (in name order; the definitions that come with an additional
adapter) and then a local, untracked override file deep-merge over it: $DK_BENCH_ENVS, default
runs/bench/envs.local.toml. Workers run with cwd = the repository, PYTHONPATH = the repository + the environment's
entries (the driver's PYTHONPATH is not inherited), PYTHONNOUSERSITE=1, XLA_PYTHON_CLIENT_PREALLOCATE=false unless set.

Result files. <out>/<job>.json, job = --job, else the part names joined by '+', else 'quick' / 'custom'. An existing
file is resumed (its protocol and options must match; --fresh starts over); a file tracked by git is never written.
Units whose cells all have a final status are skipped (--retry-errors re-runs error / skipped:time cells). Raw dumps
go to <out>/raw/<job>/, logs of failed units to <out>/logs/. Exit status 1 when a cell ended in error or skipped:time.

Environment variables (all optional): DK_BENCH_ENV_PREFIX (default runs/bench/envs: where benchmarks/envs/setup.sh
puts the environments), DK_BENCH_ENVS (default runs/bench/envs.local.toml: the local override of envs.toml),
DK_BENCH_PY_<ENV> (the interpreter of environment <ENV>, upper case, '-' as '_', e.g. DK_BENCH_PY_MJW_NATIVE),
DK_BENCH_DATA (default benchmarks/data), DK_BENCH_FIXTURE_DIR (default unset) and DK_BENCH_MODELS (default
runs/bench/models): see problem.py. None of them changes results; they only locate files.
"""

import copy
import ctypes
import datetime
import fcntl
import itertools
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.bench import REPO, adapters, prov, schema
from benchmarks.bench import problem as P

try:
    import tomllib
except ImportError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

ENVS_TOML = REPO / "benchmarks" / "envs" / "envs.toml"
ENVS_LOCAL_DEFAULT = "runs/bench/envs.local.toml"
PREFIX_DEFAULT = "runs/bench/envs"

PROTOCOL = dict(warmup_calls=5, warmup_s=1.0, rounds=5, drop_rounds=1, modes=["throughput", "latency"],
                round_min_s=0.2, round_max_s=20.0, min_reps_gpu=20, min_reps_cpu=5, max_reps=100000,
                processes_headline=3, timer="perf_counter+sync", clock_sample_ms=100, timeout_s=1800.0,
                seed=P.SELECT_SEED)
QUICK = dict(rounds=3, warmup_s=0.5, timeout_s=600.0)
PROTOCOL_IDENTITY = ("warmup_calls", "warmup_s", "rounds", "drop_rounds", "modes", "round_min_s", "round_max_s",
                     "min_reps_gpu", "min_reps_cpu", "processes_headline", "seed")  # must match on resume

HEADLINE_CORPORA = ("mf_s0", "mf_s1")
MF_CORPORA = ("mf_s0", "mf_s1", "mf_s2", "mf_s3")
DEFAULT_CORPUS = "mf_s0"  # of a custom job (--engines without --part or --corpora)
TRACK_B_CORPORA = ("menagerie_s0",)
SWEEP = (64, 256, 1024, 4096, 16384)
QUICK_NWORLD = (256, 4096)
QUICK_JAC_NWORLD = 256  # --quick times op jac only up to this batch (4096-world Jacobians take minutes on CPU / MJX)
SUBSETS_SMALL = 4  # disjoint subsets at nworld <= 256 (2 at <= 1024, else 1)
GPU_ENGINES = ("dk", "mjw", "mjw_pr1535", "mjw_pr1535_fix")  # the GPU engines of the comparison (--add-engine: more)
TRAIN_OPS = ("fwd", "fwd_train", "bwd", "vjp")
ANCHOR = dict(engine="dk", model="mjlab", nworld=4096, iterations=10, ops=("fwd", "vjp"))
THROTTLE_FIELDS = {"hw_slowdown": "hw_slowdown", "hw_thermal_slowdown": "hw_thermal",
                   "hw_power_brake_slowdown": "power_brake", "sw_thermal_slowdown": "sw_thermal",
                   "sw_power_cap": "sw_power_cap"}


@dataclass(frozen=True)
class Block:
    """A rectangle of units: engines x variants x models x corpora x nworld x iterations (None = the model's own)."""
    engines: tuple
    corpora: tuple
    nworld: tuple
    iterations: tuple
    ops: tuple
    models: tuple = ("mjlab",)
    variants: tuple = ("default",)
    dump: tuple = ()


# The job plan: parts of at most about an hour each (on a batch system one job per part; a local run does them all).
# Variant names are those the adapters define (adapters/<engine>.py `variants`).
PARTS = {
    "gpu-fast": (
        Block(GPU_ENGINES, HEADLINE_CORPORA, SWEEP, (10,), TRAIN_OPS),
        Block(GPU_ENGINES, ("mf_s0",), (4096,), (50,), TRAIN_OPS),
        Block(GPU_ENGINES, ("mf_s2", "mf_s3"), (4096,), (10,), TRAIN_OPS),
        Block(("dk_jac",), ("mf_s0",), (64, 256, 1024, 4096), (10,), ("jac",)),
    ),
    "mjx": (
        Block(("mjx",), ("mf_s0",), SWEEP, (10,), ("fwd", "fwd_train", "bwd", "vjp", "jac")),
        Block(("mjx",), ("mf_s1",), (4096,), (10,), ("fwd",)),
        Block(("mjx",), ("mf_s0",), (4096,), (50,), ("fwd",)),
    ),
    "cpu": (
        Block(("mjc",), ("mf_s0",), (64, 1024, 4096), (10,), ("fwd", "jac")),
        Block(("mjc",), ("mf_s0",), (4096,), (50,), ("fwd",)),
    ),
    "accuracy": (
        Block(GPU_ENGINES + ("mjx", "mjc"), MF_CORPORA, (4096,), (10, 50), (), dump=("outputs", "gradient")),
    ),
    "trackB": (
        Block(("mjx", "mjw", "mjc"), TRACK_B_CORPORA, (4096,), (None, 10), ("fwd",), models=("menagerie_g1",)),
    ),
}


@dataclass
class Unit:
    engine: str
    variant: str
    model: str
    corpus: str
    subsets: tuple
    nworld: int
    iterations: object  # int or None (native)
    process: int
    ops: tuple
    modes: tuple
    dump: tuple = ()
    opts: dict = field(default_factory=dict)
    headline: bool = False
    anchor: bool = False

    @property
    def it(self):
        return "native" if self.iterations is None else self.iterations

    def key(self):
        e = self.engine if self.variant == "default" else f"{self.engine}~{self.variant}"
        return f"{e}/{self.model}/{self.corpus}/{self.nworld}/it{self.it}/p{self.process}"

    def slot(self):
        return (self.model, self.corpus, self.nworld, str(self.it), self.variant, self.process)

    def planned(self):
        """[(cell id, subset, op, mode)] of every cell this unit writes (op None: the dump cell)."""
        out = []
        for s in self.subsets:
            out += [(schema.cell_id(self.engine, self.variant, self.model, self.corpus, s, self.nworld, self.it, op,
                                    mode, self.process), s, op, mode) for op in self.ops for mode in self.modes]
            if self.dump:
                out.append((schema.cell_id(self.engine, self.variant, self.model, self.corpus, s, self.nworld,
                                           self.it, None, None, self.process), s, None, None))
        return out

    def cell_ids(self):
        return [p[0] for p in self.planned()]


class PlanError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------- environments

def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class Env:
    name: str
    python: str  # absolute path of the interpreter
    pythonpath: list
    vars: dict

    def describe(self):
        return dict(python=prov.rel(self.python), pythonpath=[prov.rel(p) for p in self.pythonpath],
                    vars={k: prov.rel(v) if "/" in str(v) else v for k, v in self.vars.items()})


class Envs:
    """envs.toml plus the local override (module docstring)."""

    def __init__(self, path=ENVS_TOML):
        if tomllib is None:
            raise RuntimeError("reading benchmarks/envs/envs.toml needs Python >= 3.11 (tomllib) or tomli")
        self.files = [path]
        self.conf = tomllib.loads(Path(path).read_text())
        for extra in sorted(Path(path).parent.glob("*.toml")):  # definitions that come with additional adapters
            if extra.resolve() != Path(path).resolve():
                self.conf = _merge(self.conf, tomllib.loads(extra.read_text()))
                self.files.append(extra)
        local = os.environ.get("DK_BENCH_ENVS") or str(REPO / ENVS_LOCAL_DEFAULT)
        if Path(local).exists():
            self.conf = _merge(self.conf, tomllib.loads(Path(local).read_text()))
            self.files.append(local)
        elif os.environ.get("DK_BENCH_ENVS"):
            raise FileNotFoundError(f"DK_BENCH_ENVS={local} does not exist")
        self.prefix = Path(os.environ.get("DK_BENCH_ENV_PREFIX") or PREFIX_DEFAULT)
        if not self.prefix.is_absolute():
            self.prefix = REPO / self.prefix

    def _subst(self, s):
        """s with {driver} {prefix} {repo} {env:VAR} filled in; None when it names an unset variable."""
        out = str(s)
        while "{env:" in out:
            a = out.index("{env:")
            b = out.index("}", a)
            v = os.environ.get(out[a + 5:b])
            if not v:
                return None
            out = out[:a] + v + out[b + 1:]
        out = out.replace("{driver}", sys.executable).replace("{prefix}", str(self.prefix)).replace("{repo}", str(REPO))
        p = Path(out).expanduser()
        return str(p if p.is_absolute() or out == sys.executable else REPO / p)

    def engine_env(self, engine):
        e = (self.conf.get("engine") or {}).get(engine)
        return (e or {}).get("env"), e or {}

    def resolve(self, engine):
        """The Env of an engine; FileNotFoundError when no interpreter candidate exists."""
        name, eng = self.engine_env(engine)
        if name is None:
            name = "dk"
            print(f"[bench] engine {engine} has no entry in envs.toml: using environment 'dk'", file=sys.stderr)
        env = (self.conf.get("env") or {}).get(name)
        if env is None:
            raise KeyError(f"envs.toml: engine {engine} names environment {name!r}, which is not defined")
        cands = env.get("python", [])
        cands = [cands] if isinstance(cands, str) else list(cands)
        override = os.environ.get("DK_BENCH_PY_" + name.upper().replace("-", "_"))
        if override:
            cands.insert(0, override)
        tried = []
        for c in cands:
            p = self._subst(c)
            if p is None:
                continue
            tried.append(prov.rel(p))
            if os.path.isfile(p) and os.access(p, os.X_OK):
                break
        else:
            raise FileNotFoundError(f"environment {name!r} of engine {engine}: no interpreter among {tried or cands} "
                                    f"(create it with benchmarks/envs/setup.sh, or set DK_BENCH_PY_"
                                    f"{name.upper().replace('-', '_')})")
        pp = []
        for x in list(env.get("pythonpath", [])) + list(eng.get("pythonpath", [])):
            s = self._subst(x)
            if s is not None:
                pp.append(s)
        vars_ = {**env.get("vars", {}), **eng.get("vars", {})}
        vars_ = {k: str(v) for k, v in vars_.items()}
        return Env(name=name, python=p, pythonpath=pp, vars=vars_)

    def engines(self):
        return sorted((self.conf.get("engine") or {}).keys())

    def child_env(self, env):
        e = dict(os.environ)
        e.pop("PYTHONPATH", None)
        e["PYTHONPATH"] = os.pathsep.join([str(REPO)] + env.pythonpath)
        e["PYTHONNOUSERSITE"] = "1"
        e.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        e.update(env.vars)
        return e


# ------------------------------------------------------------------------------------------------- GPU sampler

def _child_setup():
    """In a child process: die with the driver (Linux PR_SET_PDEATHSIG), so a killed driver leaves no worker or
    nvidia-smi sampler behind."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except (OSError, AttributeError):
        pass


class Sampler:
    """nvidia-smi --query-gpu ... -lms <ms> of the first visible GPU, as a separate process writing a CSV file."""

    BASE = ("timestamp", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu", "memory.used")
    _prefix = None

    def __init__(self, path, interval_ms):
        self.path, self.ms, self.proc, self.fh = Path(path), interval_ms, None, None

    @classmethod
    def reason_prefix(cls):
        """clocks_event_reasons (drivers >= 535) or the older clocks_throttle_reasons; None without nvidia-smi."""
        if cls._prefix is None:
            cls._prefix = ""
            for pre in ("clocks_event_reasons", "clocks_throttle_reasons"):
                q = ["nvidia-smi", f"--query-gpu={pre}.active", "--format=csv,noheader"]
                g = prov.visible_gpu()
                if prov._run([*q, "-i", g] if g else q) is not None:
                    cls._prefix = pre
                    break
        return cls._prefix or None

    def fields(self):
        pre = self.reason_prefix()
        return list(self.BASE) + ([f"{pre}.active"] + [f"{pre}.{k}" for k in THROTTLE_FIELDS] if pre else [])

    def start(self):
        if not self.ms or shutil.which("nvidia-smi") is None or self.reason_prefix() is None:
            return self
        g = prov.visible_gpu()
        cmd = ["nvidia-smi", f"--query-gpu={','.join(self.fields())}", "--format=csv,noheader,nounits",
               "-lms", str(int(self.ms))] + (["-i", g] if g else ["-i", "0"])
        self.fh = open(self.path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.DEVNULL, preexec_fn=_child_setup)
        return self

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def samples(self):
        if not self.path.exists():
            return []
        names = self.fields()
        out = []
        for line in self.path.read_text().splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) != len(names):
                continue
            try:
                t = datetime.datetime.strptime(vals[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            s = dict(t=t, sm=prov._num(vals[1]), mem=prov._num(vals[2]), power=prov._num(vals[3]),
                     temp=prov._num(vals[4]), util=prov._num(vals[5]), mem_used=prov._num(vals[6]), reasons=[])
            for name, v in zip(names[8:], vals[8:]):
                if v.lower() == "active":
                    s["reasons"].append(THROTTLE_FIELDS[name.split(".", 1)[1]])
            out.append(s)
        return out


def clock_summary(samples, window, max_sm=None):
    """Summary of the samples inside window [start, end] (unix s): medians / extremes and the reasons seen;
    flags = the hardware ones (schema.THROTTLE_FLAGS). None without samples there."""
    if not samples or not window or None in window:
        return None
    w0, w1 = window
    sel = [s for s in samples if w0 - 0.05 <= s["t"] <= w1 + 0.05]
    if not sel:  # a window shorter than the sampling interval: the nearest sample
        sel = [min(samples, key=lambda s: abs(s["t"] - 0.5 * (w0 + w1)))]
        if abs(sel[0]["t"] - 0.5 * (w0 + w1)) > 1.0:
            return None

    def vals(k):
        return [s[k] for s in sel if s[k] is not None]

    def med(k):
        v = vals(k)
        return statistics.median(v) if v else None

    reasons = sorted({r for s in sel for r in s["reasons"]})
    sm = vals("sm")
    return dict(samples=len(sel), sm_mhz_median=med("sm"), sm_mhz_min=min(sm) if sm else None,
                sm_mhz_max=max(sm) if sm else None, sm_of_max=(statistics.median(sm) / max_sm) if sm and max_sm
                else None, mem_mhz_median=med("mem"), power_w_median=med("power"),
                power_w_max=max(vals("power")) if vals("power") else None,
                temp_c_max=max(vals("temp")) if vals("temp") else None, util_median=med("util"), throttle=reasons,
                flags=[f for f in schema.THROTTLE_FLAGS if f in reasons])


# ------------------------------------------------------------------------------------------------- planning

def default_corpus():
    """The corpus of a custom job: DEFAULT_CORPUS (benchmarks/data/mf_s0.npz)."""
    return DEFAULT_CORPUS


def _subsets(n_states, nworld, cap):
    if cap is not None:
        want = cap
    else:
        want = SUBSETS_SMALL if nworld <= 256 else (2 if nworld <= 1024 else 1)
    return max(1, min(want, n_states // nworld if nworld <= n_states else 1))


def _parse_iterations(vals):
    out = []
    for v in vals:
        out.append(None if str(v) == "native" else int(v))
    return tuple(out)


def make_blocks(a):
    """The blocks of a run's arguments: the parts' blocks with the explicit options applied (--engines filters them,
    --models / --corpora / --nworld / --iterations / --ops replace), or one block from the options alone."""
    if a.part:
        blocks = []
        for part in a.part:
            if part not in PARTS:
                raise PlanError(f"unknown part {part!r} (one of {', '.join(PARTS)})")
            blocks += list(PARTS[part])
    elif a.engines:
        blocks = [Block(engines=tuple(a.engines), corpora=(None,), nworld=(4096,), iterations=(10,),
                        ops=tuple(schema.OPS))]
    else:
        raise PlanError("give --part or --engines")
    added = list(getattr(a, "add_engine", None) or [])
    if added and not a.part:
        raise PlanError("--add-engine needs --part (with --engines alone, list the engine there)")
    out = []
    for b in blocks:
        kw = dict(b.__dict__)
        for e, ref in added:  # an adapter outside the default plan joins the blocks that run its reference engine
            if ref in kw["engines"] and e not in kw["engines"]:
                kw["engines"] = kw["engines"] + (e,)
        if a.engines and a.part:
            kw["engines"] = tuple(e for e in kw["engines"] if e in a.engines)
            if not kw["engines"]:
                continue
        if a.models:
            kw["models"] = tuple(a.models)
        if a.corpora:
            kw["corpora"] = tuple(a.corpora)
        if a.nworld:
            kw["nworld"] = tuple(int(x) for x in a.nworld)
        elif a.quick:
            kw["nworld"] = QUICK_NWORLD
        if a.iterations:
            kw["iterations"] = _parse_iterations(a.iterations)
        elif a.quick:
            kw["iterations"] = kw["iterations"][:1]
        if a.ops is not None:
            kw["ops"] = tuple(a.ops)
        if a.quick and not a.corpora:
            kw["corpora"] = kw["corpora"][:1]
        kw["corpora"] = tuple(default_corpus() if c is None else c for c in kw["corpora"])
        out.append(Block(**kw))
    return out


def make_plan(a, prot, registry):
    """[Unit] in run order (round-robin over engines), the anchor's cell ids (or None) and warnings."""
    warnings = []
    slots = {}  # slot -> {engine: Unit}, insertion-ordered
    sizes = {}
    problems = {}
    for b in make_blocks(a):
        for model in b.models:
            if model not in problems:
                problems[model] = P.get(model, (a.model_path or {}).get(model))
            prob = problems[model]
            for corpus, nworld, it in itertools.product(b.corpora, b.nworld, b.iterations):
                if it is None and prob.explicit_iterations:
                    raise PlanError(f"{model}: iterations must be explicit (10 or 50), not 'native'")
                if "/" in corpus or corpus.endswith(".npz"):
                    raise PlanError(f"corpus {corpus!r}: give a corpus name (put the file into $DK_BENCH_DATA)")
                if corpus not in sizes:
                    if not P.corpus_exists(corpus):
                        raise PlanError(f"corpus {corpus!r} not found ($DK_BENCH_DATA/<name>.npz, default "
                                        f"benchmarks/data; --corpora to choose)")
                    sizes[corpus] = P.corpus_size(corpus)
                headline = (model == "mjlab" and nworld == 4096 and it == 10 and corpus in HEADLINE_CORPORA)
                nproc = a.processes if a.processes else (prot["processes_headline"] if headline else 1)
                if a.quick and not a.processes:
                    nproc = 1
                nsub = _subsets(sizes[corpus], nworld, 1 if a.quick and a.subsets is None else a.subsets)
                ops = tuple(b.ops)
                if a.quick and a.ops is None and nworld > QUICK_JAC_NWORLD:
                    ops = tuple(o for o in ops if o != "jac")
                dump = tuple(b.dump) + tuple(a.dump or ())
                if b.ops and not ops and not dump:
                    continue
                for p, e in itertools.product(range(nproc), b.engines):
                    for variant in (a.variant or {}).get(e) or b.variants:
                        u = Unit(engine=e, variant=variant, model=model, corpus=corpus, subsets=tuple(range(nsub)),
                                 nworld=nworld, iterations=it, process=p, ops=ops, modes=tuple(prot["modes"]),
                                 dump=dump, opts=dict((a.opt or {}).get(e, {})), headline=headline)
                        sl = slots.setdefault((p,) + u.slot(), {})
                        if e in sl:
                            sl[e].ops = tuple(dict.fromkeys(sl[e].ops + u.ops))
                            sl[e].dump = tuple(dict.fromkeys(sl[e].dump + u.dump))
                        else:
                            sl[e] = u
    units = [u for sl in slots.values() for u in sl.values()]
    units.sort(key=lambda u: u.process)  # stable: every engine's p0 units first, the repeats spread over the job
    missing = sorted({u.engine for u in units} - set(registry))
    if missing:
        if a.skip_missing:
            warnings.append(f"no adapter for {', '.join(missing)}: skipped (--skip-missing)")
            units = [u for u in units if u.engine not in missing]
        else:
            raise PlanError(f"no adapter for engine(s) {', '.join(missing)} (benchmarks/bench/adapters/; "
                            f"--skip-missing to run the others)")
    anchor = None
    if not a.no_anchor:
        if ANCHOR["engine"] not in registry:
            warnings.append("no dk anchor cell: the dk adapter does not exist yet")
        else:
            ac = HEADLINE_CORPORA[0] if P.corpus_exists(HEADLINE_CORPORA[0]) else default_corpus()
            au = next((u for u in units if u.engine == "dk" and u.variant == "default" and u.model == "mjlab"
                       and u.corpus == ac and u.nworld == ANCHOR["nworld"] and u.iterations == ANCHOR["iterations"]
                       and 0 in u.subsets and u.process == 0), None)
            if au is None:
                if "mjlab" not in problems:
                    problems["mjlab"] = P.get("mjlab", (a.model_path or {}).get("mjlab"))
                if ac not in sizes:
                    sizes[ac] = P.corpus_size(ac)
                au = Unit(engine="dk", variant="default", model="mjlab", corpus=ac, subsets=(0,),
                          nworld=ANCHOR["nworld"], iterations=ANCHOR["iterations"], process=0, ops=(),
                          modes=tuple(prot["modes"]), opts=dict((a.opt or {}).get("dk", {})),
                          headline=ac in HEADLINE_CORPORA)
                units.insert(0, au)
            au.ops = tuple(dict.fromkeys(au.ops + ANCHOR["ops"]))
            au.anchor = True
            anchor = [schema.cell_id("dk", "default", "mjlab", ac, 0, ANCHOR["nworld"], ANCHOR["iterations"], op,
                                     "throughput", 0) for op in ANCHOR["ops"]]
    for m, prob in problems.items():
        if not prob.path().exists():
            raise PlanError(f"model {m}: {prov.rel(prob.path())} not found (fetch-models, or --model-path {m}=PATH)")
    return units, anchor, warnings, problems


# ------------------------------------------------------------------------------------------------- running

class Driver:
    def __init__(self, a, prot):
        self.a, self.prot = a, prot
        self.envs = Envs()
        self.out = Path(a.out)
        if not self.out.is_absolute():
            self.out = Path.cwd() / self.out
        self.tmp = None

    def run_unit(self, u, doc):
        """Run one unit in a fresh worker; returns its cells (every planned id present)."""
        planned = u.planned()
        key = u.key()
        info = dict(key=key, engine=u.engine, env=None, started=schema.now())
        doc.setdefault("units", {})[key] = info
        try:
            env = self.envs.resolve(u.engine)
        except Exception as e:  # noqa: BLE001
            msg = prov.scrub(f"{type(e).__name__}: {e}")
            info.update(status="error", message=msg)
            return [self._placeholder(u, pl, schema.status("error", "env"), msg) for pl in planned]
        info["env"] = env.name
        doc["environments"].setdefault(env.name, {}).update(env.describe())
        prob = P.get(u.model, (self.a.model_path or {}).get(u.model))
        spec = dict(unit=dict(key=key, engine=u.engine, variant=u.variant, model=prob.to_dict(), corpus=u.corpus,
                              subsets=list(u.subsets), nworld=u.nworld, iterations=u.iterations, process=u.process,
                              ops=list(u.ops), modes=list(u.modes), opts=u.opts, dump=list(u.dump),
                              headline=u.headline),
                    protocol=self.prot, deadline_unix=time.time() + self.prot["timeout_s"] - 5.0,
                    raw_dir=str(self.out / "raw" / doc["job"]["name"]) if u.dump else None, cmd=doc["job"]["cmd"],
                    env=env.name)  # per job: two jobs in one --out directory never share a dump file
        d = Path(tempfile.mkdtemp(prefix="unit_", dir=self.tmp))
        spec_p, out_p, log_p = d / "spec.json", d / "result.json", d / "worker.log"
        spec_p.write_text(json.dumps(spec))
        procs = prov.gpu_processes()
        if procs:
            print(f"[bench] warning: {len(procs)} other compute process(es) on the GPU before {key} (not exclusive; "
                  f"recorded as gpu_procs_before)", flush=True)
        load0 = prov.cpu_load()
        sampler = Sampler(d / "clocks.csv", None if self.a.no_sampler else self.prot["clock_sample_ms"]).start()
        cmd = [env.python, "-m", "benchmarks.bench.worker", "--spec", str(spec_p), "--out", str(out_p)]
        print(f"[bench] {key}: {', '.join(u.ops) or '-'}{' + dump' if u.dump else ''} ({env.name})", flush=True)
        t0 = time.time()
        timed_out, rc = False, None
        with open(log_p, "w") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=self.envs.child_env(env), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, preexec_fn=_child_setup)
            try:
                rc = proc.wait(timeout=self.prot["timeout_s"])
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc)
            except BaseException:
                self._kill(proc)
                sampler.stop()
                raise
        sampler.stop()
        load1 = prov.cpu_load()
        samples = sampler.samples()
        res = {}
        if out_p.exists():
            try:
                res = json.loads(out_p.read_text())
            except ValueError:
                res = {}
        tail = prov.scrub("\n".join(log_p.read_text(errors="replace").splitlines()[-30:])) if log_p.exists() else ""
        cells = {c["id"]: c for c in res.get("cells", [])}
        out = []
        for pl in planned:
            c = cells.get(pl[0])
            if c is None:
                if timed_out:
                    c = self._placeholder(u, pl, schema.status("skipped", "time"),
                                          f"the unit exceeded its timeout of {self.prot['timeout_s']:.0f} s before "
                                          f"this cell")
                else:
                    c = self._placeholder(u, pl, schema.status("error", "worker"),
                                          f"the worker exited with status {rc} without this cell; log tail:\n{tail}")
            c["load"] = dict(before=load0, after=load1)
            c["gpu_procs_before"] = None if procs is None else len(procs)
            if c.get("kind") == "timing" and schema.is_timed(c["status"]):
                gmax = (doc["machine"].get("gpu") or {}).get("max_sm_clock_mhz")
                c["clocks"] = clock_summary(samples, c.get("window_unix"), gmax)
                if c["clocks"] and c["clocks"]["flags"] and c["status"] == "ok":
                    c["status"] = schema.status("flagged", "+".join(c["clocks"]["flags"]))
            c.setdefault("elapsed_s", round(time.time() - t0, 3))
            out.append(c)
        # the job-level records of this unit
        info.update(status=res.get("status", "timeout" if timed_out else "error"), rc=rc, timed_out=timed_out,
                    elapsed_s=round(time.time() - t0, 3), message=res.get("message"), setup=res.get("setup"),
                    states=res.get("states"), pid_python=res.get("python"),
                    gpu_procs_before=None if procs is None else len(procs), clock_samples=len(samples))
        if res.get("software"):
            doc["environments"][env.name].update(res["software"])
        if res.get("adapter"):
            eid = u.engine if u.variant == "default" else f"{u.engine}~{u.variant}"
            doc["engines"].setdefault(eid, dict(adapter=res["adapter"], env=env.name, semantics=res.get("semantics"),
                                                provenance=res.get("provenance"), first_unit=key))
            info["provenance"] = res.get("provenance")
        if res.get("problem"):
            doc["models"].setdefault(u.model, res["problem"])
        if res.get("corpus"):
            doc["corpora"].setdefault(u.corpus, res["corpus"])
        failed = any(not schema.is_final(c["status"]) for c in out)
        if failed or self.a.keep_logs:
            (self.out / "logs").mkdir(parents=True, exist_ok=True)
            dst = self.out / "logs" / f"{doc['job']['name']}__{key.replace('/', '__')}.log"
            shutil.copyfile(log_p, dst)
            info["log"] = prov.rel(dst)
        if self.a.keep_clock_samples and samples:
            (self.out / "raw" / "clocks").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "clocks.csv", self.out / "raw" / "clocks" /
                            f"{doc['job']['name']}__{key.replace('/', '__')}.csv")
        shutil.rmtree(d, ignore_errors=True)
        return out

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()

    def _placeholder(self, u, planned, status, message):
        cid, subset, op, mode = planned
        return dict(id=cid, kind="dump" if op is None else "timing", engine=u.engine, variant=u.variant, label=None,
                    native=None, device=None, env=None, model=u.model, track=None, corpus=u.corpus, subset=subset,
                    nworld=u.nworld,
                    iterations=u.it, substeps=None, op=op, mode=mode, process=u.process, status=status,
                    message=message, headline=u.headline, started=schema.now(), cmd="", unit=u.key())


def _job_name(a):
    if a.job:
        return a.job
    if a.part:
        return "+".join(a.part)
    return "quick" if a.quick else "custom"


def _cmd(argv):
    return prov.scrub("python -m benchmarks.bench " + " ".join(argv))


def run(a, argv):
    """`run`: plan, resume, run every unit, write after each; exit status per the module docstring."""
    prot = dict(PROTOCOL)
    if a.quick:
        prot.update(QUICK)
    for k in ("rounds", "warmup_calls", "warmup_s", "timeout_s"):
        if getattr(a, k, None) is not None:
            prot[k] = type(PROTOCOL[k])(getattr(a, k))
    if a.modes:
        prot["modes"] = list(a.modes)
    if a.no_sampler:
        prot["clock_sample_ms"] = None
    prot["quick"] = bool(a.quick)
    tag = a.machine or prov.default_tag()
    if not schema.TAG_RE.match(tag):
        raise PlanError(f"--machine {tag!r}: lower-case letters, digits, '_', '.', '-'")
    import socket
    if tag == socket.gethostname().split(".")[0].lower():
        raise PlanError("--machine must not be the host name (no host names in results): use a tag, e.g. rtx_pro_6000")
    registry = adapters.discover()
    units, anchor, warns, problems = make_plan(a, prot, registry)
    name = _job_name(a)
    if a.dry_run:
        for w in warns:
            print("[bench] warning:", w)
        for u in units:
            print(f"{u.key():60s} ops={','.join(u.ops) or '-'} modes={','.join(u.modes)}"
                  f"{' dump=' + ','.join(u.dump) if u.dump else ''}{' headline' if u.headline else ''}"
                  f"{' anchor' if u.anchor else ''}")
        print(f"[bench] {len(units)} units, {sum(len(u.cell_ids()) for u in units)} cells -> "
              f"{prov.rel(Path(a.out) / (name + '.json'))} (dry run)")
        return 0
    drv = Driver(a, prot)
    path = drv.out / f"{name}.json"
    if prov.is_committed(path):
        raise PlanError(f"{prov.rel(path)} is a committed result: never overwritten (choose another --out)")
    drv.out.mkdir(parents=True, exist_ok=True)
    lock = open(drv.out / f".{name}.json.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise PlanError(f"another driver is writing {prov.rel(path)}") from None
    job = dict(name=name, machine=tag, parts=list(a.part or []), quick=bool(a.quick), cmd=_cmd(argv),
               opts=a.opt or {}, envs_files=[prov.rel(f) for f in drv.envs.files], warnings=warns,
               planned_cells=sum(len(u.cell_ids()) for u in units), complete=False)
    if path.exists() and not a.fresh:
        doc = schema.read(path)
        old = doc["protocol"]
        diff = [k for k in PROTOCOL_IDENTITY if old.get(k) != prot.get(k)]
        if diff or (doc["job"].get("opts") or {}) != (a.opt or {}):
            raise PlanError(f"{prov.rel(path)} was written with another protocol / options ({diff or 'opts'}): "
                            f"--fresh to start over, or another --out")
        print(f"[bench] resuming {prov.rel(path)} ({len(doc['cells'])} cells)", flush=True)
        doc["job"].update({k: v for k, v in job.items() if k not in ("cmd",)})
        doc["job"].setdefault("resumed", []).append(job["cmd"])
        doc["protocol"]["timeout_s"] = prot["timeout_s"]
    else:
        doc = schema.new_job(job=job, machine=prov.machine(tag), software=prov.software(), protocol=prot,
                             provenance=prov.provenance())
    doc["anchor"] = anchor
    for w in warns:
        print("[bench] warning:", w, flush=True)
    if prov.visible_count() != 1:
        w = ("CUDA_VISIBLE_DEVICES is not set to exactly one GPU: the workers use the first visible one; set it so the "
             "GPU is exclusive")
        print("[bench] warning:", w, flush=True)
        doc["job"]["warnings"] = list(dict.fromkeys(doc["job"]["warnings"] + [w]))
    for m, prob in problems.items():
        doc["models"].setdefault(m, prob.describe())
    doc = schema.write(path, doc)
    work = drv.out / ".work"  # per-driver scratch (unit specs, worker results, logs, clock samples); a killed
    for old_dir in work.glob("*"):  # driver's is removed by the next run
        try:
            os.kill(int(old_dir.name), 0)
        except (ValueError, ProcessLookupError):
            shutil.rmtree(old_dir, ignore_errors=True)
        except PermissionError:
            pass
    drv.tmp = str(work / str(os.getpid()))
    os.makedirs(drv.tmp, exist_ok=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    t_job = time.time()
    try:
        for n, u in enumerate(units):
            ids = u.cell_ids()
            have = {c["id"]: c for c in doc["cells"]}
            done = all(i in have and (schema.is_final(have[i]["status"]) or not a.retry_errors) for i in ids)
            if done:
                print(f"[bench] {u.key()}: done, skipped ({n + 1}/{len(units)})", flush=True)
                continue
            cells = drv.run_unit(u, doc)
            doc["cells"] = [c for c in doc["cells"] if c["id"] not in set(ids)] + cells
            for c in cells:
                med = f" {c['median_ms']:.4g} ms" if c.get("median_ms") is not None else ""
                print(f"[bench]   {c['id']}: {c['status']}{med}", flush=True)
            doc = schema.write(path, doc)
        have = {c["id"] for c in doc["cells"]}
        doc["job"]["complete"] = all(i in have for u in units for i in u.cell_ids())
        doc["job"]["elapsed_s"] = round(time.time() - t_job, 1)
        doc = schema.write(path, doc)
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)
        shutil.rmtree(drv.tmp, ignore_errors=True)
        try:
            os.rmdir(drv.out / ".work")
        except OSError:
            pass
        lock.close()
        try:
            os.unlink(drv.out / f".{name}.json.lock")
        except OSError:
            pass
    counts = {}
    for c in doc["cells"]:
        k = schema.parse_status(c["status"])[0] + (":" + c["status"].split(":", 1)[1] if c["status"].startswith(
            "skipped") else "")
        counts[k] = counts.get(k, 0) + 1
    print(f"[bench] {prov.rel(path)}: {len(doc['cells'])} cells, " + ", ".join(f"{k} {v}" for k, v in
                                                                                sorted(counts.items())))
    bad = [c["id"] for c in doc["cells"] if not schema.is_final(c["status"])]
    if bad:
        print(f"[bench] {len(bad)} cells failed or timed out (see `message`; logs in {prov.rel(drv.out / 'logs')})")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------- env-check

def env_check(a):
    """`env-check`: the GPU and CPU state, the sampler, and each engine's environment (interpreter, adapter import,
    the adapter's check_env). Exit status 1 when a checked engine fails (an environment that is not installed only
    fails with --strict or when the engine was named in --engines)."""
    print("== machine")
    g = prov.gpu()
    print("gpu:", json.dumps(g) if g else "none (nvidia-smi not found or no GPU)")
    vc = prov.visible_count()
    if vc != 1:
        print(f"WARNING: CUDA_VISIBLE_DEVICES {'unset' if vc is None else f'lists {vc} GPUs'}: benchmark runs need "
              "exactly one visible GPU (exclusive)")
    procs = prov.gpu_processes()
    if procs:
        print(f"WARNING: {len(procs)} compute process(es) on the GPU already (not exclusive)")
    print("cpu:", json.dumps(prov.cpu()), "| load", prov.cpu_load())
    print("driver python:", prov.software()["python"], "| harness", prov.harness_commit())
    pre = Sampler.reason_prefix()
    print("clock sampler:", f"nvidia-smi {pre}.* every {PROTOCOL['clock_sample_ms']} ms" if pre else "unavailable")
    print("XLA_PYTHON_CLIENT_PREALLOCATE:", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset: workers get "
                                                                                              "false>"))
    envs = Envs()
    print("envs:", ", ".join(prov.rel(f) for f in envs.files), "| prefix", prov.rel(envs.prefix))
    try:
        dk = prov.nablaengine_provenance()
        if dk:
            print(f"nablaengine: arch {dk.get('arch')}, nvcc {dk.get('nvcc')}, "
                  f"commit {dk['nablaengine'].get('commit')}")
    except Exception as e:  # noqa: BLE001
        print("nablaengine provenance:", e)
    registry = adapters.discover()
    engines = a.engines or sorted(set(registry) | set(envs.engines()))
    print("== engines")
    failed = 0
    for e in engines:
        try:
            env = envs.resolve(e)
        except Exception as ex:  # noqa: BLE001
            print(f"{e:16s} NOT INSTALLED: {prov.scrub(ex)}{'' if e in registry else ' (and no adapter yet)'}")
            failed += bool(a.engines or a.strict)
            continue
        r = subprocess.run([env.python, "-m", "benchmarks.bench.worker", "--probe", e], cwd=REPO,
                           env=envs.child_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=600)
        try:
            info = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            info = dict(ok=False, error=prov.scrub(r.stdout[-2000:]))
        pk = info.get("packages", {})
        vers = ", ".join(f"{k} {v['version']}" for k, v in pk.items() if k in ("mujoco", "mujoco-warp", "warp-lang",
                                                                                 "jax", "mujoco-mjx"))
        if info.get("ok"):
            print(f"{e:16s} ok   env {env.name} ({prov.rel(env.python)}, python {info.get('python')}) {vers}"
                  f"{' | ' + json.dumps(info['check']) if info.get('check') else ''}")
        elif e not in registry and "packages" in info:
            print(f"{e:16s} NO ADAPTER (benchmarks/bench/adapters/); env {env.name} works ({prov.rel(env.python)}, "
                  f"python {info.get('python')}) {vers}")
            failed += bool(a.engines or a.strict)
        else:
            failed += 1
            print(f"{e:16s} FAIL env {env.name} ({prov.rel(env.python)}): {info.get('error')}")
    return 1 if failed else 0
