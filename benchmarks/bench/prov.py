"""Provenance for benchmark results: the machine (GPU through nvidia-smi, CPU model / sockets / cores / threads /
affinity / governor / NUMA, OS), the software of an interpreter (Python, package versions with their wheel tags and
install origin, nvcc, the environment variables that change results), the harness commit, and NablaEngine's own
provenance block where nablaengine is importable. Standard library only, so it imports in every engine environment;
every probe is best effort (a missing tool gives None, never an exception).

  from benchmarks.bench import prov
  prov.machine("rtx_pro_6000")    # {tag, gpu, cpu, os, scheduler_cpus}
  prov.software()                 # {python, executable, packages, nvcc, env}
  prov.cpu_load()                 # [1, 5, 15 min load averages]
  prov.rel(path), prov.scrub(text)  # repository-relative paths; free text without home / host / temp paths

Paths are recorded relative to the repository (or to ~), never with host names, user names or job ids.
The GPU is the first entry of CUDA_VISIBLE_DEVICES (index or UUID, as nvidia-smi names it), else nvidia-smi's GPU 0.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("numpy", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang", "jax", "jaxlib", "jax-cuda12-plugin",
            "jax-cuda12-pjrt", "jax-cuda13-plugin", "scipy", "absl-py", "etils", "genesis-world", "quadrants", "torch")
ENV_PREFIXES = ("DK_", "MJW_", "XLA_", "JAX_", "WARP_", "QD_", "GS_")
ENV_NAMES = ("CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NVIDIA_TF32_OVERRIDE", "TF_CPP_MIN_LOG_LEVEL")
GPU_QUERY = ("name", "compute_cap", "memory.total", "driver_version", "persistence_mode", "power.limit",
             "clocks.max.sm", "clocks.max.mem")


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# ------------------------------------------------------------------------------------------------- paths and text

def rel(path):
    """path relative to the repository when inside it, else with the home directory as ~, else only its last part
    (.../name); None stays None (the semantics of nablaengine.provenance.rel, for this checkout)."""
    if path is None:
        return None
    p = Path(path).expanduser().absolute()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    for q in (p, resolved):
        for base, prefix in ((REPO, ""), (Path.home(), "~/")):
            try:
                r = str(q.relative_to(base))
                return prefix + r if r != "." else (prefix.rstrip("/") or ".")
            except ValueError:
                continue
    return f".../{p.name}"


def scrub(text):
    """Free text (an error message, a command line) with the checkout, the home directory, the temp directory and the
    host name replaced (., ~, <tmp>, <host>)."""
    if text is None:
        return None
    s = str(text)
    for base, repl in ((str(REPO) + "/", ""), (str(REPO.resolve()) + "/", ""), (str(REPO), "."),
                       (tempfile.gettempdir().rstrip("/") + "/", "<tmp>/"), (str(Path.home()) + "/", "~/")):
        if len(base) > 1:
            s = s.replace(base, repl)
    s = re.sub(r"/tmp/[^\s'\"]*", "<tmp>", s)
    host = socket.gethostname().split(".")[0]
    if len(host) >= 4:
        s = re.sub(rf"(?<![A-Za-z0-9_-]){re.escape(host)}(?![A-Za-z0-9_-])", "<host>", s)
    return s


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- GPU

def visible_gpu():
    """The first visible GPU as nvidia-smi names it: the first entry of $CUDA_VISIBLE_DEVICES, else None (GPU 0)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v and not v.startswith("-") else None


def visible_count():
    """Number of entries in CUDA_VISIBLE_DEVICES (None when unset: every GPU of the machine is visible)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return None
    return len([x for x in v.split(",") if x.strip() and not x.strip().startswith("-")])


def _smi(fields, extra=()):
    q = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", *extra]
    g = visible_gpu()
    out = (_run([*q, "-i", g]) if g else None) or _run(q)
    if not out:
        return None
    vals = [x.strip() for x in out.splitlines()[0].split(",")]
    return dict(zip(fields, vals)) if len(vals) == len(fields) else None


def _num(s):
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def gpu():
    """{name, compute_cap, memory_mib, driver, cuda_driver, persistence, power_limit_w, max_sm_clock_mhz,
    max_mem_clock_mhz, visible} of the first visible GPU, None without nvidia-smi or a GPU."""
    q = _smi(GPU_QUERY)
    if q is None:
        return None
    head = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([0-9.]+)", head)
    return dict(name=q["name"], compute_cap=q["compute_cap"], memory_mib=_num(q["memory.total"]),
                driver=q["driver_version"], cuda_driver=m.group(1) if m else None,
                persistence=q["persistence_mode"].lower() in ("enabled", "on", "1"),
                power_limit_w=_num(q["power.limit"]), max_sm_clock_mhz=_num(q["clocks.max.sm"]),
                max_mem_clock_mhz=_num(q["clocks.max.mem"]), visible=visible_count())


def gpu_processes():
    """[(pid, used MiB)] of the compute processes on the first visible GPU (nvidia-smi), None without nvidia-smi."""
    q = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    g = visible_gpu()
    out = _run([*q, "-i", g] if g else q)
    if out is None:
        return None
    procs = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            procs.append((int(parts[0]), _num(parts[1])))
    return procs


def process_gpu_mib(pid=None):
    """MiB the process pid (default: this one) holds on the first visible GPU per nvidia-smi (None if not listed;
    in a container with its own PID namespace nvidia-smi may not list it)."""
    pid = os.getpid() if pid is None else pid
    for p, mib in gpu_processes() or []:
        if p == pid:
            return mib
    return None


# ------------------------------------------------------------------------------------------------- CPU

def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu():
    """{model, sockets, cores, threads, affinity, governor, numa_nodes, cgroup_cpu_max}: cores = physical cores of the
    machine, threads = logical CPUs, affinity = CPUs this process may run on (os.sched_getaffinity: a Slurm or taskset
    allocation), governor = the scaling governor(s) of those CPUs ({governor: count} when they differ)."""
    info = _read("/proc/cpuinfo") or ""
    model = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), None)
    if model is None:
        model = platform.processor() or None
    phys = {(blk.get("physical id"), blk.get("core id")) for blk in _cpuinfo_blocks(info)}
    sockets = len({p for p, _ in phys if p is not None}) or None
    cores = len(phys) if phys and None not in next(iter(phys)) else None
    try:
        aff = sorted(os.sched_getaffinity(0))
    except AttributeError:
        aff = list(range(os.cpu_count() or 1))
    govs = {}
    for c in aff:
        g = _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor")
        if g:
            govs[g] = govs.get(g, 0) + 1
    gov = None if not govs else (next(iter(govs)) if len(govs) == 1 else govs)
    numa = len([p for p in Path("/sys/devices/system/node").glob("node[0-9]*")]) or None
    cg = _read("/sys/fs/cgroup/cpu.max")
    return dict(model=model, sockets=sockets, cores=cores, threads=os.cpu_count(), affinity=len(aff), governor=gov,
                numa_nodes=numa, cgroup_cpu_max=cg)


def _cpuinfo_blocks(info):
    for blk in info.split("\n\n"):
        d = {}
        for ln in blk.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                d[k.strip()] = v.strip()
        if d:
            yield d


def affinity():
    """Number of CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def cpu_load():
    """[1, 5, 15 minute load averages], None where unavailable."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def machine(tag):
    """The machine block of a job document."""
    return dict(tag=tag, gpu=gpu(), cpu=cpu(), os=f"{platform.system()} {platform.release()}",
                scheduler_cpus=affinity())


def default_tag():
    """A machine tag from the GPU name (e.g. 'tesla-v100-sxm2-16gb'), 'cpu' without a GPU: never the host name."""
    g = gpu()
    if not g:
        return "cpu"
    return re.sub(r"[^a-z0-9]+", "-", g["name"].lower()).strip("-")[:48] or "gpu"


# ------------------------------------------------------------------------------------------------- software

def package(name):
    """{version, wheel_tags, origin} of an installed distribution, None if absent. origin: the URL it was installed
    from (a release wheel such as warp-lang's +cu12 build), 'editable <path>' / 'local <path>' (relative), or
    'index'."""
    try:
        d = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    tags = None
    try:
        wheel = d.read_text("WHEEL") or ""
        tags = [ln.split(":", 1)[1].strip() for ln in wheel.splitlines() if ln.startswith("Tag:")] or None
    except (OSError, UnicodeDecodeError):
        pass
    origin = "index"
    try:
        du = json.loads(d.read_text("direct_url.json") or "null")
    except (OSError, ValueError):
        du = None
    if isinstance(du, dict) and du.get("url"):
        url = du["url"]
        if url.startswith("file://"):
            kind = "editable" if (du.get("dir_info") or {}).get("editable") else "local"
            origin = f"{kind} {rel(url[len('file://'):])}"
        else:
            vcs = du.get("vcs_info") or {}
            origin = url + (f"@{vcs['commit_id']}" if vcs.get("commit_id") else "")
    return dict(version=d.version, wheel_tags=tags, origin=origin)


def packages(names=PACKAGES):
    """{name: package(name)} for the installed ones."""
    out = {}
    for n in names:
        p = package(n)
        if p is not None:
            out[n] = p
    return out


def nvcc():
    """{path (relative), version} of $NVCC or nvcc on PATH, None without one."""
    exe = os.environ.get("NVCC") or None
    if exe is None:
        from shutil import which
        exe = which("nvcc")
    if not exe:
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)", _run([exe, "--version"]) or "")
    return dict(path=rel(exe), version=m.group(1) if m else None)


def environment():
    """The DK_* / MJW_* / XLA_* / JAX_* / WARP_* / QD_* / GS_* variables and a few thread / CUDA ones that are set
    (paths relative); CUDA_VISIBLE_DEVICES only as the number of visible devices (no device indices in results)."""
    out = {}
    for k, v in sorted(os.environ.items()):
        if k.startswith(ENV_PREFIXES) or k in ENV_NAMES:
            if k == "XLA_FLAGS":
                out[k] = " ".join(f"{f.split('=', 1)[0]}={rel(f.split('=', 1)[1])}" if "=" in f and "/" in f else f
                                  for f in v.split())
            else:
                out[k] = rel(v) if "/" in v else v
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        out["CUDA_VISIBLE_DEVICES"] = f"<{visible_count()} device(s)>"
    return out


def software(names=PACKAGES):
    """The software block of an interpreter: {python, implementation, executable (relative), packages, nvcc, env}."""
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                executable=rel(sys.executable), packages=packages(names), nvcc=nvcc(), env=environment())


def harness_commit():
    """{commit, dirty} of the checkout the harness runs from (git), None outside a git checkout. dirty: tracked files
    differ from the commit."""
    if not (REPO / ".git").exists():
        return None
    c = _run(["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"])
    if c is None:
        return None
    st = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    return dict(commit=c, dirty=None if st is None else bool(st))


def nablaengine_provenance():
    """nablaengine.provenance.provenance() when nablaengine is importable here, else None."""
    try:
        from nablaengine import provenance as dkp
        return dkp.provenance()
    except Exception:  # noqa: BLE001  (not installed, or no GPU / nvcc on this machine)
        return None


def provenance():
    """The provenance block of a job document: the harness commit and NablaEngine's own provenance."""
    return dict(harness=harness_commit(), nablaengine=nablaengine_provenance())


def is_committed(path):
    """True when path is tracked by git in this checkout (a committed result must never be overwritten)."""
    p = Path(path).absolute()
    try:
        p.relative_to(REPO)
    except ValueError:
        return False
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(p)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return r.returncode == 0
