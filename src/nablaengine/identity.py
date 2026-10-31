"""Identity of the engine's build units: the hashes and comparisons behind the G1 identity gate
(scripts/identity_gate.py and its manifest).

A build unit is one nvcc translation unit of a production build: its generated CUDA source, the -D options and the
nvcc command line its build function gives it (nablaengine.jax_ffi.build_plan / unit_command for the FFI library,
nablaengine.cuda.compile_command for the check binaries), the packed model constants it is uploaded with, and the file
name nvcc sees. Three levels, each cheaper than the next and each implied by the one before:

  L0  the unit as generated (no compiler run): sha256 of its source text, its -D options, the packed constants and
      the nvcc arguments (the nvcc path and the include directories as placeholders)
  L1  the preprocessed source per pass: nvcc's device pass (nvcc -E -Xcompiler -P, __CUDA_ARCH__ of the unit's
      virtual architecture) and its host pass (the host compiler's preprocessing step of nvcc's own compile, taken
      from nvcc -dryrun, with -P), blank lines collapsed. A change that the G1 build compiles out (code behind a
      generated #if that is 0 for the G1) leaves L1 unchanged
  L2  code generation, from one compile of the unit with the production command plus -keep: the PTX (comment lines
      dropped), the SASS (cuobjdump -sass, header lines dropped, white space collapsed), the resource usage per kernel
      (cuobjdump -res-usage: registers, stack, shared, local memory) and the host translation (cudafe++'s output and
      the kernel stubs, line markers dropped). A change in dead code or comments leaves L2 unchanged; anything else
      that changes L2 changes what the GPU runs (or the host code that launches it)

What a deeper identical level proves after an L0 difference (clearing_level): L1 and L2 are compiler outputs, so they
cannot see the packed constants (data uploaded at run time): a unit whose constants differ differs, whatever L1 and L2
say. L1 is the preprocessor's output, so it clears only a change of the source or of the -D options; a change of any
other nvcc option (-fmad, -Xptxas, -maxrregcount, ...) acts after preprocessing and needs an identical L2.

Normalisation (checked by the gate itself: every unit rebuilt in other directories hashes the same): nvcc names
entities with internal linkage (an anonymous namespace, static functions) after the unit's module id, which contains a
hash of the source file's absolute path (e.g. _78236708_4_u_cu_launch in _ZN34_GLOBAL__N__78236708_4_u_cu_launch1kEPf),
so the same source compiled in two directories gives different PTX, SASS, cubins and host code. Every text of a unit
has the module id that nvcc wrote (-keep's <file>.module_id) replaced by <MODULE> (and any remaining _INTERNAL_ /
_GLOBAL__N__ path hash by the regexes of MANGLE), which makes them identical across directories; units are compiled
under their production file names.
Raw cubins and objects are never compared (not reproducible across directories; the FFI library's cache key is
source-based for the same reason).

Architectures: a unit whose real architecture the local nvcc cannot target (sm_120 with nvcc 12.4: its production
flags are -arch=compute_90 -code=sm_120, the NVVM 7 front end) is compiled for its virtual architecture only (the PTX
nvcc would hand ptxas) and that PTX assembled by a separate ptxas that knows the real architecture (ptxas -arch=sm_120
-m64, as nvcc runs it; $PTXAS, else the pip nvidia.cuda_nvcc package's, else next to nvcc). The SASS of such a unit is
that ptxas's, so the toolchain record (toolchain()) names it; manifest entries are keyed by the architecture and the
toolchain (nvcc, host compiler, ptxas where separate, cuobjdump), because SASS depends on all of them.

Tools: nvcc (nablaengine.cuda.nvcc), cuobjdump ($CUOBJDUMP, else next to nvcc, on PATH, or a conda package cache;
find_tool). cuobjdump must know the architecture it disassembles (sm_120: CUDA 12.8 or newer).
"""

import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "dk-identity/1"
LEVELS = (0, 1, 2)
# internal-linkage names whose path hash the module-id substitution did not catch (another nvcc's spelling)
MANGLE = (re.compile(r"_INTERNAL_[0-9a-f]{8}_[0-9]+_[A-Za-z0-9_]+?_cu_[0-9a-f]{8}"),
          re.compile(r"_GLOBAL__N__[0-9a-f]{8}_[0-9]+_[A-Za-z0-9_]+?_cu_[0-9a-f]{8}"))
SASS_HEADER = ("Fatbin", "code for", "arch =", "code version", "host =", "compile_size", "identifier", "producer")
NVCC_PLACEHOLDER = "nvcc"


def sha256(x):
    """sha256 hex digest of a str (UTF-8), bytes or array (its raw bytes, C order)."""
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x).view(np.uint8).tobytes()
    elif isinstance(x, str):
        x = x.encode()
    return hashlib.sha256(x).hexdigest()


def normalize_symbols(text, module_ids=()):
    """The text with the unit's module ids (nvcc's -keep <file>.module_id) replaced by <MODULE>, then any internal-linkage
    path hash left (MANGLE) by <INTERNAL> / <ANON>."""
    for mid in sorted({m for m in module_ids if m}, key=len, reverse=True):
        text = text.replace(mid, "<MODULE>")
    return MANGLE[1].sub("<ANON>", MANGLE[0].sub("<INTERNAL>", text))


def normalize_preprocessed(text):
    """Preprocessed source: trailing white space dropped, runs of blank lines collapsed (-P leaves blank lines where
    directives were)."""
    lines = [x.rstrip() for x in text.split("\n")]
    return "\n".join(x for x in lines if x)


def normalize_ptx(text, module_ids=()):
    """PTX without its comment lines (the compiler build and version header; the toolchain is recorded separately)."""
    return normalize_symbols("\n".join(x for x in text.split("\n") if not x.lstrip().startswith("//")), module_ids)


def normalize_sass(text, module_ids=()):
    """cuobjdump -sass output without its header lines (file, architecture, ELF identification), white space collapsed."""
    out = []
    for line in normalize_symbols(text, module_ids).split("\n"):
        s = line.strip()
        if not s or s.startswith(SASS_HEADER):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return "\n".join(out)


def normalize_host(text, module_ids=()):
    """cudafe++'s host translation or stub file as the host compiler sees it: the #if 0 blocks dropped (cudafe++ keeps
    the bodies of device functions and kernels there, so device-only code does not count as host code), then every
    other preprocessor line (line markers, the #include of the stub and the fatbin, which carry temporary file names)."""
    out, skip = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            d = s[1:].strip()
            if skip:
                if re.match(r"if(def|ndef)?\b", d):
                    skip += 1
                elif re.match(r"endif\b", d) or (skip == 1 and re.match(r"(else|elif)\b", d)):
                    skip -= 1
            elif re.match(r"if\s+0\s*$", d):
                skip = 1
            continue
        if not skip:
            out.append(line.rstrip())
    return normalize_symbols("\n".join(out), module_ids)


def resource_usage(text, module_ids=()):
    """{kernel / function: {REG, STACK, SHARED, LOCAL, ...}} from cuobjdump -res-usage (names normalized)."""
    out, name = {}, None
    for line in normalize_symbols(text, module_ids).split("\n"):
        m = re.match(r"\s*Function (\S+?):?\s*$", line)
        if m:
            name = m.group(1)
            continue
        if name and "REG:" in line:
            out[name] = {k: int(v) for k, v in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", line)}
            name = None
    return dict(sorted(out.items()))


def _run(cmd, cwd=None, check=True, env=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=env)
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout[-4000:]}")
    return r.stdout


def find_tool(name, env_var=None):
    """A CUDA binary tool (cuobjdump, nvdisasm): ${env_var or NAME.upper()}, next to nvcc, on PATH, or a conda package
    cache (cuda-<name>-12*: $CONDA_PKGS_DIRS, $CONDA_EXE's installation, ~/miniconda3, ~/miniforge3, ~/anaconda3);
    None if none exists."""
    from nablaengine import cuda
    cands = [os.environ.get(env_var or name.upper())]
    try:
        cands.append(str(Path(cuda.nvcc()).parent / name))
    except FileNotFoundError:
        pass
    cands.append(shutil.which(name))
    pkgs = [Path(p) for p in os.environ.get("CONDA_PKGS_DIRS", "").split(",") if p]
    if os.environ.get("CONDA_EXE"):
        pkgs.append(Path(os.environ["CONDA_EXE"]).resolve().parents[1] / "pkgs")
    pkgs += [Path.home() / d / "pkgs" for d in ("miniconda3", "miniforge3", "anaconda3")]
    cands += [str(p) for d in pkgs for p in sorted(d.glob(f"cuda-{name}-12*/bin/{name}"))]
    return next((c for c in cands if c and Path(c).is_file()), None)


@functools.lru_cache(maxsize=None)
def tool_version(path):
    """'<release> (<build>)' of a CUDA tool's --version (e.g. '12.4.131 (cuda_12.4.r12.4/compiler.34097967_0)'), or of
    the host compiler's first --version line; None if it does not run."""
    try:
        out = _run([path, "--version"])
    except (OSError, RuntimeError):
        return None
    rel = re.search(r"\bV(\d+\.\d+\.\d+)", out)
    build = re.search(r"Build (\S+)", out)
    if rel:
        return rel.group(1) + (f" ({build.group(1)})" if build else "")
    return out.strip().split("\n")[0] or None


def release(version):
    """'12.4.131' of tool_version's '12.4.131 (...)'."""
    return (version or "").split(" ")[0] or None


@functools.lru_cache(maxsize=None)
def nvcc_gpu_codes(nvcc_path):
    """The real architectures nvcc can target (nvcc --list-gpu-code)."""
    return frozenset(_run([nvcc_path, "--list-gpu-code"]).split())


@functools.lru_cache(maxsize=None)
def ptxas_supports(ptxas_path, arch):
    """Whether this ptxas assembles for arch (a tiny PTX for that architecture's virtual architecture)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dk_ptxas_") as d:
        ptx = Path(d) / "k.ptx"
        ver = "8.7" if int(arch[3:].rstrip("af")) >= 100 else "7.0"
        ptx.write_text(f".version {ver}\n.target {arch}\n.address_size 64\n.visible .entry k() {{ ret; }}\n")
        r = subprocess.run([ptxas_path, f"-arch={arch}", "-m64", str(ptx), "-o", str(Path(d) / "k.cubin")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return r.returncode == 0


def find_ptxas(arch, nvcc_path):
    """A ptxas that assembles for arch: $PTXAS, the pip package nvidia.cuda_nvcc's (the JAX stack's ptxas), next to
    nvcc; None if none does."""
    cands = [os.environ.get("PTXAS")]
    try:
        import nvidia.cuda_nvcc
        cands.append(str(Path(list(nvidia.cuda_nvcc.__path__)[0]) / "bin" / "ptxas"))
    except ImportError:
        pass
    cands.append(str(Path(nvcc_path).parent / "ptxas"))
    return next((c for c in cands if c and Path(c).is_file() and ptxas_supports(c, arch)), None)


def host_compiler(nvcc_path):
    """The host compiler nvcc runs (its -dryrun's first gcc / g++ / clang command), as a path or name."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dk_hc_") as d:
        (Path(d) / "k.cu").write_text("int main() { return 0; }\n")
        out = _run([nvcc_path, "-dryrun", "-c", "-o", "k.o", "k.cu"], cwd=d, check=False)
    for line in out.split("\n"):
        if line.startswith("#$ "):
            args = shlex.split(line[3:])
            if args and Path(args[0]).name.startswith(("gcc", "g++", "cc", "c++", "clang")):
                return args[0]
    return None


@dataclass
class Unit:
    """One nvcc translation unit of a production build. command: nvcc's arguments after the nvcc path, with "{src}" and
    "{out}" for the source and output paths (the build function's own command, e.g. [..., "-c", "-o", "{out}",
    "{src}"]); include: {placeholder: directory} for -I options whose directory is machine-specific (written as
    "-I{placeholder}" in command)."""
    name: str
    file: str
    source: str
    command: list
    defines: list
    consts: object = None
    include: dict = field(default_factory=dict)

    def args(self, src, out, command=None):
        """command (default: the unit's) with the paths filled in."""
        rep = {"{src}": str(src), "{out}": str(out)}
        rep.update({f"-I{{{k}}}": f"-I{v}" for k, v in self.include.items()})
        return [rep.get(a, a) for a in (self.command if command is None else command)]


def arch_of(command):
    """(virtual architecture, real architecture) of an nvcc command: -arch=sm_XY gives (compute_XY, sm_XY),
    -arch=compute_XY -code=sm_ZW (compute_XY, sm_ZW)."""
    a = next(x.split("=", 1)[1] for x in command if x.startswith("-arch="))
    code = next((x.split("=", 1)[1] for x in command if x.startswith("-code=")), None)
    if a.startswith("sm_"):
        return "compute_" + a[3:], code or a
    return a, code or a


def level0(unit):
    """L0: the unit as generated, without a compiler run."""
    out = dict(source=sha256(unit.source), defines=sha256(json.dumps(unit.defines)),
               command=sha256(json.dumps(unit.command)), source_bytes=len(unit.source.encode()))
    if unit.consts is not None:
        c = np.ascontiguousarray(unit.consts, np.float32)
        out.update(consts=sha256(c), consts_count=int(c.size))
    return out


def _strip(args, drop_code):
    """args without the output / input / compile-only parts (-c, -o X, the source) and, if drop_code, without -code=."""
    out, it = [], iter(args)
    for a in it:
        if a == "-o":
            next(it, None)
            continue
        if a in ("-c", "{src}") or (drop_code and a.startswith("-code=")):
            continue
        out.append(a)
    return out


class Toolchain:
    """The compilers and tools of this machine for one real architecture (see the module docstring)."""

    def __init__(self, nvcc_path=None, cuobjdump=None):
        from nablaengine import cuda
        self.nvcc = nvcc_path or cuda.nvcc()
        self.cuobjdump = cuobjdump or find_tool("cuobjdump")
        self.host = host_compiler(self.nvcc)

    def route(self, real):
        """'nvcc' if nvcc targets real itself, 'ptxas' if it needs the separate ptxas, None if neither can."""
        if real in nvcc_gpu_codes(self.nvcc):
            return "nvcc"
        return "ptxas" if find_ptxas(real, self.nvcc) else None

    def record(self, real):
        """{nvcc, host, ptxas (route ptxas only), cuobjdump}: tool versions, the manifest's toolchain key."""
        rec = dict(nvcc=tool_version(self.nvcc), host=tool_version(self.host) if self.host else None)
        if self.route(real) == "ptxas":
            rec["ptxas"] = tool_version(find_ptxas(real, self.nvcc))
        rec["cuobjdump"] = tool_version(self.cuobjdump) if self.cuobjdump else None
        return rec

    def key(self, real):
        """The manifest key of this toolchain for real: '<arch> nvcc <release>[ ptxas <release>] host <...> cuobjdump
        <release>'."""
        r = self.record(real)
        parts = [real, f"nvcc {release(r['nvcc'])}"] + ([f"ptxas {release(r['ptxas'])}"] if r.get("ptxas") else [])
        host = re.search(r"(\d+\.\d+(\.\d+)?)", r["host"] or "")
        hc = Path(self.host).name.split("-")[0] if self.host else "?"
        parts += [f"{hc} {host.group(1) if host else '?'}", f"cuobjdump {release(r['cuobjdump'])}"]
        return " ".join(parts)

    def level1(self, unit, work):
        """L1 of the unit (work: an empty directory, kept by the caller): the device and the host pass."""
        virt, real = arch_of(unit.command)
        drop_code = self.route(real) != "nvcc"
        src = Path(work) / unit.file
        src.write_text(unit.source)
        base = unit.args(src.name, "unit.o", _strip(unit.command, drop_code))
        pp = Path(work) / "device.ii"
        _run([self.nvcc, *base, "-E", "-Xcompiler", "-P", "-o", pp.name, src.name], cwd=work)
        out = dict(virtual=virt, device=sha256(normalize_preprocessed(pp.read_text())))
        host = self._host_preprocess(base, src.name, work)
        if host is None:  # (a missing host pass would leave host-code changes unseen at L1)
            raise RuntimeError(f"{unit.name}: no host preprocessing step in nvcc -dryrun ({self.nvcc})")
        out["host"] = sha256(normalize_preprocessed(host))
        return out

    def _host_preprocess(self, base, src, work):
        """The host pass's preprocessed source: the host compiler's -E command of nvcc -dryrun (the one without
        __CUDA_ARCH__), run with -P into a file of ours; None if the dry run has no such command."""
        dry = _run([self.nvcc, "-dryrun", *base, "-c", "-o", "unit_dry.o", src], cwd=work)
        for line in dry.split("\n"):
            if not line.startswith("#$ "):
                continue
            args = shlex.split(line[3:])
            if (not args or "-E" not in args or any(a.startswith("-D__CUDA_ARCH__=") for a in args)
                    or not Path(args[0]).name.startswith(("gcc", "g++", "cc", "c++", "clang"))):
                continue
            if "-o" in args:
                i = args.index("-o")
                args[i + 1] = "host.ii"
            else:
                args += ["-o", "host.ii"]
            _run([*args[:1], "-P", *args[1:]], cwd=work)
            return (Path(work) / "host.ii").read_text()
        return None

    def level2(self, unit, work):
        """L2 of the unit (work: an empty directory, kept by the caller): one compile with the production command plus
        -keep, then the PTX, SASS, resource usage and host translation."""
        virt, real = arch_of(unit.command)
        route = self.route(real)
        if route is None:
            raise RuntimeError(f"{unit.name}: neither nvcc ({self.nvcc}) nor a ptxas ($PTXAS) targets {real}")
        if self.cuobjdump is None:
            raise RuntimeError("no cuobjdump: set CUOBJDUMP (a CUDA 12.8+ cuobjdump also reads sm_120)")
        src = Path(work) / unit.file
        src.write_text(unit.source)
        keep = Path(work) / "keep"
        keep.mkdir()
        args = unit.args(src.name, "unit.out")
        if route == "ptxas":
            args = [a for a in args if not a.startswith("-code=")]
            if any(a.startswith(("-Xptxas", "--ptxas-options", "-maxrregcount")) for a in args):
                raise RuntimeError(f"{unit.name}: ptxas options in the command; the separate ptxas route does not "
                                   "forward them")
        _run([self.nvcc, *args, "-keep", "-keep-dir", "keep"], cwd=work)
        stem = Path(unit.file).stem
        mids = [p.read_text().strip() for p in keep.glob(f"{stem}.module_id")]
        ptx = keep / f"{stem}.ptx"  # (an executable's device-link stub, <stem>_dlink.*, is not the unit's code)
        if not ptx.exists():
            raise RuntimeError(f"{unit.name}: no {ptx.name} in nvcc's -keep output")
        if route == "ptxas":
            cubin = keep / f"{stem}.{real}.ptxas.cubin"
            _run([find_ptxas(real, self.nvcc), f"-arch={real}", "-m64", ptx, "-o", cubin])
        else:
            cubin = keep / f"{stem}.{real}.cubin"
            if not cubin.exists():
                raise RuntimeError(f"{unit.name}: no {cubin.name} in nvcc's -keep output")
        sass = normalize_sass(_run([self.cuobjdump, "-sass", cubin]), mids)
        if "Function" not in sass:
            raise RuntimeError(f"{unit.name}: cuobjdump ({self.cuobjdump}) printed no SASS for {real}")
        host = []
        for suffix in (".cudafe1.cpp", ".cudafe1.stub.c"):
            p = keep / f"{stem}{suffix}"
            host.append(normalize_host(p.read_text(errors="replace"), mids) if p.exists() else "")
        return dict(route=route, ptx_virtual=virt, ptx=sha256(normalize_ptx(ptx.read_text(), mids)),
                    sass=sha256(sass), sass_lines=sass.count("\n") + 1,
                    resources=resource_usage(_run([self.cuobjdump, "-res-usage", cubin]), mids),
                    host=sha256("\n//--\n".join(host)))


def options(command):
    """An nvcc command line (a manifest entry's "command" string or an argument list) without its -D options: the
    options that also act after preprocessing (code generation, ptxas, the host compiler)."""
    args = command.split() if isinstance(command, str) else [str(a) for a in command]
    return [a for a in args if not a.startswith("-D")]


def clearing_level(old, new):
    """The first level at which a unit whose L0 differs between a manifest entry and a recomputed one may still be
    identical (module docstring): 0 if L0 is identical; None if the packed constants differ (the units are uploaded
    with them at run time, no compiler output contains them: the G1 runs other numbers whatever L1 and L2 say); 2 if
    the nvcc options other than -D differ (e.g. -fmad=false, -Xptxas, -maxrregcount: they act on code generation
    after preprocessing, so an identical L1 does not show an identical build); else 1 (the source or the -D options
    differ: an identical preprocessed source shows a change the G1 build compiles out)."""
    diff = compare(old, new).get(0, [])
    if not diff:
        return 0
    if {"consts", "consts_count"} & set(diff):
        return None
    if options(old.get("command", "")) != options(new.get("command", "")):
        return 2
    return 1


def compare(old, new):
    """Differences between a manifest unit entry and a recomputed one, per level: {0: [field, ...], 1: [...], 2: [...]}
    (a level absent from either side is not compared). L2's resource usage counts as part of the SASS."""
    diff = {}
    for lv in LEVELS:
        a, b = old.get(f"L{lv}"), new.get(f"L{lv}")
        if a is None or b is None:
            continue
        keys = sorted(set(a) | set(b))
        diff[lv] = [k for k in keys if a.get(k) != b.get(k)]
    return diff


def manifest_skeleton(about, model):
    return dict(schema=SCHEMA, about=about, model=model, normalization=dict(
        module_id="nvcc -keep's <file>.module_id replaced by <MODULE> in PTX, SASS, resource names and host code",
        mangle=[p.pattern for p in MANGLE], sass_header_dropped=list(SASS_HEADER),
        ptx="comment lines dropped", host="#if 0 blocks (device code bodies) and preprocessor lines dropped",
        preprocessed="trailing white space and blank lines dropped"), targets={})
