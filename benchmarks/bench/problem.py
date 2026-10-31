"""The physics problems every adapter solves, the per-world domain randomization (DR) for MuJoCo C, and the corpus
loader with its fixed seeded state selection. numpy and the standard library at import; mujoco only inside
load_model / DR (every engine environment has it).

Track A (headline, every engine): the mjlab G1, fixtures/mjlab/g1_train.mjb with g1_meta.json (foot-floor pairs,
torso body). One control step = 4 x mj_step of 5 ms with ctrl held (qfrc_applied 0); Newton, pyramidal cone,
implicitfast as stored; iterations set explicitly by every cell (10 headline, 50 reference; the .mjb stores 5, which
is never used), ls_iterations 8, tolerance 1e-8, sensors off (mjDSBL_SENSOR). Per-world DR from the corpus:
pair_friction[foot_pairs, 0:2] = friction[w] and body_ipos[torso] = body_ipos0[torso] + com[w], without
mj_setConst (as nablaengine.reference.warp_physics._set_dr and benchmarks/bench/truth/fdlib._set_dr; MuJoCo Warp keeps
the compiled invweight0 too).
Track B (MJX, MJW, MuJoCo C only): MuJoCo Menagerie unitree_g1/scene_mjx.xml at the commit pinned by
benchmarks/bench/models (its path is an argument), native options, 5 x 4 ms = one 20 ms control step, no DR.

  from benchmarks.bench import problem as P
  prob = P.get("mjlab")                        # or P.get("menagerie_g1", model_path=...)
  m = P.load_model(prob, iterations=10)        # MjModel with the problem's options (Track A needs iterations)
  dr = P.DR(m, prob)
  dr.apply(m, friction[w], com[w])             # MuJoCo C, one world (in place, relative to the nominal values)
  pf, bi = dr.batched(friction, com, np.float32)   # (B, npair, 5), (B, nbody, 3) for batched models (None: no DR)
  corpus = P.load_corpus("mf_s0")              # benchmarks/data/mf_s0.npz
  states = P.select(corpus, nworld=1024, subset=1)  # a fixed seeded permutation; subsets are disjoint

Corpora. Benchmark corpora (kind "bench", written by benchmarks/bench/corpus): benchmarks/data/<name>.npz (directory:
$DK_BENCH_DATA, default benchmarks/data) with qpos (S, nq), qvel (S, nv), qacc_warmstart (S, nv), ctrl (S, nu),
friction (S,) and com (S, 3) (Track A), cot_qpos (S, 3, nq), cot_qvel (S, 3, nv), and metadata (0-d entries; a JSON
string under `metadata` / `meta` / `meta_json`; other per-state arrays such as label_* are kept in Corpus.extra).
Corpora in the engine's fixture format (kind "fixture"): <name>_fwd.npz (qpos, qvel, qacc_warmstart, ctrl, friction,
com, ...) and <name>_adj.npz (cot{c}_qpos, cot{c}_qvel, c = 0..2), looked up by name in $DK_BENCH_FIXTURE_DIR only
when that variable is set (unset by default: names resolve to benchmark corpora only), or given as a path to the
_fwd.npz file. "a+b" concatenates corpora (in that order). Inputs keep their stored dtype (float32); engines in
float64 convert them exactly.

Selection. select(corpus, nworld, subset) takes the states perm[subset * nworld:(subset + 1) * nworld] of
perm = numpy.random.default_rng(SELECT_SEED).permutation(S): never "the first n", and subsets 0..S // nworld - 1 are
disjoint. nworld > S tiles the permutation (subset 0 only; States.tiled = nworld / S, recorded).
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRACK_A_MODEL = "fixtures/mjlab/g1_train.mjb"
TRACK_A_META = "fixtures/mjlab/g1_meta.json"
TRACK_B_FILE = "mujoco_menagerie/unitree_g1/scene_mjx.xml"  # under the fetched models' root
MODELS_ROOT_DEFAULT = "runs/bench/models"  # $DK_BENCH_MODELS: where benchmarks/bench/models fetches the models
DATA_DEFAULT = "benchmarks/data"  # $DK_BENCH_DATA: the benchmark corpora
FIXTURE_DIR_VAR = "DK_BENCH_FIXTURE_DIR"  # optional directory of fixture-format corpora (unset: none)
SELECT_SEED = 20261028
INPUTS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com")
STATE_KEYS = ("qpos", "qvel", "qacc_warmstart", "ctrl")
DR_KEYS = ("friction", "com")
NCOT = 3


def _abs(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO / p


def _rel(p):
    from benchmarks.bench import prov
    return prov.rel(p)


# ------------------------------------------------------------------------------------------------- problems

@dataclass(frozen=True)
class Problem:
    """One physics problem. iterations is not part of it: every cell sets it (Track A must, explicit_iterations)."""
    id: str
    track: str
    label: str
    model: str  # model file (.mjb / .xml), repository-relative where possible
    substeps: int
    timestep: float = None  # expected opt.timestep (checked); None: not checked
    explicit_iterations: bool = True  # load_model refuses iterations=None
    ls_iterations: int = None  # None: the model's own
    tolerance: float = None  # None: the model's own
    sensors: bool = False  # False: mjDSBL_SENSOR
    dr: bool = False  # per-world foot friction + torso COM from the states
    foot_pairs: tuple = None
    torso_body: str = None
    meta: str = None  # g1_meta.json (Track A)
    source: dict = field(default=None, hash=False, compare=False)  # where the model comes from (repo, commit, file)

    def path(self):
        return _abs(self.model)

    def to_dict(self):
        d = asdict(self)
        d["foot_pairs"] = list(self.foot_pairs) if self.foot_pairs is not None else None
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if d.get("foot_pairs") is not None:
            d["foot_pairs"] = tuple(int(x) for x in d["foot_pairs"])
        return cls(**d)

    def control_dt(self):
        return None if self.timestep is None else self.substeps * self.timestep

    def describe(self):
        """The job document's models entry: path, sha256 (and of the model's directory for an XML with assets),
        source, the problem's options."""
        p = self.path()
        d = dict(self.to_dict(), path=_rel(p), exists=p.exists())
        d.pop("model")
        if p.exists():
            d["sha256"] = _sha256(p)
            if p.suffix == ".xml":
                d["dir_sha256"] = dir_sha256(p.parent)
        return d


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dir_sha256(d):
    """sha256 over 'sha256  <relative path>' lines of every file below d in byte order (an XML model and its
    assets)."""
    d = Path(d)
    files = sorted(p for p in d.rglob("*") if p.is_file())
    lines = "".join(f"{_sha256(p)}  {p.relative_to(d).as_posix()}\n" for p in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def track_a(model_path=None, meta_path=None, sensors=False, tolerance=1e-8):
    """The mjlab G1 problem (module docstring). model_path / meta_path default to fixtures/mjlab."""
    meta_p = _abs(meta_path or (Path(model_path).with_name("g1_meta.json") if model_path else TRACK_A_META))
    mt = json.loads(meta_p.read_text())
    return Problem(id="mjlab", track="A", label="mjlab G1 (fixtures/mjlab/g1_train.mjb)",
                   model=_rel(_abs(model_path or TRACK_A_MODEL)), substeps=int(mt["substeps_per_control_step"]),
                   timestep=float(mt["timestep"]), explicit_iterations=True, ls_iterations=8, tolerance=tolerance,
                   sensors=sensors, dr=True, foot_pairs=tuple(int(x) for x in mt["foot_pairs"]),
                   torso_body=mt["torso_body"], meta=_rel(meta_p),
                   source=dict(kind="repository", mujoco_version=mt.get("mujoco_version")))


def models_root():
    return _abs(os.environ.get("DK_BENCH_MODELS") or MODELS_ROOT_DEFAULT)


def _pins():
    """benchmarks/bench/models/pins.json (owned by the models fetcher) if present: the pinned commits."""
    p = REPO / "benchmarks" / "bench" / "models" / "pins.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def track_b(model_path=None, iterations_native=True, sensors=False):
    """Menagerie unitree_g1/scene_mjx.xml (module docstring): model_path, else the fetched copy under
    $DK_BENCH_MODELS (default runs/bench/models)/mujoco_menagerie/unitree_g1/scene_mjx.xml. Native solver options
    (cells pass iterations None for the model's own, or a number for the parity row), 5 substeps of 4 ms, no DR."""
    p = _abs(model_path) if model_path else models_root() / TRACK_B_FILE
    pins = _pins()
    pin = None
    if isinstance(pins, dict):
        pin = pins.get("menagerie_g1") or next((v for k, v in pins.items() if "menagerie" in k.lower()), None)
    if isinstance(pin, dict):  # the per-file hashes stay in pins.json; describe() hashes the model directory
        pin = {k: v for k, v in pin.items() if k != "files"}
    return Problem(id="menagerie_g1", track="B", label="MuJoCo Menagerie unitree_g1/scene_mjx.xml",
                   model=_rel(p), substeps=5, timestep=0.004, explicit_iterations=not iterations_native,
                   ls_iterations=None, tolerance=None, sensors=sensors, dr=False,
                   source=dict(kind="menagerie", file="unitree_g1/scene_mjx.xml", pin=pin))


PROBLEMS = {"mjlab": track_a, "menagerie_g1": track_b}


def get(name, model_path=None):
    """The problem by id (PROBLEMS), with an optional model file."""
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r} (one of {sorted(PROBLEMS)})")
    return PROBLEMS[name](model_path=model_path)


def load_model(prob, iterations=None):
    """The MjModel of prob with its options: iterations (None: the model's own, refused for Track A),
    ls_iterations / tolerance where the problem sets them, mjDSBL_SENSOR unless sensors; the timestep is checked."""
    import mujoco
    p = prob.path()
    if not p.exists():
        raise FileNotFoundError(f"model file {_rel(p)} not found" + (
            " (fetch it: python -m benchmarks.bench fetch-models, or pass --model-path)" if prob.track == "B" else ""))
    if prob.meta and p.suffix == ".mjb":
        mv = json.loads(_abs(prob.meta).read_text()).get("mujoco_version")
        if mv and mv != mujoco.__version__:
            raise RuntimeError(f"{p.name} was saved by MuJoCo {mv}, this is {mujoco.__version__}")
    m = mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))
    if prob.timestep is not None and abs(m.opt.timestep - prob.timestep) > 1e-12:
        raise ValueError(f"{prob.id}: opt.timestep {m.opt.timestep} != {prob.timestep}")
    if iterations is None and prob.explicit_iterations:
        raise ValueError(f"{prob.id}: iterations must be set explicitly (10 or 50; the .mjb stores "
                         f"{m.opt.iterations})")
    if iterations is not None:
        m.opt.iterations = int(iterations)
    if prob.ls_iterations is not None:
        m.opt.ls_iterations = int(prob.ls_iterations)
    if prob.tolerance is not None:
        m.opt.tolerance = float(prob.tolerance)
    if not prob.sensors:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    return m


def options(m):
    """The effective solver options of an MjModel (for provenance)."""
    import mujoco
    return dict(timestep=float(m.opt.timestep), iterations=int(m.opt.iterations),
                ls_iterations=int(m.opt.ls_iterations), tolerance=float(m.opt.tolerance),
                solver=mujoco.mjtSolver(m.opt.solver).name, cone=mujoco.mjtCone(m.opt.cone).name,
                integrator=mujoco.mjtIntegrator(m.opt.integrator).name, disableflags=int(m.opt.disableflags),
                sensors=not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SENSOR)),
                nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), nbody=int(m.nbody), npair=int(m.npair))


class DR:
    """Per-world domain randomization of a problem on MuJoCo C models (no mj_setConst). Holds the nominal
    pair_friction and body_ipos of the model it was made from; a problem without DR makes every call a no-op."""

    def __init__(self, m, prob):
        import mujoco
        self.enabled = bool(prob.dr)
        self.fric0 = np.array(m.pair_friction, np.float64)
        self.ipos0 = np.array(m.body_ipos, np.float64)
        if self.enabled:
            self.foot = np.asarray(prob.foot_pairs, np.int64)
            self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prob.torso_body)
            if self.torso < 0:
                raise ValueError(f"no body {prob.torso_body!r} in the model")
            if self.foot.size and (self.foot.min() < 0 or self.foot.max() >= m.npair):
                raise ValueError("foot_pairs outside the model's pairs")

    def apply(self, m, friction, com):
        """One world, in place (as truth/fdlib._set_dr): pair_friction = nominal, then [foot, :2] = friction;
        body_ipos[torso] = nominal + com. friction and com are read exactly as float64."""
        if not self.enabled:
            return m
        m.pair_friction[:] = self.fric0
        m.pair_friction[self.foot, :2] = np.float64(friction)
        m.body_ipos[self.torso] = self.ipos0[self.torso] + np.asarray(com, np.float64)
        return m

    def reset(self, m):
        """The nominal values back."""
        m.pair_friction[:] = self.fric0
        m.body_ipos[:] = self.ipos0
        return m

    def batched(self, friction, com, dtype=np.float64):
        """(pair_friction (B, npair, 5), body_ipos (B, nbody, 3)) in dtype for batched models (mujoco.batch
        expand, MuJoCo Warp put_model batch_sizes, MJX in_axes); the arithmetic in dtype (float32: as MuJoCo Warp's
        _set_dr, float32(ipos0) + com). (None, None) without DR."""
        if not self.enabled:
            return None, None
        f = np.asarray(friction).astype(dtype)
        c = np.asarray(com).astype(dtype)
        B = len(f)
        pf = np.repeat(self.fric0.astype(dtype)[None], B, 0)
        pf[:, self.foot, 0] = f[:, None]
        pf[:, self.foot, 1] = f[:, None]
        bi = np.repeat(self.ipos0.astype(dtype)[None], B, 0)
        bi[:, self.torso] = self.ipos0[self.torso].astype(dtype) + c
        return pf, bi

    def models(self, m, friction, com):
        """One MjModel copy per world with its DR (for APIs that take a model list, e.g. mujoco.rollout)."""
        import copy
        out = []
        for f, c in zip(friction, com):
            mm = copy.copy(m)
            self.apply(mm, f, c)
            out.append(mm)
        return out


# ------------------------------------------------------------------------------------------------- corpora

@dataclass
class Corpus:
    name: str
    kind: str  # "bench" | "fixture"
    files: list  # [{path (relative), sha256}]
    arrays: dict  # INPUTS present (stored dtype)
    cot_qpos: np.ndarray = None  # (S, 3, nq)
    cot_qvel: np.ndarray = None  # (S, 3, nv)
    meta: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # other per-state arrays (S, ...)

    @property
    def n(self):
        return len(self.arrays["qpos"])

    def subsets(self, nworld):
        """Number of disjoint subsets of nworld states (1 when nworld >= S: tiled)."""
        return max(1, self.n // nworld)

    def describe(self):
        """The job document's corpora entry."""
        return dict(kind=self.kind, n=self.n, files=self.files, sha256=self.files[0]["sha256"],
                    path=self.files[0]["path"], cotangents=self.cot_qpos is not None,
                    dr=all(k in self.arrays for k in DR_KEYS), meta=_jsonable(self.meta),
                    extra={k: list(v.shape) for k, v in self.extra.items()})


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def data_dir():
    return _abs(os.environ.get("DK_BENCH_DATA") or DATA_DEFAULT)


def fixture_dir():
    """$DK_BENCH_FIXTURE_DIR as a path, or None when unset (the default: no fixture-format corpora by name)."""
    v = os.environ.get(FIXTURE_DIR_VAR)
    return _abs(v) if v else None


def corpus_files(name):
    """(kind, main file, adjoint file or None) of a corpus name or path; FileNotFoundError listing the places tried."""
    p = Path(name)
    if name.endswith(".npz") or "/" in name:
        p = _abs(name)
        if p.name.endswith("_fwd.npz"):
            adj = p.with_name(p.name[:-len("_fwd.npz")] + "_adj.npz")
            return "fixture", p, adj if adj.exists() else None
        return "bench", p, None
    bench = data_dir() / f"{name}.npz"
    if bench.exists():
        return "bench", bench, None
    tried = [_rel(bench)]
    fdir = fixture_dir()
    if fdir is not None:
        fwd = fdir / f"{name}_fwd.npz"
        if fwd.exists():
            adj = fwd.with_name(f"{name}_adj.npz")
            return "fixture", fwd, adj if adj.exists() else None
        tried.append(_rel(fwd))
    raise FileNotFoundError(f"corpus {name!r}: not found ({' nor '.join(tried)})")


def corpus_exists(name):
    try:
        for part in name.split("+"):
            corpus_files(part)
        return True
    except FileNotFoundError:
        return False


def corpus_size(name):
    """Number of states of a corpus (or 'a+b'), read from the npz header only."""
    n = 0
    for part in name.split("+"):
        _, main, _ = corpus_files(part)
        with np.load(main) as z:
            n += z["qpos"].shape[0]  # np.load of an npz reads members lazily
    return n


def _load_one(name):
    kind, main, adj = corpus_files(name)
    files = [dict(path=_rel(main), sha256=_sha256(main))]
    with np.load(main) as z:
        raw = {k: z[k] for k in z.files}
    arrays = {k: raw.pop(k) for k in INPUTS if k in raw}
    for k in STATE_KEYS:
        if k not in arrays:
            raise KeyError(f"corpus {name}: {_rel(main)} has no {k}")
    meta, extra = {}, {}
    cq = cv = None
    if kind == "bench":
        cq, cv = raw.pop("cot_qpos", None), raw.pop("cot_qvel", None)
    else:
        if adj is not None:
            files.append(dict(path=_rel(adj), sha256=_sha256(adj)))
            with np.load(adj) as za:
                if all(f"cot{c}_qpos" in za.files for c in range(NCOT)):
                    cq = np.stack([za[f"cot{c}_qpos"] for c in range(NCOT)], 1)
                    cv = np.stack([za[f"cot{c}_qvel"] for c in range(NCOT)], 1)
    for k in ("metadata", "meta", "meta_json"):  # a JSON string (0-d array)
        if k in raw and raw[k].ndim == 0:
            text = str(raw.pop(k))
            try:
                v = json.loads(text)
            except ValueError:
                v = None
            if isinstance(v, dict):
                meta.update(v)
            else:
                meta[k] = text
    S = len(arrays["qpos"])
    for k, v in raw.items():
        if v.ndim == 0:
            meta[k] = v.item() if v.dtype.kind != "U" else str(v)
        elif v.shape[0] == S:
            extra[k] = v
        else:
            meta[k] = f"<array {list(v.shape)} {v.dtype}>"
    if cq is not None and (cq.shape[:2] != (S, NCOT) or cv is None or cv.shape[:2] != (S, NCOT)):
        raise ValueError(f"corpus {name}: cotangents must be (S, {NCOT}, nq) / (S, {NCOT}, nv), got "
                         f"{None if cq is None else cq.shape} / {None if cv is None else cv.shape}")
    return Corpus(name=name, kind=kind, files=files, arrays=arrays, cot_qpos=cq, cot_qvel=cv, meta=meta, extra=extra)


def load_corpus(name):
    """The Corpus of a name (module docstring), 'a+b' concatenated; a path to an .npz works as well."""
    parts = [_load_one(p) for p in name.split("+")]
    if len(parts) == 1:
        return parts[0]
    keys = set.intersection(*(set(c.arrays) for c in parts))
    arrays = {k: np.concatenate([c.arrays[k] for c in parts]) for k in INPUTS if k in keys}
    has_cot = all(c.cot_qpos is not None for c in parts)
    ekeys = set.intersection(*(set(c.extra) for c in parts))
    return Corpus(name=name, kind="+".join(sorted({c.kind for c in parts})), files=[f for c in parts for f in c.files],
                  arrays=arrays, cot_qpos=np.concatenate([c.cot_qpos for c in parts]) if has_cot else None,
                  cot_qvel=np.concatenate([c.cot_qvel for c in parts]) if has_cot else None,
                  meta={c.name: _jsonable(c.meta) for c in parts},
                  extra={k: np.concatenate([c.extra[k] for c in parts]) for k in ekeys})


@dataclass
class States:
    """The nworld states one unit runs: inputs of one control step and the stored output cotangents."""
    corpus: str
    subset: int
    nworld: int
    seed: int
    index: np.ndarray  # (nworld,) indices into the corpus
    tiled: float  # nworld / S when the corpus is tiled, else 1
    qpos: np.ndarray
    qvel: np.ndarray
    qacc_warmstart: np.ndarray
    ctrl: np.ndarray
    friction: np.ndarray = None  # (nworld,), None without DR inputs
    com: np.ndarray = None  # (nworld, 3)
    cot_qpos: np.ndarray = None  # (nworld, 3, nq)
    cot_qvel: np.ndarray = None  # (nworld, 3, nv)

    @property
    def nq(self):
        return self.qpos.shape[1]

    @property
    def nv(self):
        return self.qvel.shape[1]

    @property
    def nu(self):
        return self.ctrl.shape[1]

    def inputs(self, dtype=None):
        """{qpos, qvel, qacc_warmstart, ctrl[, friction, com]} (dtype None: as stored), C-contiguous."""
        out = {}
        for k in INPUTS:
            v = getattr(self, k)
            if v is not None:
                out[k] = np.ascontiguousarray(v if dtype is None else v.astype(dtype))
        return out

    def cotangent(self, c=0, dtype=None):
        """(cot_qpos (nworld, nq), cot_qvel (nworld, nv)) of the stored cotangent c (0..2)."""
        if self.cot_qpos is None:
            raise ValueError(f"corpus {self.corpus} has no stored cotangents")
        cq, cv = self.cot_qpos[:, c], self.cot_qvel[:, c]
        if dtype is not None:
            cq, cv = cq.astype(dtype), cv.astype(dtype)
        return np.ascontiguousarray(cq), np.ascontiguousarray(cv)

    def check(self, nq, nv, nu, dr):
        """ValueError when the arrays do not fit a model (nq, nv, nu) or DR inputs are missing."""
        for k, n in (("qpos", nq), ("qvel", nv), ("qacc_warmstart", nv), ("ctrl", nu)):
            v = getattr(self, k)
            if v.shape != (self.nworld, n):
                raise ValueError(f"states.{k} has shape {v.shape}, the model needs ({self.nworld}, {n})")
        if dr and (self.friction is None or self.com is None):
            raise ValueError(f"corpus {self.corpus} has no friction / com, but the problem randomizes them")

    def describe(self):
        return dict(corpus=self.corpus, subset=self.subset, nworld=self.nworld, seed=self.seed, tiled=self.tiled,
                    index_sha256=hashlib.sha256(np.ascontiguousarray(self.index, np.int64).tobytes()).hexdigest(),
                    cotangents=self.cot_qpos is not None)


def selection(n, nworld, subset=0, seed=SELECT_SEED):
    """Indices (nworld,) of the states of subset `subset` (module docstring)."""
    perm = np.random.default_rng(seed).permutation(n)
    if nworld <= n:
        k = n // nworld
        if not 0 <= subset < k:
            raise ValueError(f"subset {subset} out of range: {n} states hold {k} disjoint subsets of {nworld}")
        return perm[subset * nworld:(subset + 1) * nworld]
    if subset != 0:
        raise ValueError(f"nworld {nworld} > {n} states: the corpus is tiled, only subset 0 exists")
    return np.resize(perm, nworld)


def select(corpus, nworld, subset=0, seed=SELECT_SEED):
    """The States of a Corpus (or a corpus name) for nworld worlds (module docstring)."""
    if isinstance(corpus, str):
        corpus = load_corpus(corpus)
    idx = selection(corpus.n, nworld, subset, seed)
    a = corpus.arrays
    return States(corpus=corpus.name, subset=subset, nworld=nworld, seed=seed, index=idx,
                  tiled=max(1.0, nworld / corpus.n), qpos=a["qpos"][idx], qvel=a["qvel"][idx],
                  qacc_warmstart=a["qacc_warmstart"][idx], ctrl=a["ctrl"][idx],
                  friction=a["friction"][idx] if "friction" in a else None,
                  com=a["com"][idx] if "com" in a else None,
                  cot_qpos=corpus.cot_qpos[idx] if corpus.cot_qpos is not None else None,
                  cot_qvel=corpus.cot_qvel[idx] if corpus.cot_qvel is not None else None)
