"""The benchmark truth of a corpus: production (make), the truth file (benchmarks/data/truth/<corpus>.npz) and the
cached forward truth, as the accuracy scorers read them. MuJoCo C float64 only (truth/mjc.py); the model check
against native MuJoCo Warp 3.17.0 runs in its own environment (truth/mjw_labels.py, started as a subprocess).

Production per corpus (make):
  1. forward truth and C-only labels of every state, plus the it10 / it50 floor runs (mjc.label_task), cached in the
     cache directory (--cache-dir, default runs/bench/truth) as fwd_<corpus>.npz, keyed by the inputs digest, the
     model's sha256, the options, the MuJoCo version and VERSION;
  2. native MuJoCo Warp labels (mjw_<corpus>.npz, cached the same way) and the model check (mjc.model_check);
  3. on corpora with stored cotangents of Track A: the groups, the seeded sample (mjc.sample: up to 64 states per
     group) and the finite-difference truth of the sampled states (mjc.fd_task = fdlib.fd_job; cached in
     fd_<corpus>.npz and extended state by state);
  4. the truth file, written deterministically (fixed zip timestamps) and at most MAX_BYTES.

Truth file keys (S corpus states, N sampled states, Nj of them with the full Jacobian, ncol = 2 nv + nu = 99):
  meta                  0-d JSON string: format, corpus (name, kind, path, file sha256, inputs and cotangent digests),
                        problem (model path, sha256, substeps, DR), options of the truth and the floor runs, the FD
                        construction, label and group definitions with their counts, sampling, model check,
                        provenance (MuJoCo / numpy versions, harness commit, fdlib.py's sha256, CPU)
  labels (S,) uint16    mjc.LABEL_BITS of every state;  model_flags (S,) int8  mjc.MODEL_FLAGS, -1 = not checked
  frame_rotation (S,) float32  the model check's largest tangent-frame rotation (1 - |cos|; nan: not checked)
  niter, niter_<floor> (S,) int16  most solver iterations of the truth / floor runs (it10, it50, Track B native) in
                        any substep
  min_nu, min_abs_dist, min_kink, clamp_margin (S,) float32  the continuous labels (m, m, -, relative; inf: none)
  check_index (K,) int32, check_out (K, nq + nv) float64  forward truth (qpos' qpos form, qvel') of K = 64 fixed
                        states: a regenerated forward truth is checked against them
  gradient truth (Track A corpora with cotangents):
  index (N,) int32      the sampled states (sorted);  group_<G> (N,) bool  sampled for group G (mjc.GROUPS)
  fd_vjp (N, 3, ncol) float64        VJPs of the corpus's 3 stored cotangents through J(h = 1e-6): the truth
  fd_vjp_floor (N, 3, ncol) float64  the same through J(h = 1e-7): |fd_vjp_floor - fd_vjp| is the truth floor
  bad (N, ncol) uint8                column-validity bits (mjc.BAD); a column is valid when 0
  out (N, nq + nv) float64           the FD base run's outputs (= the forward truth of those states)
  jac_index (Nj,) int32              positions into index of the states with the full Jacobian (--jacobian-groups,
                                     default G0: the size budget)
  jacobian (Nj, nq + nv, ncol) float32   J(h = 1e-6), rows qpos' (qpos form) then qvel', columns qpos tangent, qvel,
                                     ctrl (the layout of the harness's op jac)
  jacobian_dcol (Nj, ncol) float32   per column |J(1e-6) - J(1e-7)| over all rows: the Jacobian's truth floor

  from benchmarks.bench.truth import store
  tr = store.load("mf_s0")            # Truth of benchmarks/data/truth/mf_s0.npz
  fw = tr.forward()                   # dict: out (S, nq + nv), out_<floor>, ... (cached or regenerated)
  tr.groups(), tr.model_ok(), tr.labels()
"""

import datetime
import io
import json
import os
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov
from benchmarks.bench.truth import common

FORMAT = "dk-bench-truth/1"
MAX_BYTES = 8 * 2 ** 20  # the size budget of one truth file (as a corpus file's)
NCHECK = 64
CHECK_SEED = 20261029
CHECK_RTOL = 1e-9  # a regenerated forward truth must match check_out to this (per state, relative to max(|x|, 1))
JACOBIAN_GROUPS = ("G0",)
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _mjc():
    from benchmarks.bench.truth import mjc  # mujoco + truth/fdlib.py: imported when needed
    return mjc


# ------------------------------------------------------------------------------------------------- deterministic npz

def write_npz(path, arrays, max_bytes=MAX_BYTES):
    """Write arrays (the meta as a 0-d string) as an npz zip with fixed timestamps (same arrays, same bytes for the
    same zlib), atomically; ValueError over max_bytes. Returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            for k in sorted(arrays):
                buf = io.BytesIO()
                v = np.asarray(arrays[k])
                np.lib.format.write_array(buf, v if v.flags.c_contiguous else v.copy(order="C"), allow_pickle=False)
                info = zipfile.ZipInfo(f"{k}.npy", date_time=_ZIP_TIME)
                info.external_attr = 0o644 << 16
                zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        size = tmp.stat().st_size
        if max_bytes and size > max_bytes:
            raise ValueError(f"{path.name}: {size} bytes, over the budget of {max_bytes}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return prov.sha256(path)


# ------------------------------------------------------------------------------------------------- cached steps

def _base_meta(corpus, prob, desc):
    import mujoco
    return dict(version=common.VERSION, corpus=corpus.name, inputs=common.inputs_digest(corpus.arrays),
                problem=prob.id, model_sha256=desc.get("sha256"), substeps=int(prob.substeps),
                mujoco=mujoco.__version__)


def forward_labels(corpus, prob, desc, cache, pool=None, jobs=None, recompute=False, log=print):
    """mjc.Pool.labels of every corpus state, cached as <cache>/fwd_<corpus>.npz."""
    mjc = _mjc()
    meta = dict(_base_meta(corpus, prob, desc), kind="forward_labels", truth=mjc.TRUTH, floors=mjc.floor_options(prob),
                near_margin=mjc.NEAR_MARGIN)
    path = cache / f"fwd_{corpus.name}.npz"
    got = None if recompute else common.load_cached(path, meta)
    if got is not None:
        return got, dict(path=prov.rel(path), cached=True)
    own = pool is None
    t0 = time.perf_counter()
    pool = pool or mjc.Pool(prob, jobs or len(os.sched_getaffinity(0)))
    try:
        lab = pool.labels(corpus.arrays)
    finally:
        if own:
            pool.close()
    dt = time.perf_counter() - t0
    common.save_cached(path, meta, lab)
    log(f"[truth] {corpus.name}: forward truth + labels of {len(lab['out'])} states, {dt:.1f} s on {pool.jobs} "
        f"processes")
    return lab, dict(path=prov.rel(path), cached=False, seconds=round(dt, 2), jobs=pool.jobs)


def mjw_python():
    """(interpreter, child environment) of the native MuJoCo Warp environment (envs.toml, engine mjw)."""
    from benchmarks.bench import protocol
    envs = protocol.Envs()
    env = envs.resolve("mjw")
    return env.python, envs.child_env(env)


def mjw_labels(corpus, prob, desc, cache, iterations=None, recompute=False, log=print):
    """Native MuJoCo Warp labels (truth/mjw_labels.py in the mjw-native environment), cached as
    <cache>/mjw_<corpus>.npz; (arrays, info), arrays None when the environment is missing (info says why)."""
    from benchmarks.bench.truth import mjw_labels as ML
    it = int(iterations or ML.ITERATIONS)
    path = cache / f"mjw_{corpus.name}.npz"
    want = dict(inputs=common.inputs_digest(corpus.arrays), iterations=it, problem=prob.id,
                substeps=int(prob.substeps))
    if path.exists() and not recompute:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if all(meta.get(k) == v for k, v in want.items()) and meta.get("versions", {}).get("mujoco-warp") \
                    == "3.17.0":
                return {k: z[k] for k in z.files if k != "meta"}, dict(path=prov.rel(path), cached=True, **meta)
    try:
        py, env = mjw_python()
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        return None, dict(error=f"no native MuJoCo Warp environment: {e}")
    cmd = [py, "-m", "benchmarks.bench.truth.mjw_labels", "--corpus", corpus.name, "--iterations", str(it),
           "--out", str(path)]
    if prob.model != P.get(prob.id).model:
        cmd += ["--model-path", str(prob.path())]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(P.REPO), env=env, capture_output=True, text=True)
    (cache / f"mjw_{corpus.name}.log").write_text(prov.scrub(r.stdout[-20000:] + "\n" + r.stderr[-20000:]))
    if r.returncode != 0:
        tail = prov.scrub(r.stderr.strip()[-2000:])
        raise RuntimeError(f"native MuJoCo Warp labels of {corpus.name} failed (exit {r.returncode}): {tail}")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arr = {k: z[k] for k in z.files if k != "meta"}
    if meta.get("versions", {}).get("mujoco-warp") != "3.17.0":
        raise RuntimeError(f"the mjw environment runs mujoco-warp {meta.get('versions')}, not 3.17.0")
    log(f"[truth] {corpus.name}: native MuJoCo Warp {meta['versions']['mujoco-warp']} labels, "
        f"{time.perf_counter() - t0:.1f} s")
    return arr, dict(path=prov.rel(path), cached=False, **meta)


def fd_truth(corpus, prob, desc, worlds, cache, pool, recompute=False, log=print):
    """mjc.Pool.fd of the states `worlds`, cached (and extended) in <cache>/fd_<corpus>.npz: dict of FD_KEYS arrays
    in the order of worlds."""
    mjc = _mjc()
    meta = dict(_base_meta(corpus, prob, desc), kind="fd", truth=mjc.TRUTH, steps=list(mjc.FD_STEPS),
                step_rtol=mjc.FD_STEP_RTOL, step_atol=mjc.FD_STEP_ATOL)
    path = cache / f"fd_{corpus.name}.npz"
    old = None if recompute else common.load_cached(path, meta)
    have = {} if old is None else {int(w): i for i, w in enumerate(old["worlds"])}
    miss = [int(w) for w in worlds if int(w) not in have]
    rows = {}
    t0 = time.perf_counter()
    if miss:
        rows = pool.fd(corpus.arrays, miss)
        log(f"[truth] {corpus.name}: FD truth of {len(miss)} states, {time.perf_counter() - t0:.1f} s on {pool.jobs} "
            f"processes ({sum(r['seconds'] for r in rows.values()):.0f} CPU s)")
    allw = sorted(set(have) | set(rows))
    get = lambda w, k: rows[w][k] if w in rows else old[k][have[w]]  # noqa: E731
    merged = {k: np.stack([np.asarray(get(w, k)) for w in allw]) for k in mjc.FD_KEYS}
    if rows:
        common.save_cached(path, meta, dict(worlds=np.asarray(allw, np.int64), **merged))
    pos = {w: i for i, w in enumerate(allw)}
    sel = [pos[int(w)] for w in worlds]
    return {k: v[sel] for k, v in merged.items()}, dict(path=prov.rel(path), computed=len(miss),
                                                        seconds=round(time.perf_counter() - t0, 2))


# ------------------------------------------------------------------------------------------------- make

def check_states(S):
    return np.sort(np.random.default_rng(CHECK_SEED).permutation(S)[:min(NCHECK, S)]).astype(np.int32)


def _counts(mask_dict):
    return {k: int(np.asarray(v).sum()) for k, v in mask_dict.items()}


def _file_provenance():
    """What the truth's numbers depend on (stored in the file, so a regenerated file is bitwise the same)."""
    import mujoco
    return dict(mujoco=mujoco.__version__, numpy=np.__version__,
                fd_helpers=dict(path=prov.rel(_mjc().FDLIB_PATH), sha256=prov.sha256(_mjc().FDLIB_PATH)))


def run_provenance():
    """Where and when a truth was made (the index file truth.json, not the truth file): date, harness commit, Python,
    CPU model and the CPUs this process may use."""
    cpu = prov.cpu() or {}
    return dict(date=datetime.date.today().isoformat(), harness=prov.harness_commit(),
                python=platform.python_version(), cpu=cpu.get("model"), affinity=cpu.get("affinity"))


def make(name, cache=None, jobs=None, pool_cache=None, gradient=True, mjw=True, jacobian_groups=JACOBIAN_GROUPS,
         sample_cap=None, seed=None, recompute=False, out=None, force=False, model_path=None, log=print):
    """Produce the truth of corpus `name` and write its truth file (module docstring); returns a summary dict.
    pool_cache: {problem id: mjc.Pool} reused across corpora (created here when missing)."""
    mjc = _mjc()
    t_start = time.perf_counter()
    cache = common.cache_dir(cache)
    corpus = P.load_corpus(name)
    prob = common.problem_of(corpus, model_path)
    desc = common.check_model(corpus, prob)
    out = Path(out) if out else common.truth_path(corpus.name)
    if out.exists() and prov.is_committed(out) and not force:
        raise FileExistsError(f"{prov.rel(out)} is committed: pass --force to replace it")
    jobs = int(jobs or len(os.sched_getaffinity(0)))
    pools = pool_cache if pool_cache is not None else {}
    if prob.id not in pools:
        pools[prob.id] = mjc.Pool(prob, jobs)
    pool = pools[prob.id]
    try:
        lab, fwd_info = forward_labels(corpus, prob, desc, cache, pool, recompute=recompute, log=log)
        S = len(lab["out"])
        bits = mjc.label_bits(lab)
        if mjw:
            wl, mjw_info = mjw_labels(corpus, prob, desc, cache, recompute=recompute, log=log)
        else:
            wl, mjw_info = None, dict(error="skipped (--no-mjw)")
        if wl is not None:
            flags, rot = mjc.model_check(lab, wl)
            flags = flags.astype(np.int8)
        else:
            flags, rot = np.full(S, -1, np.int8), np.full(S, np.nan)
        masks = mjc.group_masks(bits)
        arrays = dict(labels=bits, model_flags=flags, frame_rotation=rot.astype(np.float32),
                      niter=lab["niter"].astype(np.int16),
                      **{f"niter_{k}": lab[f"niter_{k}"].astype(np.int16) for k in mjc.floor_options(prob)},
                      **{k: lab[k].astype(np.float32) for k in mjc.CONT_LABELS})
        ci = check_states(S)
        arrays.update(check_index=ci, check_out=lab["out"][ci])
        cot = corpus.cot_qpos is not None
        grad = gradient and cot and prob.track == "A"
        meta_grad, fd_info = None, None
        if grad:
            cap = mjc.SAMPLE_CAP if sample_cap is None else int(sample_cap)
            sd = mjc.SAMPLE_SEED if seed is None else int(seed)
            pick, worlds = mjc.sample(masks, corpus.name, cap, sd)
            fd, fd_info = fd_truth(corpus, prob, desc, worlds, cache, pool, recompute=recompute, log=log)
            if not np.array_equal(fd["out"], lab["out"][worlds]):
                raise RuntimeError(f"{corpus.name}: the FD base runs differ from the forward truth (bitwise)")
            cq, cv = corpus.cot_qpos[worlds], corpus.cot_qvel[worlds]
            J = fd["J"]
            fdv, fdv2 = mjc.vjp(J[:, 0], cq, cv), mjc.vjp(J[:, -1], cq, cv)
            jsel = np.zeros(len(worlds), bool)
            for g in jacobian_groups:
                jsel |= np.isin(worlds, pick[g])
            jix = np.flatnonzero(jsel).astype(np.int32)
            arrays.update(index=worlds.astype(np.int32), fd_vjp=fdv, fd_vjp_floor=fdv2, bad=fd["bad"].astype(np.uint8),
                          out=fd["out"], jac_index=jix, jacobian=J[jix, 0].astype(np.float32),
                          jacobian_dcol=np.linalg.norm(J[jix, 0] - J[jix, -1], axis=1).astype(np.float32),
                          **{f"group_{g}": np.isin(worlds, pick[g]) for g in mjc.GROUPS})
            ok = flags[worlds] == 0
            nv, nu = corpus.arrays["qvel"].shape[1], corpus.arrays["ctrl"].shape[1]
            meta_grad = dict(
                states=int(len(worlds)), sampled={g: int(len(pick[g])) for g in mjc.GROUPS},
                sampled_model_ok={g: int((np.isin(worlds, pick[g]) & ok).sum()) for g in mjc.GROUPS}
                if wl is not None else None,
                cap=cap, seed=sd, scheme="per group: numpy.random.default_rng([seed, crc32('corpus'), crc32(corpus), "
                                         "crc32(group)]).permutation(S), the first `cap` states of the group",
                fd=dict(construction="central differences of the whole control step, one column of (qpos tangent, "
                                     "qvel, ctrl) at a time, qpos perturbed with mj_integratePos "
                                     "(benchmarks/bench/truth/fdlib.py fd_job)",
                        steps=list(mjc.FD_STEPS), truth_step=mjc.FD_STEPS[0], floor_step=mjc.FD_STEPS[-1],
                        step_rtol=mjc.FD_STEP_RTOL, step_atol=mjc.FD_STEP_ATOL, bits=mjc.BAD,
                        invalid_share=float(np.mean(fd["bad"] != 0)),
                        by_reason={r: float(np.mean((fd["bad"] & b) != 0)) for r, b in mjc.BAD.items()},
                        truth_nonconv_states=int((fd["niter"] >= mjc.TRUTH["iterations"]).sum())),
                cotangents=common.cot_digest(corpus.cot_qpos, corpus.cot_qvel), ncot=int(corpus.cot_qpos.shape[1]),
                jacobian=dict(groups=list(jacobian_groups), states=int(len(jix)), dtype="float32"),
                blocks=dict(qpos=[0, nv], qvel=[nv, 2 * nv], ctrl=[2 * nv, 2 * nv + nu]))
        f0 = corpus.files[0]
        meta = dict(
            format=FORMAT,
            corpus=dict(name=corpus.name, kind=corpus.kind, n=S, path=f0["path"], sha256=f0["sha256"],
                        files=corpus.files, inputs=common.inputs_digest(corpus.arrays),
                        cotangents=common.cot_digest(corpus.cot_qpos, corpus.cot_qvel) if cot else None,
                        public=corpus.kind == "bench"),
            problem={k: desc.get(k) for k in ("id", "track", "label", "path", "sha256", "dir_sha256", "substeps",
                                              "timestep", "dr", "foot_pairs", "torso_body", "source")},
            options=dict(truth=mjc.TRUTH, floors=mjc.floor_options(prob), sensors=False,
                         note="MuJoCo C float64 (mj_step), ctrl held over the substeps, qfrc_applied 0, the corpus "
                              "inputs (float32) read exactly as float64, per-world DR without mj_setConst"),
            labels=dict(bits=mjc.LABEL_BITS,
                        counts={k: int(((bits & b) != 0).sum()) for k, b in mjc.LABEL_BITS.items()},
                        thresholds=dict(axis_5mm=mjc.AXIS_5MM, axis_1mm=mjc.AXIS_1MM, near_margin=mjc.NEAR_MARGIN,
                                        flip_dist=mjc.FLIP_DIST, kink_rel=mjc.KINK_REL, basis_band=mjc.BASIS_BAND,
                                        g9_run=mjc.G9_RUN),
                        definitions="benchmarks/bench/truth/mjc.py module docstring"),
            groups=dict(definitions=mjc.GROUP_DEFS, population=_counts(masks),
                        population_model_ok=_counts({g: m & (flags == 0) for g, m in masks.items()})
                        if wl is not None else None),
            model_check=dict(available=wl is not None, flags=mjc.MODEL_FLAGS, frame_rotation=mjc.FRAME_ROT,
                             counts={k: int(((flags & b) != 0).sum()) for k, b in mjc.MODEL_FLAGS.items()}
                             if wl is not None else None,
                             model_ok=int((flags == 0).sum()) if wl is not None else None,
                             reference={k: mjw_info.get(k) for k in ("engine", "versions", "iterations",
                                                                     "ls_iterations", "tolerance_effective",
                                                                     "capacity", "warn_overflow", "error")
                                        if k in mjw_info}),
            check=dict(states=int(len(ci)), seed=CHECK_SEED, rtol=CHECK_RTOL),
            gradient=meta_grad, provenance=_file_provenance())
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True, separators=(",", ":")))
        sha = write_npz(out, arrays)
        size = out.stat().st_size
        summary = dict(corpus=corpus.name, kind=corpus.kind, file=prov.rel(out), bytes=size, sha256=sha,
                       labels=meta["labels"]["counts"], groups=meta["groups"]["population"],
                       model_check=meta["model_check"]["counts"], model_ok=meta["model_check"]["model_ok"],
                       gradient=None if meta_grad is None else dict(states=meta_grad["states"],
                                                                    sampled=meta_grad["sampled"],
                                                                    sampled_model_ok=meta_grad["sampled_model_ok"],
                                                                    invalid_share=meta_grad["fd"]["invalid_share"],
                                                                    by_reason=meta_grad["fd"]["by_reason"]),
                       forward=fwd_info, mjw={k: v for k, v in mjw_info.items() if k in ("path", "cached", "error",
                                                                                         "seconds")},
                       fd=fd_info, seconds=round(time.perf_counter() - t_start, 1), provenance=run_provenance())
        return summary
    finally:
        if pool_cache is None:
            for p in pools.values():
                p.close()


# ------------------------------------------------------------------------------------------------- reading

class Truth:
    """A truth file (module docstring) with lazy access to its corpus and forward truth."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files if k != "meta"}
            self.meta = json.loads(str(z["meta"]))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{prov.rel(self.path)}: not a {FORMAT} file")
        self.name = self.meta["corpus"]["name"]
        self._corpus = None
        self._forward = None

    # -- corpus and problem
    def corpus(self, check=True):
        """The problem.Corpus, checked against the truth's inputs and cotangent digests."""
        if self._corpus is None:
            c = P.load_corpus(self.name)
            if check:
                if common.inputs_digest(c.arrays) != self.meta["corpus"]["inputs"]:
                    raise ValueError(f"corpus {self.name}: its inputs are not the truth's ({prov.rel(self.path)})")
                want = self.meta["corpus"].get("cotangents")
                if want and common.cot_digest(c.cot_qpos, c.cot_qvel) != want:
                    raise ValueError(f"corpus {self.name}: its stored cotangents are not the truth's")
            self._corpus = c
        return self._corpus

    def problem(self, model_path=None):
        return common.problem_of(self.corpus(), model_path)

    @property
    def n(self):
        return int(self.meta["corpus"]["n"])

    # -- labels
    def labels(self):
        return _mjc().unpack_labels(self.arrays["labels"]) if "labels" in self.arrays else {}

    def label_bits(self):
        return self.arrays["labels"]

    def groups(self):
        """{group: bool (S,)} over all corpus states (mjc.GROUP_DEFS)."""
        return _mjc().group_masks(self.arrays["labels"])

    def model_flags(self):
        return self.arrays["model_flags"]

    def model_ok(self):
        """(S,) bool, or None when the truth has no model check."""
        f = self.arrays["model_flags"]
        return None if np.all(f < 0) else f == 0

    # -- gradient truth
    @property
    def has_gradient(self):
        return "fd_vjp" in self.arrays

    def gradient(self):
        """dict(index, fd_vjp, fd_vjp_floor, bad, out, groups {g: (N,) bool}, model_ok (N,) or None, jac_index,
        jacobian, jacobian_dcol, blocks {name: slice}, labels (N,) bits)."""
        if not self.has_gradient:
            raise ValueError(f"{prov.rel(self.path)} has no gradient truth")
        a = self.arrays
        idx = a["index"].astype(np.int64)
        mo = self.model_ok()
        blocks = {k: slice(*v) for k, v in self.meta["gradient"]["blocks"].items()}
        return dict(index=idx, fd_vjp=a["fd_vjp"], fd_vjp_floor=a["fd_vjp_floor"], bad=a["bad"], out=a["out"],
                    groups={g: a[f"group_{g}"] for g in _mjc().GROUPS if f"group_{g}" in a},
                    model_ok=None if mo is None else mo[idx], jac_index=a["jac_index"].astype(np.int64),
                    jacobian=a["jacobian"], jacobian_dcol=a["jacobian_dcol"], blocks=blocks,
                    labels=a["labels"][idx])

    # -- forward truth
    def forward(self, cache=None, jobs=None, recompute=False, log=print):
        """The forward truth of every state and the floor runs: dict out, warm, out_it10, out_it50, niter, ... (cached
        in the cache directory, regenerated with MuJoCo C when missing) after checking it against check_out."""
        if self._forward is None:
            c = self.corpus()
            prob = self.problem()
            desc = common.check_model(c, prob)
            if desc.get("sha256") != self.meta["problem"].get("sha256"):
                raise ValueError(f"model {desc['path']} sha256 {desc.get('sha256')} is not the truth's "
                                 f"{self.meta['problem'].get('sha256')}")
            lab, info = forward_labels(c, prob, desc, common.cache_dir(cache), jobs=jobs, recompute=recompute,
                                       log=log)
            ci, co = self.arrays["check_index"].astype(np.int64), self.arrays["check_out"]
            err = np.linalg.norm(lab["out"][ci] - co, axis=1) / np.maximum(np.linalg.norm(co, axis=1), 1.0)
            if not np.all(err <= CHECK_RTOL):
                raise ValueError(f"{self.name}: the regenerated forward truth differs from the truth file's check "
                                 f"values (max {err.max():.3g} > {CHECK_RTOL})")
            info["check_max"] = float(err.max())
            info["check_bitwise"] = bool(np.array_equal(lab["out"][ci], co))
            lab["_info"] = info
            self._forward = lab
        return self._forward


def summary_of(tr):
    """make()'s summary fields that the truth file itself records (its meta, sha256 and size): for truth.json."""
    m = tr.meta
    g = m.get("gradient")
    return dict(corpus=tr.name, kind=m["corpus"].get("kind"), file=prov.rel(tr.path), bytes=tr.path.stat().st_size,
                sha256=prov.sha256(tr.path), labels=m["labels"]["counts"], groups=m["groups"]["population"],
                model_check=m["model_check"]["counts"], model_ok=m["model_check"]["model_ok"],
                gradient=None if g is None else dict(states=g["states"], sampled=g["sampled"],
                                                     sampled_model_ok=g["sampled_model_ok"],
                                                     invalid_share=g["fd"]["invalid_share"],
                                                     by_reason=g["fd"]["by_reason"]))


def load(name_or_path):
    """The Truth of a corpus name (<data dir>/truth/<name>.npz) or of a truth file path."""
    p = Path(name_or_path)
    if not (str(name_or_path).endswith(".npz") or "/" in str(name_or_path)):
        p = common.truth_path(str(name_or_path))
    elif not p.is_absolute():
        p = P.REPO / p
    if not p.exists():
        raise FileNotFoundError(f"no truth file {prov.rel(p)} (python -m benchmarks.bench truth make --corpora "
                                f"{name_or_path})")
    return Truth(p)
