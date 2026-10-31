"""MuJoCo C float64 machinery of the benchmark truth: the per-state forward truth with MuJoCo C-only state labels, the
solver-truncation floors, the finite-difference gradient truth, the C-only groups, the model check against native
MuJoCo Warp and the fixed seeded sampling. CPU only; numpy, mujoco and truth/fdlib.py, whose MuJoCo C helpers define
the semantics reused here: _set_dr (per-world DR), _contacts / _limits / _zones (per-substep contact codes, tangent
frames, zones with efc_force > 0) and fd_job (the central differences of one control step and their column-validity
bits). Nothing here imports MuJoCo Warp or NablaEngine.

Options of the MuJoCo C runs (on the problem's model, benchmarks/bench/problem.py: sensors off, ctrl held over the
problem's substeps, qfrc_applied 0, per-world foot-floor friction and torso COM offset on Track A):
  truth   Newton 2000 iterations, tolerance 1e-13, ls_iterations 100
  it50    50 iterations, tolerance 1e-8, ls 8: the reference iteration count; the C-only G9 label and a floor row
  it10    10 iterations, tolerance 1e-8, ls 8: the headline iteration count; a floor row (pure solver truncation)
  native  Track B only (a problem whose iterations are not set explicitly): the model's own options (Menagerie
          scene_mjx.xml: 5 iterations, tolerance 1e-8, ls 8); a floor row of the native Track B rows

State labels (MuJoCo C alone, the truth run unless stated; any of the control step's substeps; LABEL_BITS):
  G1_axis_lt_5mm   a self contact between two sphere / capsule geoms with nu = dist + r1 + r2 (the centre or axis
                   distance before the radii) below 5 mm; G1_axis_lt_1mm below 1 mm (report only: the contact
                   normal's derivative grows like 1 / nu)
  G2_contact_flip  min |dist| < 1e-6 m over the substep's contacts and over every pair within 1e-5 m of touching
                   (MuJoCo C collision of the substep's input state with pair and geom margins 1e-5): a contact that
                   appears or disappears within rounding of the state (the float64 half of
                   nablaengine.reference.adjoint_groups' G2; its MuJoCo Warp reference-set half has no C analogue)
  G3_kink          a contact or limit row with |J a* - aref| < 1e-4 (|J a*| + |aref|), i.e. near its active / inactive
                   boundary (a* = the solver's qacc; J a* by mj_mulJacVec); a friction-loss row with
                   | |J a* - aref| - R f | < 1e-4 R f (its quadratic / linear boundary; R = efc_R, f = frictionloss)
  G4_clamped       an actuator force (actuator_force) outside the jnt_actfrcrange of its joint (the clamp mask of
                   fdlib's validity bit), or a force-limited actuator at its forcerange
  G5_limits        a joint- or tendon-limit row present
  G6_self_contact  a contact between two geoms whose bodies are not welded to the world
  G7_basis_switch  a self contact whose normal has ||n_y| - 0.5| < 0.02: mju_makeFrame's choice of the default
                   tangent axis switches at |n_y| = 0.5 (MuJoCo C and MuJoCo Warp share it). MuJoCo Warp's
                   plane-capsule fallback (|capsule axis projected on the plane| < 0.5) is a MuJoCo Warp convention,
                   not a discontinuity of MuJoCo C's dynamics, so it is not a C-only label; those states are caught
                   by the model check's frame bit instead
  G9_nonconv       the it50 run reaches its 50 iterations in some substep and island (MuJoCo Warp's labels,
                   nablaengine.reference.adjoint_groups, count the same at 50 iterations)
  G0_normal        none of G1_axis_lt_5mm, G2, G3, G7, G9
  truth_nonconv    the truth run reaches 2000 iterations (every gradient column then carries the solver bit)
  nonfinite        the truth run's output is not finite, or MuJoCo C warned of a bad qpos / qvel / qacc
Groups (GROUPS, of the gradient truth and the per-group forward subsets, defined on these labels): G0 = G0_normal & ~G4_clamped, G4 = G0_normal & G4_clamped, G5 = G0_normal & G5_limits &
~G4_clamped, G6 = G0_normal & G6_self_contact & ~G4_clamped, G1 = G1_axis_lt_5mm (report only).

Model check (model_check, for any substep count): the truth
run's per-substep contacts (code = pair * 4 + plane-capsule end), their zones (edge by edge, active = efc_force > 0),
the joint-limit rows and their zones, and the tangent frames of condim-3 contacts whose edges are in mixed zones,
against native MuJoCo Warp 3.17.0's forward (truth/mjw_labels.py): flags model_contacts 1, model_frame 2 (1 - |cos|
of t1 against MuJoCo Warp's t1 or t2 above 1e-4), model_zones 4; 0 = model-ok.

Gradient truth (fd_task = fdlib.fd_job, Track A's 4 substeps): for each column j of (qpos tangent (nv,
perturbed with mj_integratePos, so the quaternion stays normalised), qvel (nv), ctrl (nu)) one control step at x +- h
e_j for h = 1e-6 and 1e-7; J (2, nq + nv, 2 nv + nu) with qpos' in qpos form; column bits (BAD) step (the two steps
differ by more than 1e-4 relative + 1e-6 absolute on the qpos' or qvel' rows), contacts (a substep's contact set
differs), rows (rows (type, id) or zones differ, active = efc_force > 0), clamp (the actuator clamp mask differs),
ctrlrange (ctrl +- h leaves the ctrlrange), solver (a run hit the iteration limit; fd_job reads island 0's
solver_niter, the truth run's maximum over the islands is in the labels).

Sampling (sample): per group a fixed random order of all states, numpy.random.default_rng([seed, crc32("corpus"),
crc32(corpus), crc32(group)]).permutation(S), and the
first `cap` states of the group in that order; every state of the group is eligible (the model-check flags are kept
with the truth, so scores are given on all sampled states and on the model-ok ones).

  from benchmarks.bench.truth import mjc
  pool = mjc.Pool(problem, jobs=24)
  lab = pool.labels(arrays)               # dict of per-state arrays (out, labels, con_*, out_it10, ...)
  fd = pool.fd(arrays, worlds)            # {world: {J, bad, out, niter, ...}}
  pool.close()
"""

import copy
import multiprocessing as mp
import os
import zlib
from pathlib import Path

import mujoco
import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench.truth import common
from benchmarks.bench.truth import fdlib

REPO = Path(__file__).resolve().parents[3]
FDLIB_PATH = Path(fdlib.__file__).resolve()  # the FD helpers (their sha256 goes into the truth's provenance)

TRUTH = dict(iterations=2000, tolerance=1e-13, ls_iterations=100)
FLOORS = dict(it10=dict(iterations=10, tolerance=1e-8, ls_iterations=8),
              it50=dict(iterations=50, tolerance=1e-8, ls_iterations=8))
NATIVE_FLOOR = "native"  # Track B only: the model's own solver options (the trackB part's native rows)
G9_RUN = "it50"
FD_STEPS = (1e-6, 1e-7)  # the first gives the truth, the second its floor
FD_STEP_RTOL, FD_STEP_ATOL = 1e-4, 1e-6  # fd_job's step_rtol / step_atol
MAXC = common.MAXC  # contact slots per substep in the packed arrays (fdlib.MAXC, checked below)
NEAR_MARGIN = 1e-5  # m: collision margin of the G2 near-contact search
FLIP_DIST = 1e-6  # m: G2 threshold on |dist|
KINK_REL = 1e-4  # G3 threshold
AXIS_5MM, AXIS_1MM = 5e-3, 1e-3  # m: G1
BASIS_BAND = 0.02  # G7: ||n_y| - 0.5| below this
FRAME_ROT = 1e-4  # model check: 1 - |cos| of the tangent frames
SAMPLE_CAP, SAMPLE_SEED, SAMPLE_SET = 64, 0, "corpus"
CHUNK = 32  # states per pool task

LABEL_BITS = dict(G1_axis_lt_5mm=1, G1_axis_lt_1mm=2, G2_contact_flip=4, G3_kink=8, G4_clamped=16, G5_limits=32,
                  G6_self_contact=64, G7_basis_switch=128, G9_nonconv=256, G0_normal=512, truth_nonconv=1024,
                  nonfinite=2048)
OUTSIDE_G0 = ("G1_axis_lt_5mm", "G2_contact_flip", "G3_kink", "G7_basis_switch", "G9_nonconv")
GROUPS = ("G0", "G4", "G5", "G6", "G1")
GROUP_DEFS = dict(G0="G0_normal & ~G4_clamped", G4="G0_normal & G4_clamped", G5="G0_normal & G5_limits & ~G4_clamped",
                  G6="G0_normal & G6_self_contact & ~G4_clamped", G1="G1_axis_lt_5mm (report only)")
MODEL_FLAGS = dict(model_contacts=1, model_frame=2, model_zones=4)
CONT_LABELS = ("min_nu", "min_abs_dist", "min_kink", "clamp_margin")
PACK_KEYS = ("con_code", "con_t", "con_dim", "con_zone", "lim_rows", "lim_act")
FD_KEYS = ("J", "bad", "out", "niter", "ncon", "seconds")


assert fdlib.MAXC == MAXC, f"fdlib.MAXC {fdlib.MAXC} != {MAXC}"
BAD = dict(fdlib.BAD)  # column-validity bits: step 1, contacts 2, rows 4, clamp 8, ctrlrange 16, solver 32
C_EQ = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
C_FRIC = (int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF), int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON))
C_LIMIT = (int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT), int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON))
BAD_WARNINGS = (int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL),
                int(mujoco.mjtWarning.mjWARN_BADQACC))


# ------------------------------------------------------------------------------------------------- model setup

def set_options(m, opts):
    m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = (int(opts["iterations"]), float(opts["tolerance"]),
                                                               int(opts["ls_iterations"]))
    return m


def truth_model(prob, opts=TRUTH):
    """The problem's MjModel (problem.load_model: sensors off, Track A's ls / tolerance) with the options opts."""
    return set_options(P.load_model(prob, iterations=int(opts["iterations"])), opts)


def floor_options(prob):
    """{name: options} of the floor runs of a problem: FLOORS, plus NATIVE_FLOOR (the model's own) on Track B."""
    out = {k: dict(v) for k, v in FLOORS.items()}
    if not prob.explicit_iterations:
        m = P.load_model(prob, iterations=None)
        out[NATIVE_FLOOR] = dict(iterations=int(m.opt.iterations), tolerance=float(m.opt.tolerance),
                                 ls_iterations=int(m.opt.ls_iterations))
    return out


def fd_tables(m, prob):
    """fdlib's per-worker lookup tables (as fdlib._worker_init) for model m of problem prob; Track B (no DR): no foot
    pairs and torso 0 with a zero COM offset, so _set_dr is the identity."""
    if prob.dr:
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prob.torso_body)
        foot = np.asarray(prob.foot_pairs, int)
    else:
        torso, foot = 0, np.zeros(0, int)
    pair_of = np.full((m.ngeom, m.ngeom), -1, np.int64)
    for i in range(m.npair):
        pair_of[m.pair_geom1[i], m.pair_geom2[i]] = pair_of[m.pair_geom2[i], m.pair_geom1[i]] = i
    jid = m.actuator_trnid[:, 0]
    lim = m.jnt_actfrclimited[jid].astype(bool)
    return dict(m=m, d=mujoco.MjData(m), torso=torso, ipos0=m.body_ipos[torso].copy(), fric0=m.pair_friction.copy(),
                foot=foot, pair_of=pair_of, plane=m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE,
                capsule=m.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE,
                lo=np.where(lim, m.jnt_actfrcrange[jid, 0], -np.inf),
                hi=np.where(lim, m.jnt_actfrcrange[jid, 1], np.inf),
                ctrl_lo=np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 0], -np.inf),
                ctrl_hi=np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 1], np.inf),
                act_dof=m.jnt_dofadr[jid], act_kv=fdlib.kv0_damping(m))


# ------------------------------------------------------------------------------------------------- worker side

_W = {}  # per worker process: the problem, models and data of every run, lookup tables


def init_worker(prob_dict, nsub):
    """Pool initializer: the truth model in fdlib's _C (for fd_job and its helpers), the floor models, the
    near-contact model and the label tables."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    prob = P.Problem.from_dict(prob_dict)
    m = truth_model(prob)
    fdlib._C.clear()
    fdlib._C.update(fd_tables(m, prob))
    floors = {}
    for name, opts in floor_options(prob).items():
        mf = set_options(copy.copy(m), opts)
        floors[name] = (mf, mujoco.MjData(mf))
    mm = copy.copy(m)
    mm.pair_margin[:] = NEAR_MARGIN
    mm.geom_margin[:] = NEAR_MARGIN
    robot = m.body_weldid[m.geom_bodyid] != 0
    sphcap = np.isin(m.geom_type, (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)))
    flim = np.flatnonzero(m.actuator_forcelimited.astype(bool))
    _W.clear()
    _W.update(prob=prob, nsub=int(nsub), floors=floors, near=(mm, mujoco.MjData(mm)), robot=robot, sphcap=sphcap,
              size0=m.geom_size[:, 0].copy(), flim=flim, flo=m.actuator_forcerange[flim, 0].copy(),
              fhi=m.actuator_forcerange[flim, 1].copy())


def _apply_dr(m, fric, com):
    """fdlib._set_dr on another model copy (the floor runs)."""
    C = fdlib._C
    m.body_ipos[C["torso"]] = C["ipos0"] + com
    m.pair_friction[:] = C["fric0"]
    m.pair_friction[C["foot"], :2] = fric


def _load(d, q, v, u, warm):
    """The input state as fdlib._run sets it."""
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, warm
    d.time = 0.0
    d.qfrc_applied[:] = 0.0


def _niter(d):
    return int(np.max(d.solver_niter[:max(int(d.nisland), 1)]))


def _near_distance(qin):
    """min |dist| over the pairs within NEAR_MARGIN of touching at qpos qin (inf if none)."""
    mm, dm = _W["near"]
    dm.qpos[:] = qin
    mujoco.mj_kinematics(mm, dm)
    mujoco.mj_collision(mm, dm)
    n = dm.ncon
    return float(np.min(np.abs(dm.contact.dist[:n]))) if n else np.inf


def _substep_labels(m, d, lab):
    """Update the label accumulators lab with the substep just stepped (contacts, rows, actuators)."""
    C = fdlib._C
    n = d.ncon
    if n:
        con = d.contact
        g1, g2 = np.asarray(con.geom1[:n]), np.asarray(con.geom2[:n])
        dist = np.asarray(con.dist[:n])
        lab["min_abs_dist"] = min(lab["min_abs_dist"], float(np.min(np.abs(dist))))
        selfc = _W["robot"][g1] & _W["robot"][g2]
        if selfc.any():
            lab["G6_self_contact"] = True
            ny = np.asarray(con.frame[:n, 1])[selfc]
            if np.any(np.abs(np.abs(ny) - 0.5) < BASIS_BAND):
                lab["G7_basis_switch"] = True
            sc = selfc & _W["sphcap"][g1] & _W["sphcap"][g2]
            if sc.any():
                nu = dist[sc] + _W["size0"][g1[sc]] + _W["size0"][g2[sc]]
                lab["min_nu"] = min(lab["min_nu"], float(nu.min()))
    ne = d.nefc
    if ne:
        ja = np.zeros(ne)
        mujoco.mj_mulJacVec(m, d, ja, d.qacc)
        aref = d.efc_aref[:ne]
        jar = ja - aref
        typ = d.efc_type[:ne]
        fr = np.isin(typ, C_FRIC)
        ineq = ~fr & (typ != C_EQ)
        meas = []
        if ineq.any():
            meas.append(np.abs(jar[ineq]) / np.maximum(np.abs(ja[ineq]) + np.abs(aref[ineq]), 1e-30))
        if fr.any():
            rf = d.efc_R[:ne][fr] * d.efc_frictionloss[:ne][fr]
            meas.append(np.abs(np.abs(jar[fr]) - rf) / np.maximum(rf, 1e-30))
        if meas:
            lab["min_kink"] = min(lab["min_kink"], float(np.min(np.concatenate(meas))))
        if np.isin(typ, C_LIMIT).any():
            lab["G5_limits"] = True
    raw = d.actuator_force
    lo, hi = C["lo"], C["hi"]
    if ((raw < lo) | (raw > hi)).any():
        lab["G4_clamped"] = True
    if len(_W["flim"]):
        f = raw[_W["flim"]]
        if np.any((f <= _W["flo"]) | (f >= _W["fhi"])):
            lab["G4_clamped"] = True
    fin = np.isfinite(lo) & np.isfinite(hi)
    if fin.any():
        bound = np.maximum(np.abs(lo[fin]), np.abs(hi[fin]))
        mg = np.minimum(np.abs(raw[fin] - lo[fin]), np.abs(raw[fin] - hi[fin])) / np.maximum(bound, 1e-30)
        lab["clamp_margin"] = min(lab["clamp_margin"], float(mg.min()))


def pack(con, nsub):
    """Per-substep fdlib._contacts(full) + (_limits,) -> fixed-size arrays (fdlib._pack for any substep count): con_code (nsub, MAXC) (-1 = none), con_t (nsub, MAXC, 6), con_dim, con_zone, lim_rows, lim_act."""
    code = np.full((nsub, MAXC), -1, np.int64)
    tang = np.zeros((nsub, MAXC, 6))
    dim = np.zeros((nsub, MAXC), np.int32)
    zone = np.zeros((nsub, MAXC), np.int32)
    lim = np.zeros((nsub, 2), np.int64)
    for k, (c, t, dm, zn, lm) in enumerate(con):
        n = min(len(c), MAXC)
        code[k, :n], tang[k, :n], dim[k, :n], zone[k, :n] = c[:n], t[:n], dm[:n], zn[:n]
        lim[k] = lm
    return dict(con_code=code, con_t=tang, con_dim=dim, con_zone=zone, lim_rows=lim[:, 0], lim_act=lim[:, 1])


def _truth_state(q, v, warm, u, fric, com):
    """The truth run of one state with its labels and contact signature."""
    C = fdlib._C
    m, d = C["m"], C["d"]
    nsub = _W["nsub"]
    fdlib._set_dr(C, fric, com)
    _load(d, q, v, u, warm)
    lab = dict(min_nu=np.inf, min_abs_dist=np.inf, min_kink=np.inf, clamp_margin=np.inf, G4_clamped=False,
               G5_limits=False, G6_self_contact=False, G7_basis_switch=False)
    con, nit, ncon, nefc, warned = [], 0, 0, 0, False
    for _ in range(nsub):
        qin = d.qpos.copy()
        mujoco.mj_step(m, d)
        con.append(fdlib._contacts(C, d, True) + (fdlib._limits(d),))
        nit = max(nit, _niter(d))
        ncon, nefc = max(ncon, d.ncon), max(nefc, d.nefc)
        _substep_labels(m, d, lab)
        lab["min_abs_dist"] = min(lab["min_abs_dist"], _near_distance(qin))
        warned = warned or any(d.warning[w].number for w in BAD_WARNINGS)
    out = np.concatenate([d.qpos, d.qvel])
    res = dict(out=out, warm=d.qacc_warmstart.copy(), niter=nit, ncon_max=ncon, nefc_max=nefc,
               nonfinite=bool(warned or not np.isfinite(out).all()), **pack(con, nsub))
    res.update(lab)
    return res


def _floor_state(name, q, v, warm, u, fric, com):
    m, d = _W["floors"][name]
    _apply_dr(m, fric, com)
    _load(d, q, v, u, warm)
    nit = 0
    for _ in range(_W["nsub"]):
        mujoco.mj_step(m, d)
        nit = max(nit, _niter(d))
    return np.concatenate([d.qpos, d.qvel]), d.qacc_warmstart.copy(), nit


def label_task(job):
    """job = (lo, rows [(qpos, qvel, warm, ctrl, friction, com) float64]) -> (lo, {key: stacked per state})."""
    lo, rows = job
    res = []
    for q, v, warm, u, fric, com in rows:
        r = _truth_state(q, v, warm, u, fric, com)
        for name in _W["floors"]:
            r[f"out_{name}"], r[f"warm_{name}"], r[f"niter_{name}"] = _floor_state(name, q, v, warm, u, fric, com)
        res.append(r)
    return lo, {k: np.stack([np.asarray(r[k]) for r in res]) for k in res[0]}


def fd_task(job):
    """fdlib.fd_job on this worker's truth model (job: its argument tuple); the arrays FD_KEYS."""
    assert _W["nsub"] == fdlib.NSUB, f"fd_job runs {fdlib.NSUB} substeps, the problem has {_W['nsub']}"
    r = fdlib.fd_job(job)
    return {k: r[k] for k in ("key",) + FD_KEYS}


def tables_task(_):
    """This worker's fdlib tables (for checking them against fdlib._worker_init's)."""
    C = fdlib._C
    m = C["m"]
    return dict({k: v for k, v in C.items() if k not in ("m", "d")},
                options=(m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations, m.opt.disableflags))


# ------------------------------------------------------------------------------------------------- driver side

def input_rows(arrays, idx=None):
    """[(qpos, qvel, warm, ctrl, friction, com)] float64 rows of a corpus's arrays (friction 0, com 0 without DR)."""
    n = len(arrays["qpos"])
    idx = np.arange(n) if idx is None else np.asarray(idx)
    f64 = lambda k: np.asarray(arrays[k], np.float64)  # noqa: E731
    q, v, w, u = f64("qpos"), f64("qvel"), f64("qacc_warmstart"), f64("ctrl")
    fr = f64("friction") if "friction" in arrays else np.zeros(n)
    com = f64("com") if "com" in arrays else np.zeros((n, 3))
    return [(q[i], v[i], w[i], u[i], float(fr[i]), com[i]) for i in idx]


class Pool:
    """A spawn process pool of `jobs` MuJoCo C workers for one problem (init_worker)."""

    def __init__(self, prob, jobs):
        self.prob, self.jobs = prob, max(1, int(jobs))
        self.nsub = int(prob.substeps)
        self.pool = mp.get_context("spawn").Pool(self.jobs, init_worker, (prob.to_dict(), self.nsub))

    def labels(self, arrays, chunk=CHUNK):
        """The truth run (+ labels, contact signature) and the floor runs of every state: dict of (S, ...) arrays."""
        rows = input_rows(arrays)
        jobs = [(lo, rows[lo:lo + chunk]) for lo in range(0, len(rows), chunk)]
        parts = sorted(self.pool.imap_unordered(label_task, jobs), key=lambda x: x[0])
        return {k: np.concatenate([p[k] for _, p in parts]) for k in parts[0][1]}

    def fd(self, arrays, worlds, steps=FD_STEPS, rtol=FD_STEP_RTOL, atol=FD_STEP_ATOL):
        """{world: fd_task result} for the states `worlds`."""
        rows = input_rows(arrays, worlds)
        jobs = [(int(w), *r[:4], r[4], r[5], list(steps), rtol, atol) for w, r in zip(worlds, rows)]
        return {r["key"]: r for r in self.pool.imap_unordered(fd_task, jobs, chunksize=2)}

    def tables(self):
        return self.pool.apply(tables_task, (None,))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self.pool.terminate()
            self.pool = None


# ------------------------------------------------------------------------------------------------- labels and groups

def label_bits(lab):
    """The LABEL_BITS mask (S,) uint16 of a labels() dict."""
    S = len(lab["out"])
    b = {k: np.zeros(S, bool) for k in LABEL_BITS}
    b["G1_axis_lt_5mm"] = lab["min_nu"] < AXIS_5MM
    b["G1_axis_lt_1mm"] = lab["min_nu"] < AXIS_1MM
    b["G2_contact_flip"] = lab["min_abs_dist"] < FLIP_DIST
    b["G3_kink"] = lab["min_kink"] < KINK_REL
    for k in ("G4_clamped", "G5_limits", "G6_self_contact", "G7_basis_switch"):
        b[k] = np.asarray(lab[k], bool)
    b["G9_nonconv"] = lab[f"niter_{G9_RUN}"] >= FLOORS[G9_RUN]["iterations"]
    b["G0_normal"] = ~np.any([b[k] for k in OUTSIDE_G0], axis=0)
    b["truth_nonconv"] = lab["niter"] >= TRUTH["iterations"]
    b["nonfinite"] = np.asarray(lab["nonfinite"], bool)
    out = np.zeros(S, np.uint16)
    for k, bit in LABEL_BITS.items():
        out[b[k]] |= np.uint16(bit)
    return out


def unpack_labels(bits):
    """{label: bool (S,)} of a LABEL_BITS mask."""
    bits = np.asarray(bits)
    return {k: (bits & bit) != 0 for k, bit in LABEL_BITS.items()}


def group_masks(bits):
    """{group: bool (S,)} (GROUP_DEFS) of a LABEL_BITS mask."""
    b = unpack_labels(bits)
    normal, clamped = b["G0_normal"], b["G4_clamped"]
    return dict(G0=normal & ~clamped, G4=normal & clamped, G5=normal & b["G5_limits"] & ~clamped,
                G6=normal & b["G6_self_contact"] & ~clamped, G1=b["G1_axis_lt_5mm"])


def sample(masks, corpus, cap=SAMPLE_CAP, seed=SAMPLE_SEED, groups=GROUPS):
    """{group: sorted state indices} (module docstring) and their union."""
    pick = {}
    for g in groups:
        W = len(masks[g])
        order = np.random.default_rng([seed, zlib.crc32(SAMPLE_SET.encode()), zlib.crc32(corpus.encode()),
                                       zlib.crc32(g.encode())]).permutation(W)
        pick[g] = np.sort(order[masks[g][order]][:cap])
    worlds = np.unique(np.concatenate([pick[g] for g in groups] + [np.zeros(0, int)])).astype(np.int64)
    return pick, worlds


def model_check(c, w):
    """Per state the MODEL_FLAGS of MuJoCo C's truth run c against MuJoCo Warp's w (both dicts of PACK_KEYS arrays
    over the same states, (S, nsub, ...)), and the largest frame rotation."""
    S, nsub = c["con_code"].shape[:2]
    if w["con_code"].shape[:2] != (S, nsub):
        raise ValueError(f"model check: MuJoCo C has {(S, nsub)} (states, substeps), MuJoCo Warp "
                         f"{w['con_code'].shape[:2]}")
    flags = np.zeros(S, np.int32)
    rot = np.zeros(S)
    for i in range(S):
        for k in range(nsub):
            a, b = c["con_code"][i, k], w["con_code"][i, k]
            if not np.array_equal(a, b):
                flags[i] |= MODEL_FLAGS["model_contacts"]
                continue
            n = int((a >= 0).sum())
            if (not np.array_equal(c["con_zone"][i, k, :n], w["con_zone"][i, k, :n])
                    or c["lim_rows"][i, k] != w["lim_rows"][i, k] or c["lim_act"][i, k] != w["lim_act"][i, k]):
                flags[i] |= MODEL_FLAGS["model_zones"]
            for j in range(n):
                zc = c["con_zone"][i, k, j]
                if c["con_dim"][i, k, j] < 3 or zc in (0, 15):
                    continue
                t1c = c["con_t"][i, k, j, :3]
                tw = np.asarray(w["con_t"][i, k, j], np.float64)
                rot[i] = max(rot[i], min(1 - abs(t1c @ tw[:3]), 1 - abs(t1c @ tw[3:])))
        if rot[i] > FRAME_ROT:
            flags[i] |= MODEL_FLAGS["model_frame"]
    return flags, rot


def vjp(J, cot_qpos, cot_qvel):
    """FD VJPs (S, C, ncol) of Jacobians J (S, nq + nv, ncol) for cotangents (S, C, nq), (S, C, nv)
    (float64, cotangent by cotangent)."""
    nq = cot_qpos.shape[2]
    cq = lambda c: np.ascontiguousarray(cot_qpos[:, c]).astype(np.float64)  # noqa: E731
    cv = lambda c: np.ascontiguousarray(cot_qvel[:, c]).astype(np.float64)  # noqa: E731
    return np.stack([np.einsum("wr,wrc->wc", cq(c), J[:, :nq]) + np.einsum("wr,wrc->wc", cv(c), J[:, nq:])
                     for c in range(cot_qpos.shape[1])], 1)
