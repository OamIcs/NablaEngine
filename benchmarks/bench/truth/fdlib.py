"""MuJoCo C float64 central differences of one control step: the helpers behind the gradient truth (truth/mjc.py).

Truth. For one state, MuJoCo C (float64, the model with the state's foot friction and torso COM offset as in
nablaengine.reference.warp_physics._set_dr; the solver options of the caller's model) runs one control step
(NSUB x mj_step) at the unperturbed input and at x +- h e_j for every column j of (qpos tangent (nv, perturbed with
mj_integratePos), qvel (nv), ctrl (nu)) and each step size h of `steps`: the full Jacobian
d(qpos', qvel') / d(qpos, qvel, ctrl) (qpos' in qpos form). A column is invalid (bits BAD, `bad`):
  step       the two step sizes differ by more than step_rtol relative plus step_atol (the roundoff of a column block
             at h = 1e-7 is ~1e-7, so a derivative that is exactly 0, e.g. qvel' along a horizontal root translation,
             is not rejected), on the qpos' or the qvel' rows
  contacts   on some side, some substep's contact set (pair, slot) differs from the unperturbed run
  rows       ... its constraint rows (type, id) or their zones differ (active = efc_force > 0: MuJoCo C 3.17 solves
             per island and leaves efc_state stale, e.g. limit rows that carry force read SATISFIED; the states are
             in iefc_state)
  clamp      ... its actuator clamp mask (|PD force| beyond jnt_actfrcrange) differs
  ctrlrange  a ctrl column whose +-h leaves the ctrlrange (MuJoCo clamps ctrl there)
  solver     a run hit the iteration limit

kv0 truth (fd_job with kv0 true): the exact derivative of a modified forward in which, on every substep s, each dof
whose actuator force jnt_actfrcrange clamps in the unperturbed run's substep s gets the clamped force
- kv (qvel - v_base(s)) (v_base(s): the unperturbed qvel at the start of substep s), added as qfrc_applied (kv0_damping:
kv = -gear^2 biasprm[2] of the joint PD actuators); the implicitfast matrix as MuJoCo C's (it keeps such a dof's -kv);
everything else unchanged. It is the derivative that an adjoint which keeps the actuator damping on clamped dofs
(clamp_kv=0) computes exactly.

Per-process state: _C (the model, its data and lookup tables), filled by _worker_init or by the caller (truth/mjc.py
fills it for any problem); the helpers read it.
"""
import time

import mujoco
import numpy as np

NSUB = 4  # substeps per control step
MAXC = 64  # contact slots per substep in the caches
BAD = dict(step=1, contacts=2, rows=4, clamp=8, ctrlrange=16, solver=32)  # reasons a column is invalid
MODEL = dict(model_contacts=1, model_frame=2, model_zones=4)  # state level: MuJoCo C differs from MuJoCo Warp
LIMIT_JOINT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)

_C = {}  # per worker: model, data and lookup tables


def _worker_init(iterations, tolerance, ls_iterations):
    from nablaengine import fixtures  # the default model (only this initializer needs nablaengine)
    m, mt = fixtures.load_model()
    m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = iterations, tolerance, ls_iterations
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    torso = fixtures.torso_id(m, mt)
    pair_of = np.full((m.ngeom, m.ngeom), -1, np.int64)
    for i in range(m.npair):
        pair_of[m.pair_geom1[i], m.pair_geom2[i]] = pair_of[m.pair_geom2[i], m.pair_geom1[i]] = i
    jid = m.actuator_trnid[:, 0]
    lim = m.jnt_actfrclimited[jid].astype(bool)
    _C.update(m=m, d=mujoco.MjData(m), torso=torso, ipos0=m.body_ipos[torso].copy(), fric0=m.pair_friction.copy(),
              foot=np.asarray(mt["foot_pairs"], int), pair_of=pair_of,
              plane=m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, capsule=m.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE,
              lo=np.where(lim, m.jnt_actfrcrange[jid, 0], -np.inf), hi=np.where(lim, m.jnt_actfrcrange[jid, 1], np.inf),
              ctrl_lo=np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 0], -np.inf),
              ctrl_hi=np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 1], np.inf),
              act_dof=m.jnt_dofadr[jid], act_kv=kv0_damping(m))


def kv0_damping(m):
    """Per actuator its force's derivative in its dof's velocity, negated (kv): -gear^2 biasprm[2] of a joint actuator
    with a fixed gain and an affine bias (the mjlab G1's position actuators: kp ctrl - kp q - kv v); None if the model
    has another kind (the kv0 truth, fd_job, then refuses), or two actuators share a dof."""
    ok = (np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
          and np.all(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
          and np.all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
          and np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE) and not m.actuator_forcelimited.any()
          and len(np.unique(m.jnt_dofadr[m.actuator_trnid[:, 0]])) == m.nu)
    return -m.actuator_gear[:, 0] ** 2 * m.actuator_biasprm[:, 2] if ok else None


def _zones(d):
    """The rows' zones of the substep just stepped: active (force > 0) or not. MuJoCo C 3.17 solves per island and does
    not update efc_state (limit rows that carry force read SATISFIED, every pyramid edge reads QUADRATIC; the states are
    in iefc_state); efc_force = -D jar on the active rows of the final iterate."""
    return d.efc_force[:d.nefc] > 0


def _contacts(C, d, full):
    """The contacts of the substep just stepped, sorted by code = pair * 4 + slot (adjoint_groups' slot: a plane-capsule
    contact's end, 0 = +axis; else 0): code, and with `full` also (frame tangent rows t1 t2 (n, 6), dim, zone: bit e =
    pyramid edge e (or the single row of condim 1) active)."""
    n = d.ncon
    if n == 0:
        return np.zeros(0, np.int64), np.zeros((0, 6)), np.zeros(0, np.int32), np.zeros(0, np.int32)
    con = d.contact
    g1, g2 = np.asarray(con.geom1[:n]), np.asarray(con.geom2[:n])
    cap = np.where(C["plane"][g1], g2, g1)
    pc = (C["plane"][g1] | C["plane"][g2]) & C["capsule"][cap]
    ax = d.geom_xmat[cap][:, [2, 5, 8]]
    end = np.where(np.sum((con.pos[:n] - d.geom_xpos[cap]) * ax, 1) > 0, 0, 1)
    code = C["pair_of"][g1, g2] * 4 + np.where(pc, end, 0)
    o = np.argsort(code, kind="stable")
    if not full:
        return (code[o],)
    dim = np.array(con.dim[:n])
    nrow = np.where(dim > 1, 2 * (dim - 1), 1)  # pyramidal cone
    act = _zones(d)
    zone = np.array([int(np.sum(act[a:a + r] << np.arange(r))) for a, r in zip(con.efc_address[:n], nrow)], np.int32)
    return code[o], np.array(con.frame[:n])[o, 3:9], dim[o], zone[o]


def _limits(d):
    """The substep's joint-limit rows as masks over joints: (rows present, rows active)."""
    ne = d.nefc
    sel = d.efc_type[:ne] == LIMIT_JOINT
    j = d.efc_id[:ne][sel].astype(np.int64)
    return int(np.sum(np.int64(1) << j)), int(np.sum((np.int64(1) << j)[_zones(d)[sel]]))


def _run(C, q, v, u, warm, full=False, kv0=None):
    """One control step from (q, v, u): (qpos' and qvel' (nq + nv), per-substep signatures (contact codes, rows,
    zones, clamp mask), max solver iterations, per-substep _contacts + (_limits,) with `full`, the qvel at the start of
    every substep (NSUB, nv)). kv0 = (clamped (NSUB, nu) bool, v_base (NSUB, nv)): the kv0-semantics forward (fd_job):
    before substep s, qfrc_applied = -kv (qvel - v_base[s]) on the dofs of the actuators clamped[s] (MuJoCo C adds
    qfrc_applied explicitly, outside the implicitfast matrix), zero elsewhere and without kv0."""
    m, d = C["m"], C["d"]
    d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q, v, u, warm
    d.time = 0.0
    d.qfrc_applied[:] = 0.0
    sig, con, nit, vin = [], [], 0, np.zeros((NSUB, m.nv))
    for k in range(NSUB):
        vin[k] = d.qvel
        if kv0 is not None:
            a = kv0[0][k]
            dof = C["act_dof"][a]
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[dof] = -C["act_kv"][a] * (d.qvel[dof] - kv0[1][k, dof])
        mujoco.mj_step(m, d)
        c = _contacts(C, d, full)
        ne = d.nefc
        raw = d.actuator_force
        sig.append((c[0].tobytes(), (d.efc_type[:ne].tobytes(), d.efc_id[:ne].tobytes()), _zones(d).tobytes(),
                    ((raw < C["lo"]) | (raw > C["hi"])).tobytes()))
        con.append(c + (_limits(d),) if full else c)
        nit = max(nit, int(d.solver_niter[0]))
    d.qfrc_applied[:] = 0.0
    return np.concatenate([d.qpos, d.qvel]), sig, nit, con, vin


def _set_dr(C, fric, com):
    """The state's domain randomization, as reference/warp_physics._set_dr: foot-floor friction, torso COM offset."""
    m = C["m"]
    m.body_ipos[C["torso"]] = C["ipos0"] + com
    m.pair_friction[:] = C["fric0"]
    m.pair_friction[C["foot"], :2] = fric


def _pack(con):
    """Per-substep _contacts + _limits (full) -> fixed-size arrays con_code (NSUB, MAXC) (-1 = none), con_t, con_dim,
    con_zone, lim_rows, lim_act (NSUB,)."""
    code = np.full((NSUB, MAXC), -1, np.int64)
    tang = np.zeros((NSUB, MAXC, 6))
    dim = np.zeros((NSUB, MAXC), np.int32)
    zone = np.zeros((NSUB, MAXC), np.int32)
    lim = np.zeros((NSUB, 2), np.int64)
    for k, (c, t, dm, zn, lm) in enumerate(con):
        n = min(len(c), MAXC)
        code[k, :n], tang[k, :n], dim[k, :n], zone[k, :n] = c[:n], t[:n], dm[:n], zn[:n]
        lim[k] = lm
    return dict(con_code=code, con_t=tang, con_dim=dim, con_zone=zone, lim_rows=lim[:, 0], lim_act=lim[:, 1])


def model_job(job):
    """The unperturbed control step of a batch of states, for the model check before sampling. job = (key, worlds,
    [(qpos, qvel, warm, ctrl, friction, com)]); returns (key, worlds, {out, niter, con_*: stacked per state})."""
    key, worlds, rows = job
    res = []
    for q, v, warm, u, fric, com in rows:
        _set_dr(_C, fric, com)
        o, _, nit, con, _ = _run(_C, *(np.asarray(x, np.float64) for x in (q, v, u, warm)), True)
        res.append(dict(out=o, niter=nit, **_pack(con)))
    return key, worlds, {k: np.stack([r[k] for r in res]) for k in res[0]}


def fd_job(job):
    """Central differences of one state (see the module docstring). job = (key, qpos, qvel, warm, ctrl, friction, com,
    steps, step_rtol, step_atol[, kv0]); with kv0 true, of the kv0-semantics forward (the module docstring's kv0 truth:
    every perturbed run adds -kv (qvel - v_base(s)) on the dofs clamped in the unperturbed run's substep s; the
    unperturbed run is the plain one), also returning its clamp masks (NSUB, nu). Returns a dict of numpy arrays."""
    key, q0, v0, warm, u0, fric, com, steps, rtol, atol, *opt = job
    t0 = time.time()
    C = _C
    m = C["m"]
    _set_dr(C, fric, com)
    q0, v0, u0, warm = (np.asarray(x, np.float64) for x in (q0, v0, u0, warm))
    nq, nv, nu = m.nq, m.nv, m.nu
    ncol = 2 * nv + nu
    base, bsig, bnit, bcon, bvin = _run(C, q0, v0, u0, warm, True)
    kv0 = None
    if opt and opt[0]:
        assert C["act_kv"] is not None, "kv0 truth: the model's actuators are not joint PD actuators (kv0_damping)"
        kv0 = (np.stack([np.frombuffer(s[3], bool) for s in bsig]), bvin)
    J = np.zeros((len(steps), nq + nv, ncol))
    bad = np.zeros(ncol, np.int32)
    lim = m.opt.iterations
    if bnit >= lim:
        bad |= BAD["solver"]
    for j in range(ncol):
        for si, h in enumerate(steps):
            out = []
            for sg in (1.0, -1.0):
                q, v, u = q0.copy(), v0.copy(), u0.copy()
                if j < nv:
                    dq = np.zeros(nv)
                    dq[j] = sg * h
                    mujoco.mj_integratePos(m, q, dq, 1.0)
                elif j < 2 * nv:
                    v[j - nv] += sg * h
                else:
                    u[j - 2 * nv] += sg * h
                o, sig, nit, _, _ = _run(C, q, v, u, warm, kv0=kv0)
                for a, b in zip(sig, bsig):
                    bad[j] |= ((a[0] != b[0]) * BAD["contacts"] | (a[1] != b[1] or a[2] != b[2]) * BAD["rows"]
                               | (a[3] != b[3]) * BAD["clamp"])
                if nit >= lim:
                    bad[j] |= BAD["solver"]
                out.append(o)
            J[si, :, j] = (out[0] - out[1]) / (2 * h)
    for blk in (slice(0, nq), slice(nq, nq + nv)):  # the two step sizes, qpos' and qvel' rows separately
        a, b = J[0, blk], J[-1, blk]
        den = np.maximum(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
        bad[np.linalg.norm(a - b, axis=0) > rtol * den + atol] |= BAD["step"]
    hm = max(steps)
    bad[2 * nv:][(u0 + hm > C["ctrl_hi"]) | (u0 - hm < C["ctrl_lo"])] |= BAD["ctrlrange"]
    res = dict(key=key, J=J, bad=bad, out=base, niter=bnit, ncon=np.array([len(c[0]) for c in bcon]),
               seconds=time.time() - t0, **_pack(bcon))
    if kv0 is not None:
        res["clamp"] = kv0[0]
    return res
