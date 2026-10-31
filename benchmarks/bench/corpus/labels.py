"""Per-state labels of one control step with MuJoCo C (float64), for generating and validating benchmark corpora.

For each state (qpos, qvel, qacc_warmstart, ctrl and, for a model with domain randomization, the per-state foot-floor
friction and torso COM offset) one control step of `substeps` mj_step calls with ctrl held runs on a copy of the model
from that exact input (float32 corpus values read as float64), as the MuJoCo C baseline and the truth do. Per substep it
reads the contacts, constraint rows, limit rows, solver iterations and actuator forces and returns, per state:

  ncon_max, nefc_max, nl_max   most contacts / constraint rows / limit rows in any substep            int32
  ncon0, nefc0                 contacts / rows of substep 0 (the input state's own)                  int32
  niter_max                    most solver iterations in any substep (and any island)                int32
  self_contact                 a contact between two geoms of bodies not welded to the world          bool
  clamped                      an actuated joint's actuator force (the sum of gear * actuator_force
                               of its joint-transmission actuators) outside its jnt_actfrcrange, or a
                               force-limited actuator at its forcerange, in some substep              bool
  limit                        a limit row (joint or tendon) in some substep                          bool
  nonconv                      solver iterations at the iteration limit in some substep               bool
  bad                          MuJoCo warned of a bad qpos / qvel / qacc (it then resets the data) or
                               the result is not finite                                                bool
  out_qpos, out_qvel           the state after the control step                                       float64

Nothing here depends on the robot's names: the domain randomization takes the pair indices and the body id from the
caller (g1_meta.json's foot_pairs and torso_body for the G1). Work is spread over Python threads, each with its own
model copy and MjData; mj_step releases the GIL.

  lab = labels.Labeller(model, substeps=4, dr=(foot_pairs, torso_id), threads=24)
  out = lab(states, iterations=10)     # states: dict of (S, .) arrays with the corpus keys
"""

import copy
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

BAD_WARNINGS = (int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL),
                int(mujoco.mjtWarning.mjWARN_BADQACC))
INT_KEYS = ("ncon_max", "nefc_max", "nl_max", "ncon0", "nefc0", "niter_max")
BOOL_KEYS = ("self_contact", "clamped", "limit", "nonconv", "bad")


def robot_geoms(m):
    """Per geom: True when its body is not welded to the world (a geom that moves with the robot)."""
    return m.body_weldid[m.geom_bodyid] != 0


class Labeller:
    """MuJoCo C control-step labels of a model (see the module docstring)."""

    def __init__(self, m, substeps, dr=None, threads=1, chunk=64):
        """m: MjModel with the physics options of the benchmark (its opt.iterations is set per call); substeps:
        mj_step calls per control step; dr: None or (pair indices whose friction[:2] is the per-state friction,
        body id whose body_ipos gets the per-state COM offset); threads: Python threads; chunk: states per task."""
        self.m, self.substeps, self.threads, self.chunk = m, int(substeps), max(1, int(threads)), int(chunk)
        self.dr = None if dr is None else (np.asarray(dr[0], int), int(dr[1]))
        self.robot = robot_geoms(m)
        jt = m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
        self.act_j = np.flatnonzero(jt)
        self.act_jnt = m.actuator_trnid[self.act_j, 0]
        self.act_gear = m.actuator_gear[self.act_j, 0]
        self.jnt_lim = m.jnt_actfrclimited.astype(bool)
        self.jnt_lo, self.jnt_hi = m.jnt_actfrcrange[:, 0], m.jnt_actfrcrange[:, 1]
        self.f_lim = np.flatnonzero(m.actuator_forcelimited.astype(bool))
        self.f_lo, self.f_hi = m.actuator_forcerange[self.f_lim, 0], m.actuator_forcerange[self.f_lim, 1]

    def _clamped(self, force):
        q = np.zeros(self.m.njnt)
        np.add.at(q, self.act_jnt, self.act_gear * force[self.act_j])
        out = bool(np.any(self.jnt_lim & ((q < self.jnt_lo) | (q > self.jnt_hi))))
        if len(self.f_lim):
            f = force[self.f_lim]
            out |= bool(np.any((f <= self.f_lo) | (f >= self.f_hi)))
        return out

    def _job(self, args):
        lo, hi, s, iterations = args
        m = copy.copy(self.m)
        m.opt.iterations = int(iterations)
        d = mujoco.MjData(m)
        n = hi - lo
        out = {k: np.zeros(n, np.int32) for k in INT_KEYS}
        out.update({k: np.zeros(n, bool) for k in BOOL_KEYS})
        out["out_qpos"], out["out_qvel"] = np.zeros((n, m.nq)), np.zeros((n, m.nv))
        if self.dr is not None:
            pairs, body = self.dr
            ipos0 = m.body_ipos[body].copy()
        for i in range(n):
            j = lo + i
            if self.dr is not None:
                m.pair_friction[pairs, 0] = m.pair_friction[pairs, 1] = s["friction"][j]
                m.body_ipos[body] = ipos0 + s["com"][j]
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:], d.ctrl[:] = s["qpos"][j], s["qvel"][j], s["ctrl"][j]
            d.qacc_warmstart[:] = s["qacc_warmstart"][j]
            ncon = nefc = nl = nit = 0
            self_c = clamped = False
            for k in range(self.substeps):
                mujoco.mj_step(m, d)
                nc = d.ncon
                if k == 0:
                    out["ncon0"][i], out["nefc0"][i] = nc, d.nefc
                ncon, nefc, nl = max(ncon, nc), max(nefc, d.nefc), max(nl, d.nl)
                nit = max(nit, int(d.solver_niter[:max(d.nisland, 1)].max()))
                if nc and not self_c:
                    con = d.contact
                    self_c = bool(np.any(self.robot[con.geom1] & self.robot[con.geom2]))
                clamped = clamped or self._clamped(d.actuator_force)
            out["ncon_max"][i], out["nefc_max"][i], out["nl_max"][i], out["niter_max"][i] = ncon, nefc, nl, nit
            out["self_contact"][i], out["clamped"][i] = self_c, clamped
            out["out_qpos"][i], out["out_qvel"][i] = d.qpos, d.qvel
            warned = any(d.warning[w].number for w in BAD_WARNINGS)
            out["bad"][i] = warned or not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        out["limit"] = out["nl_max"] > 0
        out["nonconv"] = out["niter_max"] >= int(iterations)
        return lo, out

    def __call__(self, states, iterations):
        """Labels of every state of `states` (dict of (S, .) arrays: qpos, qvel, qacc_warmstart, ctrl and with DR
        friction, com) at `iterations` solver iterations; dict of (S,) arrays (and out_qpos / out_qvel)."""
        s = {k: np.asarray(states[k], np.float64) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl")}
        if self.dr is not None:
            s["friction"] = np.asarray(states["friction"], np.float64)
            s["com"] = np.asarray(states["com"], np.float64)
        S = len(s["qpos"])
        jobs = [(lo, min(lo + self.chunk, S), s, iterations) for lo in range(0, S, self.chunk)]
        if self.threads == 1:
            parts = list(map(self._job, jobs))
        else:
            with ThreadPoolExecutor(self.threads) as ex:
                parts = list(ex.map(self._job, jobs))
        if not parts:
            m = self.m
            out = {k: np.zeros(0, np.int32) for k in INT_KEYS}
            out.update({k: np.zeros(0, bool) for k in BOOL_KEYS})
            out.update(out_qpos=np.zeros((0, m.nq)), out_qvel=np.zeros((0, m.nv)))
            return out
        parts.sort(key=lambda x: x[0])
        return {k: np.concatenate([p[k] for _, p in parts]) for k in parts[0][1]}
