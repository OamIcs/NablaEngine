"""The kinematics outputs of nablaengine.jax_ffi, step(..., kinematics=True) and jax_ffi.kinematics, against MuJoCo
MJX (mjx.kinematics + smooth.com_pos + smooth.com_vel with the per-world torso COM offset added to
body_ipos[torso]), float32 with jax.default_matmul_precision("highest"), on corpus states.

Corpora: --corpora (default mf_s0, mf_s1: benchmarks/data, nablaengine.fixtures), their input states and the
output states of their MuJoCo Warp reference (out_qpos / out_qvel, with the corpus's com); the step, batch and timing
sections use the first corpus.

Sections (--sections, default all; each in its own process):
  mjx        jax_ffi.kinematics_forward against MJX on every corpus's input and output states: xpos / subtree_com max
             abs error, xquat max abs error (and how many quaternions are closer to -q than to q: sign flips), cvel
             relative error per world (|d| / |ref| over the world's nbody x 6, rel_rows) median / p99 / max and per
             body row p99, qpos_n against MJX's written-back qpos (the quaternion's max abs error; the other entries
             bitwise). Gates (GATES): xpos, subtree_com <= 1e-5 m, xquat <= 1e-6, cvel median <= 1e-6 and p99 <= 1e-5,
             qpos_n's quaternion <= 1e-6 and its other entries bitwise. Context (not gated): both against a float64
             truth (section f64) on a subset of the states
  vjp        jax.vjp through jax_ffi.kinematics against jax.vjp of the same MJX map, random N(0, 1) cotangents on each
             output block alone and on all five together: g_qpos, g_qvel relative error per world median / p99 / max
             (a world whose reference row is zero must be zero too); the quaternion gradient's radial part |g . q| / |g|
             (MJX's normalization removes it). Gates: median <= 1e-5, p99 <= 1e-4
  f64        (context, in a process with JAX_ENABLE_X64) the same MJX map in float64 on --f64-worlds states per corpus:
             forward outputs and the VJPs of section vjp's cotangents, to which mjx and vjp compare both the engine and
             MJX float32 (the rounding level)
  step       step(..., kinematics=True) (interface defaults; 4096 states): q', v', warm' bitwise equal to
             step(kinematics=False) (primal: dk_step_kin against dk_step; under jax.vjp: dk_fwd_kin against dk_fwd); its
             kinematics outputs bitwise equal to jax_ffi.kinematics(q', v', com); jax.vjp with cotangents on all eight
             outputs bitwise equal to the composition of step's VJP and kinematics' VJP (for clamp_kv 0 and the soft
             clamp 2), and against the composition with MJX's kinematics VJP (rounding level, reported); remat: a scan of
             H steps with a loss on the kinematics outputs, plain / jax.checkpoint(body) / jax.checkpoint(body,
             policy=remat_policy()): forward FFI calls counted (the library's counters: H, 2 H, H) and the policy's
             gradients bitwise the plain scan's
  batch      B in {1, 40, 160, 4096}: per-world rows of kinematics_forward / its VJP and of step(kinematics=True) / its
             VJP bitwise those of B = 4096; repeated calls with the same B compile nothing (jax.monitoring's
             backend-compile events) and trace once; jax.grad of a scalar loss through step(kinematics=True) inside jit
             and inside lax.scan (the next state is the step's qpos_n) finite and bitwise equal to the same loss
             unrolled in Python
  timing     B = 4096 (10 iterations, jit, warm, medians over --timing-repeats interleaved repeats of --reps calls):
             the extra cost of kinematics=True in the step's forward (dk_step_kin - dk_step, dk_fwd_kin - dk_fwd) and
             backward (dk_bwd_kin - dk_bwd) and in jax.vjp; jax_ffi.kinematics forward and forward + VJP against MJX's
             three steps forward and forward + VJP (outputs and gradients returned, so XLA drops nothing); the bare
             kernels k_kin and k_kin_vjp (the library's shims through ctypes, default stream)
--kinematics-scan adds scans whose carried state advances through jax_ffi.kinematics alone (the next qpos is
qpos_n + theta_t): in step, their remat modes (the compiled HLO's dk_kin custom calls: 1, 2, 1; gradients bitwise), and
in batch, their gradient against the unrolled loss.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      python scripts/kinematics_check.py --out runs/kinematics_check.json

Exit status: 0 when every section passes, 1 otherwise (the JSON is written either way).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("f64", "mjx", "vjp", "step", "batch", "timing")
CORPORA = []  # --corpora (set in main)
OUTS = ("qpos_n", "xpos", "xquat", "cvel", "subtree_com")
GATES = dict(xpos=1e-5, subtree_com=1e-5, xquat=1e-6, cvel_median=1e-6, cvel_p99=1e-5, qpos_n_quat=1e-6,
             vjp_median=1e-5, vjp_p99=1e-4)

from nablaengine import fixtures, provenance, workdir  # noqa: E402


def same_bits(a, b):
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    return bool(a.shape == b.shape and a.dtype.itemsize == b.dtype.itemsize
                and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def rel_rows(a, b):
    """Per world (first axis): |a - b| / max(|b|, 1e-9) over the flattened rest (float64)."""
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-9)


def stats(e):
    e = np.asarray(e, np.float64).ravel()
    if not e.size:
        return dict(n=0)
    return dict(n=int(e.size), median=float(np.median(e)), p99=float(np.quantile(e, 0.99)), max=float(e.max()))


def states(corpus):
    """{"in": (qpos, qvel, com), "out": (out_qpos, out_qvel, com)} float32 of the corpus."""
    z = fixtures.load_corpus(corpus)
    f = lambda k: np.asarray(z[k], np.float32)  # noqa: E731
    return {"in": (f("qpos"), f("qvel"), f("com")), "out": (f("out_qpos"), f("out_qvel"), f("com"))}


def cotangents(shapes, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def mjx_kinematics():
    """MJX's map (qpos, qvel, com) -> (qpos, xpos, xquat, cvel, subtree_com) with the torso COM offset com added to
    body_ipos[torso], vmapped over the worlds."""
    import jax
    from mujoco import mjx
    from mujoco.mjx._src import smooth
    m, mt = fixtures.load_model()
    mx = mjx.put_model(m)
    d0 = mjx.make_data(mx)
    torso = fixtures.torso_id(m, mt)

    def kin(qpos, qvel, com):
        mxi = mx.replace(body_ipos=mx.body_ipos.at[torso].add(com))
        d = d0.replace(qpos=qpos, qvel=qvel)
        d = mjx.kinematics(mxi, d)
        d = smooth.com_pos(mxi, d)
        d = smooth.com_vel(mxi, d)
        return d.qpos, d.xpos, d.xquat, d.cvel, d.subtree_com

    return jax.vmap(kin)


def _f64_path(args):
    return Path(args.work) / "f64.npz"


def _vjp_cases():
    """(name, the output blocks with a cotangent): each block alone, then all five."""
    return [(n, (i,)) for i, n in enumerate(OUTS)] + [("all", tuple(range(5)))]


def _case_cots(c, blocks):
    return [x if i in blocks else np.zeros_like(x) for i, x in enumerate(c)]


# ---------------------------------------------------------------- f64 (context)

def sec_f64(args):
    """(JAX_ENABLE_X64) MJX in float64 on the first --f64-worlds states of each corpus and state set: forward and the
    VJPs of the vjp section's cotangents (the same seeds) -> npz."""
    import jax
    import jax.numpy as jnp
    assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"
    from nablaengine.g1_engine import kinematics as K
    m, _ = fixtures.load_model()
    fk = mjx_kinematics()

    def fwd_vjp(q, v, c, *cot):
        outs, f = jax.vjp(fk, q, v, c)
        return outs, f(tuple(cot))[:2]

    fn = jax.jit(fwd_vjp)
    res = {}
    for ci, corpus in enumerate(CORPORA):
        for si, (src, (q, v, c)) in enumerate(states(corpus).items()):
            n = min(args.f64_worlds, len(q))
            x = [jnp.asarray(a[:n], jnp.float64) for a in (q, v, c)]
            cots = cotangents(list(K.shapes(m, len(q)).values()), seed=100 + 10 * ci + si)  # as sec_vjp
            for case, blocks in _vjp_cases():
                outs, g = fn(*x, *(jnp.asarray(a[:n], jnp.float64) for a in _case_cots(cots, blocks)))
                res[f"{corpus}_{src}_g_{case}_qpos"], res[f"{corpus}_{src}_g_{case}_qvel"] = (np.asarray(g[0]),
                                                                                             np.asarray(g[1]))
            for name, o in zip(OUTS, outs):
                res[f"{corpus}_{src}_{name}"] = np.asarray(o)
    np.savez(_f64_path(args), **res)
    print("f64", len(res), "arrays", flush=True)
    return dict(worlds=args.f64_worlds, path=provenance.rel(_f64_path(args)), arrays=len(res), **{"pass": True})


def _f64(args):
    p = _f64_path(args)
    return np.load(p) if p.exists() else None


# ---------------------------------------------------------------- mjx

def _forward_errors(ours, ref):
    """The mjx section's statistics of ours against ref (both (qpos_n, xpos, xquat, cvel, subtree_com))."""
    qn, xp, xq, cv, sc = (np.asarray(x, np.float64) for x in ours)
    rqn, rxp, rxq, rcv, rsc = (np.asarray(x, np.float64) for x in ref)
    dq, dqn = np.abs(xq - rxq), np.abs(xq + rxq)
    flips = int((np.linalg.norm(xq + rxq, axis=-1) < np.linalg.norm(xq - rxq, axis=-1)).sum())
    rows = np.linalg.norm(cv - rcv, axis=-1) / np.maximum(np.linalg.norm(rcv, axis=-1), 1e-6)
    other = np.r_[0:3, 7:qn.shape[1]]
    return dict(xpos_max_abs=float(np.abs(xp - rxp).max()), subtree_com_max_abs=float(np.abs(sc - rsc).max()),
                xquat_max_abs=float(dq.max()), xquat_max_abs_up_to_sign=float(np.minimum(dq, dqn).max()),
                xquat_sign_flips=flips, cvel_rel_world=stats(rel_rows(cv, rcv)),
                cvel_rel_body_row=stats(rows[:, 1:]), cvel_max_abs=float(np.abs(cv - rcv).max()),
                qpos_n_quat_max_abs=float(np.abs(qn[:, 3:7] - rqn[:, 3:7]).max()),
                qpos_n_other_max_abs=float(np.abs(qn[:, other] - rqn[:, other]).max()))


def sec_mjx(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    F.init()
    fk = mjx_kinematics()
    ours_fn = jax.jit(F.kinematics_forward)
    with jax.default_matmul_precision("highest"):
        mjx_fn = jax.jit(fk)
    f64 = _f64(args)
    res, ok = {}, True
    for corpus in CORPORA:
        for src, (q, v, c) in states(corpus).items():
            x = [jnp.asarray(a) for a in (q, v, c)]
            ours = [np.asarray(o) for o in ours_fn(*x)]
            with jax.default_matmul_precision("highest"):
                ref = [np.asarray(o) for o in mjx_fn(*x)]
            e = _forward_errors(ours, ref)
            e["worlds"] = len(q)
            e["qpos_n_other_bitwise"] = bool(same_bits(ours[0][:, :3], ref[0][:, :3])
                                             and same_bits(ours[0][:, 7:], ref[0][:, 7:])
                                             and same_bits(ours[0][:, :3], q[:, :3]) and same_bits(ours[0][:, 7:], q[:, 7:]))
            e["qpos_n_quat_is_xquat1"] = same_bits(ours[0][:, 3:7], ours[2][:, 1])
            e["world_body_row"] = dict(xpos_zero=bool(not ours[1][:, 0].any()),
                                       xquat_identity=bool(np.array_equal(ours[2][:, 0], np.tile([1, 0, 0, 0],
                                                                                                  (len(q), 1)))),
                                       cvel_zero=bool(not ours[3][:, 0].any()),
                                       subtree_com_is_root=same_bits(ours[4][:, 0], ours[4][:, 1]))
            if f64 is not None:  # context: both against float64 MJX on its subset
                n = len(f64[f"{corpus}_{src}_xpos"])
                tr = [f64[f"{corpus}_{src}_{name}"] for name in OUTS]
                e["vs_float64"] = dict(worlds=n, engine=_forward_errors([o[:n] for o in ours], tr),
                                       mjx_float32=_forward_errors([o[:n] for o in ref], tr))
            g = dict(xpos=e["xpos_max_abs"] <= GATES["xpos"],
                     subtree_com=e["subtree_com_max_abs"] <= GATES["subtree_com"],
                     xquat=e["xquat_max_abs"] <= GATES["xquat"],
                     cvel=e["cvel_rel_world"]["median"] <= GATES["cvel_median"]
                     and e["cvel_rel_world"]["p99"] <= GATES["cvel_p99"],
                     qpos_n=e["qpos_n_quat_max_abs"] <= GATES["qpos_n_quat"] and e["qpos_n_other_bitwise"],
                     world_body=all(e["world_body_row"].values()) and e["qpos_n_quat_is_xquat1"])
            e["gates"] = g
            e["pass"] = bool(all(g.values()))
            ok &= e["pass"]
            res[f"{corpus}_{src}"] = e
            print("mjx", corpus, src, json.dumps({k: e[k] for k in ("xpos_max_abs", "subtree_com_max_abs",
                                                                     "xquat_max_abs", "qpos_n_quat_max_abs", "pass")}),
                  json.dumps(e["cvel_rel_world"]), flush=True)
    res["gates"] = GATES
    res["pass"] = bool(ok)
    return res


# ---------------------------------------------------------------- vjp

def sec_vjp(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    from nablaengine.g1_engine import kinematics as K
    m, _ = fixtures.load_model()
    F.init()
    fk = mjx_kinematics()
    f64 = _f64(args)

    def ours(q, v, c, *cot):
        outs, f = jax.vjp(F.kinematics, q, v, c)
        return f(tuple(cot))[:2]

    def mjx(q, v, c, *cot):
        outs, f = jax.vjp(fk, q, v, c)
        return f(tuple(cot))[:2]

    ours_fn = jax.jit(ours)
    with jax.default_matmul_precision("highest"):
        mjx_fn = jax.jit(mjx)
    res, ok = {}, True
    for ci, corpus in enumerate(CORPORA):
        for si, (src, (q, v, c)) in enumerate(states(corpus).items()):
            cots = cotangents(list(K.shapes(m, len(q)).values()), seed=100 + 10 * ci + si)  # as sec_f64
            x = [jnp.asarray(a) for a in (q, v, c)]
            r = {}
            for case, blocks in _vjp_cases():
                ct = [jnp.asarray(a) for a in _case_cots(cots, blocks)]
                g = [np.asarray(a) for a in ours_fn(*x, *ct)]
                with jax.default_matmul_precision("highest"):
                    gr = [np.asarray(a) for a in mjx_fn(*x, *ct)]
                e = {}
                for name, a, b in zip(("g_qpos", "g_qvel"), g, gr):
                    zero_ref = np.linalg.norm(b, axis=1) == 0
                    e[name] = stats(rel_rows(a, b)[~zero_ref])
                    e[name]["zero_reference_rows"] = int(zero_ref.sum())
                    e[name]["zero_reference_rows_zero"] = bool(not a[zero_ref].any())
                    e[name]["pass"] = bool((not e[name]["n"] or (e[name]["median"] <= GATES["vjp_median"]
                                                                   and e[name]["p99"] <= GATES["vjp_p99"]))
                                           and e[name]["zero_reference_rows_zero"])
                u = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
                for who, gg in (("engine", g[0]), ("mjx", gr[0])):
                    gq = gg[:, 3:7].astype(np.float64)
                    e[f"radial_part_{who}"] = float((np.abs((gq * u).sum(1)) / np.maximum(np.linalg.norm(gq, axis=1),
                                                                                            1e-30)).max())
                if f64 is not None:  # context: both against float64 MJX on its subset
                    e["vs_float64"] = {}
                    for who, gg in (("engine", g), ("mjx_float32", gr)):
                        e["vs_float64"][who] = {}
                        for name, x_ in zip(("g_qpos", "g_qvel"), gg):
                            t_ = f64[f"{corpus}_{src}_g_{case}_{name[2:]}"]
                            nz = np.linalg.norm(t_, axis=1) > 0
                            e["vs_float64"][who][name] = stats(rel_rows(x_[:len(t_)], t_)[nz])
                e["pass"] = bool(e["g_qpos"]["pass"] and e["g_qvel"]["pass"])
                ok &= e["pass"]
                r[case] = e
            res[f"{corpus}_{src}"] = r
            print("vjp", corpus, src, json.dumps({k: (e["g_qpos"].get("median"), e["g_qpos"].get("p99"),
                                                      e["g_qvel"].get("median"), e["g_qvel"].get("p99"), e["pass"])
                                                  for k, e in r.items()}), flush=True)
    res["gates"] = dict(median=GATES["vjp_median"], p99=GATES["vjp_p99"])
    res["pass"] = bool(ok)
    return res


# ---------------------------------------------------------------- step

def _inputs(corpus, B=None):
    import jax.numpy as jnp
    z = fixtures.load_corpus(corpus)
    sl = slice(0, B)
    return [jnp.asarray(np.asarray(z[k][sl], np.float32)) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl",
                                                                     "friction", "com")]


def _hlo_count(text, target):
    return text.count(f'custom_call_target="{target}"')


def sec_step(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    F.init()
    a = _inputs(CORPORA[0])
    res, ok = {}, True
    # q', v', warm' unchanged; the kinematics outputs those of jax_ffi.kinematics
    o0 = jax.jit(lambda *x: F.step(*x))(*a)
    o1 = jax.jit(lambda *x: F.step(*x, kinematics=True))(*a)
    kin = jax.jit(F.kinematics_forward)(o1[0], o1[1], a[5])
    f0 = jax.jit(lambda *x: F.forward(*x, tape=True))(*a)
    f1 = jax.jit(lambda *x: F.forward(*x, tape=True, kinematics=True))(*a)
    r = dict(primal_state_bitwise=[same_bits(x, y) for x, y in zip(o0, o1[:3])],
             primal_kinematics_bitwise_kinematics=[same_bits(x, y) for x, y in zip(o1[3:], kin)],
             fwd_tape_state_info_tape_bitwise=[same_bits(x, y) for x, y in zip(f0, f1[:5])],
             fwd_kinematics_bitwise_primal=[same_bits(x, y) for x, y in zip(f1[5:], o1[3:])])
    r["pass"] = bool(all(all(v) for v in r.values()))
    res["outputs"] = r
    ok &= r["pass"]
    print("step outputs", json.dumps(r), flush=True)
    # the VJP: fused against the composition (engine kinematics: bitwise; MJX kinematics: rounding)
    fk = mjx_kinematics()
    rng = np.random.default_rng(7)
    shapes = [x.shape for x in o1]
    comp = {}
    for kv, soft in ((0, None), (2, 0.2)):
        opts = dict(clamp_kv=kv, clamp_soft=soft)
        cots = tuple(jnp.asarray(rng.standard_normal(s), jnp.float32) for s in shapes)

        def fused(*x):
            outs, f = jax.vjp(lambda *y: F.step(*y, kinematics=True, **opts), *x)
            return outs, f(cots)

        def composed(*x, kin_fn=F.kinematics):
            def g(*y):
                q, v, w = F.step(*y, **opts)
                return (q, v, w) + tuple(kin_fn(q, v, y[5]))
            outs, f = jax.vjp(g, *x)
            return outs, f(cots)

        (oa, ga), (ob, gb) = jax.jit(fused)(*a), jax.jit(composed)(*a)
        with jax.default_matmul_precision("highest"):
            oc, gc = jax.jit(lambda *x: composed(*x, kin_fn=fk))(*a)
        e = dict(outputs_bitwise=[same_bits(x, y) for x, y in zip(oa, ob)],
                 grads_bitwise_composition=[same_bits(x, y) for x, y in zip(ga, gb)],
                 grads_vs_mjx_composition={n: stats(rel_rows(x, y)) for n, x, y in
                                           zip(("g_qpos", "g_qvel", "g_warm", "g_ctrl"), ga[:4], gc[:4])
                                           if np.asarray(y).any()},
                 zero_cotangents_warm_friction_com=bool(not np.asarray(ga[2]).any() and not np.asarray(ga[4]).any()
                                                        and not np.asarray(ga[5]).any()),
                 finite=bool(all(np.isfinite(np.asarray(x)).all() for x in ga)))
        e["pass"] = bool(all(e["outputs_bitwise"]) and all(e["grads_bitwise_composition"])
                         and e["zero_cotangents_warm_friction_com"] and e["finite"])
        ok &= e["pass"]
        comp[f"clamp_kv{kv}"] = e
        print("step vjp", kv, json.dumps(e), flush=True)
    res["vjp"] = comp
    res["remat"] = _remat_step(args, a)
    ok &= res["remat"]["pass"]
    if args.kinematics_scan:
        res["remat_kinematics"] = _remat_kin(args, a)
        ok &= res["remat_kinematics"]["pass"]
    res["pass"] = bool(ok)
    return res


def _remat_step(args, a):
    """A scan of H steps (B worlds) with a loss on the kinematics outputs, three remat modes: forward calls (the
    library's counters) and the gradients (against the plain scan's, bitwise)."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    H, B = args.horizon, args.remat_worlds
    q0, v0, w0, u0, fr, com = (x[:B] for x in a)
    rng = np.random.default_rng(3)
    theta = jnp.asarray(0.01 * rng.standard_normal((H,) + u0.shape), jnp.float32)
    cw = [jnp.asarray(rng.standard_normal(s), jnp.float32) for s in ((B, 31, 3), (B, 31, 6), (B, 31, 4))]

    def body(c, th):
        q, v, w = c
        q1, v1, w1, qn, xp, xq, cv, sc = F.step(q, v, w, u0 + th, fr, com, kinematics=True)
        loss = (jnp.sum(xp * cw[0]) + 1e-2 * jnp.sum(cv * cw[1]) + jnp.sum(xq * cw[2]) + jnp.sum(sc[:, 1] ** 2)
                + jnp.sum((xp[:, 16, 2] - 0.9) ** 2))  # a nonlinear term on the torso height
        return (qn, v1, w1), loss / B

    def window(theta, mode):
        b = body
        if mode == "checkpoint":
            b = jax.checkpoint(body)
        elif mode == "policy":
            b = jax.checkpoint(body, policy=F.remat_policy())
        _, losses = jax.lax.scan(b, (q0, v0, w0), theta)
        return jnp.sum(losses)

    out = {}
    for mode in ("none", "checkpoint", "policy"):
        fn = jax.jit(jax.value_and_grad(lambda th, mode=mode: window(th, mode)))
        jax.block_until_ready(fn(theta))  # compile
        F.read_diagnostics(reset=True)
        val, g = fn(theta)
        jax.block_until_ready(g)
        d = F.read_diagnostics(reset=True)
        out[mode] = dict(fwd_calls=d["fwd_calls"], bwd_calls=d["bwd_calls"], loss=float(val), grad=np.asarray(g))
    expect = dict(none=H, checkpoint=2 * H, policy=H)
    r = {m: dict(fwd_calls=v["fwd_calls"], bwd_calls=v["bwd_calls"], expected_fwd_calls=expect[m],
                 grad_bitwise_none=same_bits(v["grad"], out["none"]["grad"]),
                 grad_rel_none=stats(rel_rows(v["grad"].reshape(H * B, -1), out["none"]["grad"].reshape(H * B, -1))))
         for m, v in out.items()}
    r["H"], r["B"] = H, B
    r["pass"] = bool(r["policy"]["fwd_calls"] == H and r["policy"]["bwd_calls"] == H and r["none"]["fwd_calls"] == H
                     and r["policy"]["grad_bitwise_none"]
                     and np.isfinite(out["none"]["grad"]).all())
    print("remat step", json.dumps(r), flush=True)
    return r


def _remat_kin(args, a):
    """jax_ffi.kinematics in a scan (the carried state advances through it alone: the next qpos is qpos_n + theta_t),
    three remat modes:
    the dk_kin / dk_kin_bwd custom calls of the compiled gradient and the gradients (bitwise against the plain one)."""
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    H, B = args.horizon, args.remat_worlds
    q0, v0, com = a[0][:B], a[1][:B], a[5][:B]
    rng = np.random.default_rng(5)
    theta = jnp.asarray(1e-3 * rng.standard_normal((H,) + q0.shape), jnp.float32)
    cw = jnp.asarray(rng.standard_normal((B, 31, 6)), jnp.float32)

    def body(c, th):
        q, v = c
        qn, xp, xq, cv, sc = F.kinematics(q, v, com)
        loss = jnp.sum((xp[:, :, 2] - 0.5) ** 2) + 1e-2 * jnp.sum(cv * cw) + jnp.sum(xq[:, 16] ** 3) + jnp.sum(sc ** 2)
        return (qn + th, 0.99 * v), loss / B

    def window(theta, mode):
        b = {"none": body, "checkpoint": jax.checkpoint(body),
             "policy": jax.checkpoint(body, policy=F.remat_policy())}[mode]
        _, losses = jax.lax.scan(b, (q0, v0), theta)
        return jnp.sum(losses)

    tg = F._STATE["targets"]
    out = {}
    for mode in ("none", "checkpoint", "policy"):
        fn = jax.jit(jax.grad(lambda th, mode=mode: window(th, mode)))
        txt = fn.lower(theta).compile().as_text()
        out[mode] = dict(dk_kin_calls=_hlo_count(txt, tg["kin"]), dk_kin_bwd_calls=_hlo_count(txt, tg["kin_bwd"]),
                         grad=np.asarray(fn(theta)))
    expect = dict(none=1, checkpoint=2, policy=1)
    r = {m: dict(dk_kin_calls=v["dk_kin_calls"], dk_kin_bwd_calls=v["dk_kin_bwd_calls"],
                 expected_dk_kin_calls=expect[m], grad_bitwise_none=same_bits(v["grad"], out["none"]["grad"]))
         for m, v in out.items()}
    r["note"] = "custom calls in the compiled gradient's HLO (each scan body once): the forward loop's, and with " \
                "jax.checkpoint(body) the backward loop's recomputation"
    r["pass"] = bool(r["policy"]["dk_kin_calls"] == 1 and r["policy"]["dk_kin_bwd_calls"] == 1
                     and r["none"]["dk_kin_calls"] == 1 and r["policy"]["grad_bitwise_none"]
                     and np.isfinite(out["none"]["grad"]).all())
    print("remat kinematics", json.dumps(r), flush=True)
    return r


# ---------------------------------------------------------------- batch

def sec_batch(args):
    import jax
    import jax.numpy as jnp
    from jax import monitoring
    from nablaengine import jax_ffi as F
    compiles = []
    monitoring.register_event_duration_secs_listener(
        lambda name, dur, **kw: compiles.append(name) if name == "/jax/core/compile/backend_compile_duration" else None)
    F.init()
    st0 = dict(F.STATS)
    a = _inputs(CORPORA[0])
    traces = {"kin": [], "kin_vjp": [], "step": [], "step_vjp": []}

    def kin(q, v, c):
        traces["kin"].append(q.shape[0])
        return F.kinematics(q, v, c)

    def kin_vjp(q, v, c, *cot):
        traces["kin_vjp"].append(q.shape[0])
        outs, f = jax.vjp(F.kinematics, q, v, c)
        return outs, f(tuple(cot))[:2]

    def step(*x):
        traces["step"].append(x[0].shape[0])
        return F.step(*x, kinematics=True)

    def step_vjp(x, cot):
        traces["step_vjp"].append(x[0].shape[0])
        outs, f = jax.vjp(lambda *y: F.step(*y, kinematics=True), *x)
        g = f(tuple(cot))
        return outs, (g[0], g[1], g[3])

    fns = dict(kin=jax.jit(kin), kin_vjp=jax.jit(kin_vjp), step=jax.jit(step), step_vjp=jax.jit(step_vjp))
    rng = np.random.default_rng(11)
    full = None
    order = (4096, 160, 40, 1, 4096, 160, 40, 1)
    rows, calls = {}, []
    for B in order:
        x = [y[:B] for y in a]
        c0 = len(compiles)
        if full is None:
            o = fns["step"](*x)
            cots = [jnp.asarray(rng.standard_normal(s.shape), jnp.float32) for s in o]
            full = dict(cots=cots)
        cs = [y[:B] for y in full["cots"]]
        r = dict(kin=fns["kin"](x[0], x[1], x[5]), kin_vjp=fns["kin_vjp"](x[0], x[1], x[5], *cs[3:]),
                 step=fns["step"](*x), step_vjp=fns["step_vjp"](x, cs))
        r = jax.tree_util.tree_map(np.asarray, r)
        calls.append(dict(B=B, compiles=len(compiles) - c0))
        if B == 4096 and "ref" not in full:
            full["ref"] = r
        rows[B] = jax.tree_util.tree_all(jax.tree_util.tree_map(
            lambda u, w, B=B: same_bits(u, w[:B]), r, full["ref"]))
    st1 = dict(F.STATS)
    res = dict(order=list(order), calls=calls, rows_bitwise_B4096={str(k): bool(v) for k, v in rows.items()},
               traces={k: v for k, v in traces.items()}, compiles_total=len(compiles),
               library_builds_during=st1["builds"] - st0["builds"], library_loads_during=st1["loads"] - st0["loads"])
    res["no_recompile"] = bool(all(c["compiles"] == 0 for c in calls[4:])
                               and all(sorted(v) == [1, 40, 160, 4096] for v in traces.values()))
    # jax.grad of a scalar loss through both entries, in jit and in lax.scan, against the same loss unrolled
    res["grad"] = _grad_scan(a, kinematics_scan=args.kinematics_scan)
    res["pass"] = bool(all(rows.values()) and res["no_recompile"] and res["library_builds_during"] == 0
                       and res["grad"]["pass"])
    print("batch", json.dumps({k: v for k, v in res.items() if k != "calls"}), flush=True)
    return res


def _grad_scan(a, H=4, B=256, kinematics_scan=False):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    q0, v0, w0, u0, fr, com = (x[:B] for x in a)
    rng = np.random.default_rng(13)
    theta = jnp.asarray(0.01 * rng.standard_normal((H,) + u0.shape), jnp.float32)
    cw = jnp.asarray(rng.standard_normal((B, 31, 3)), jnp.float32)

    def step_loss(c, th):  # a rollout: the next state is the step's qpos_n
        q, v, w = c
        q1, v1, w1, qn, xp, xq, cv, sc = F.step(q, v, w, u0 + th, fr, com, kinematics=True)
        return (qn, v1, w1), jnp.sum(xp * cw) + jnp.sum(cv[:, 1:, 3:] ** 2) + jnp.sum(xq[:, 16, 0])

    def kin_loss(c, th):  # --kinematics-scan: the state advanced through jax_ffi.kinematics alone
        q, v, w = c
        qp = q.at[:, 7:].add(th)
        qn, xp, xq, cv, sc = F.kinematics(qp, v, com)
        return (qn, 0.98 * v, w), jnp.sum(xp * cw) + jnp.sum(sc[:, 1] ** 2) + jnp.sum(cv ** 2) * 1e-2

    res, ok = {}, True
    for name, f in (("step", step_loss),) + ((("kinematics", kin_loss),) if kinematics_scan else ()):
        def scan_loss(th, f=f):
            _, ls = jax.lax.scan(f, (q0, v0, w0), th)
            return jnp.sum(ls)

        def unrolled_loss(th, f=f):
            c, tot = (q0, v0, w0), 0.0
            for t in range(H):
                c, l_ = f(c, th[t])
                tot = tot + l_
            return tot

        gs = np.asarray(jax.jit(jax.grad(scan_loss))(theta))
        gu = np.asarray(jax.jit(jax.grad(unrolled_loss))(theta))
        e = dict(finite=bool(np.isfinite(gs).all()), nonzero=bool(np.abs(gs).max() > 0),
                 scan_bitwise_unrolled=same_bits(gs, gu), scan_rel_unrolled=stats(rel_rows(gs.reshape(H * B, -1),
                                                                                           gu.reshape(H * B, -1))))
        e["pass"] = bool(e["finite"] and e["nonzero"] and e["scan_rel_unrolled"]["max"] <= 1e-5)
        ok &= e["pass"]
        res[name] = e
    res["H"], res["B"] = H, B
    res["pass"] = bool(ok)
    return res


# ---------------------------------------------------------------- timing

def _bench(f, reps):
    import jax
    jax.block_until_ready(f())
    jax.block_until_ready(f())
    t = time.perf_counter()
    out = None
    for _ in range(reps):
        out = f()
    jax.block_until_ready(out)
    return 1e3 * (time.perf_counter() - t) / reps


def _bare_kernels(a, reps, repeats):
    """k_kin / k_kin_vjp alone: the library's shims through ctypes on JAX's buffers (default stream), ms per launch."""
    import ctypes
    import glob
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    q, v, c = a[0], a[1], a[5]
    B = q.shape[0]
    outs = jax.block_until_ready(jax.jit(F.kinematics_forward)(q, v, c))  # (uploads the constants)
    rng = np.random.default_rng(0)
    cots = [jnp.asarray(rng.standard_normal(o.shape), jnp.float32) for o in outs]
    gq, gv = jax.block_until_ready(jax.jit(F.kinematics_backward)(q, v, c, *cots))
    lib = ctypes.CDLL(F.init()["path"])
    rt = ctypes.CDLL(sorted(glob.glob(os.path.join(os.path.dirname(jax.__file__), "..", "nvidia", "cuda_runtime",
                                                   "lib", "libcudart.so*")))[0])
    P = lambda x: ctypes.c_void_p(x.unsafe_buffer_pointer())  # noqa: E731
    tag = F.init()["forward_unit"]  # the unit with the forward / kinematics entries (dk_fwd when it has its own flags)
    kin, kvjp = getattr(lib, f"{tag}_kinematics"), getattr(lib, f"{tag}_kinematics_vjp")
    calls = dict(k_kin=lambda: kin(ctypes.c_void_p(0), B, P(q), P(v), P(c), *[P(o) for o in outs]),
                 k_kin_vjp=lambda: kvjp(ctypes.c_void_p(0), B, P(q), P(v), P(c), *[P(x) for x in cots],
                                        ctypes.c_void_p(0), ctypes.c_void_p(0), P(gq), P(gv)))
    res = {}
    for name, f in calls.items():
        for _ in range(5):
            assert f() == 0
        rt.cudaDeviceSynchronize()
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            for _ in range(reps):
                f()
            rt.cudaDeviceSynchronize()
            ts.append(1e3 * (time.perf_counter() - t) / reps)
        res[name] = dict(ms=float(np.median(ts)), all=ts)
    return res


def sec_timing(args):
    import jax
    import jax.numpy as jnp
    from nablaengine import jax_ffi as F
    F.init()
    a = _inputs(CORPORA[0])
    B = a[0].shape[0]
    fk = mjx_kinematics()
    rng = np.random.default_rng(17)
    step0 = jax.jit(lambda *x: F.step(*x))
    step1 = jax.jit(lambda *x: F.step(*x, kinematics=True))
    o1 = step1(*a)
    cots = [jnp.asarray(rng.standard_normal(x.shape), jnp.float32) for x in o1]
    fwd0 = jax.jit(lambda *x: F.forward(*x, tape=True))
    fwd1 = jax.jit(lambda *x: F.forward(*x, tape=True, kinematics=True))
    _, _, _, _, tp = fwd0(*a)
    q1, v1 = o1[0], o1[1]
    bwd0 = jax.jit(lambda tp: F.backward(tp, a[4], a[5], cots[0], cots[1]))
    bwd1 = jax.jit(lambda tp: F.backward(tp, a[4], a[5], cots[0], cots[1], kinematics=(q1, v1, *cots[3:])))

    def vjp0(*x):
        outs, f = jax.vjp(lambda *y: F.step(*y), *x)
        return outs, f(tuple(cots[:3]))

    def vjp1(*x):
        outs, f = jax.vjp(lambda *y: F.step(*y, kinematics=True), *x)
        return outs, f(tuple(cots))

    def kin_vjp(q, v, c):
        outs, f = jax.vjp(F.kinematics, q, v, c)
        return outs, f(tuple(cots[3:]))

    def mjx_vjp(q, v, c):
        outs, f = jax.vjp(fk, q, v, c)
        return outs, f(tuple(cots[3:]))

    j = dict(vjp0=jax.jit(vjp0), vjp1=jax.jit(vjp1), kin_fwd=jax.jit(F.kinematics_forward), kin_vjp=jax.jit(kin_vjp),
             mjx_fwd=jax.jit(fk), mjx_vjp=jax.jit(mjx_vjp))
    calls = (("step", lambda: step0(*a)), ("step_kinematics", lambda: step1(*a)), ("forward_tape", lambda: fwd0(*a)),
             ("forward_tape_kinematics", lambda: fwd1(*a)), ("backward", lambda: bwd0(tp)),
             ("backward_kinematics", lambda: bwd1(tp)), ("vjp", lambda: j["vjp0"](*a)),
             ("vjp_kinematics", lambda: j["vjp1"](*a)), ("kinematics_forward", lambda: j["kin_fwd"](q1, v1, a[5])),
             ("kinematics_forward_vjp", lambda: j["kin_vjp"](q1, v1, a[5])),
             ("mjx_forward", lambda: j["mjx_fwd"](q1, v1, a[5])), ("mjx_forward_vjp", lambda: j["mjx_vjp"](q1, v1, a[5])))
    allt = {n: [] for n, _ in calls}
    for _ in range(args.timing_repeats):
        for n, f in calls:
            allt[n].append(_bench(f, args.reps))
    med = {n: float(np.median(v)) for n, v in allt.items()}
    # device-bound: K calls in one lax.scan (each call's input the previous one's qpos_n), ms per call
    K = 50

    def scan_fn(kin_fn, vjp):
        def body(c, _):
            q, v, acc = c
            if vjp:
                outs, f = jax.vjp(kin_fn, q, v, a[5])
                g = f(tuple(cots[3:]))
                q, v = outs[0] + 1e-9 * g[0], v + 1e-9 * g[1]
            else:
                outs = kin_fn(q, v, a[5])
                q = outs[0]
            # every output into the carry: XLA drops the parts of MJX whose outputs are unused (the engine's FFI
            # call is never eliminated), which made MJX's forward look ~4x cheaper
            acc = acc + sum(o.reshape(o.shape[0], -1).sum(1) for o in outs[1:])
            return (q, v, acc), None
        return jax.jit(lambda q, v: jax.lax.scan(body, (q, v, jnp.zeros_like(q[:, 0])), None, length=K)[0])

    scans = dict(kinematics_forward=scan_fn(F.kinematics, False), kinematics_forward_vjp=scan_fn(F.kinematics, True),
                 mjx_forward=scan_fn(fk, False), mjx_forward_vjp=scan_fn(fk, True))
    scant = {n: [] for n in scans}
    for _ in range(args.timing_repeats):
        for n, f in scans.items():
            scant[n].append(_bench(lambda f=f: f(q1, v1), 3) / K)
    smed = {n: float(np.median(v)) for n, v in scant.items()}
    res = dict(B=B, iterations=F.ITERATIONS, reps=args.reps, repeats=args.timing_repeats, ms_per_call=med,
               ms_all=allt, gpu=_gpu_name(),
               extra=dict(step_forward_ms=med["step_kinematics"] - med["step"],
                          forward_tape_ms=med["forward_tape_kinematics"] - med["forward_tape"],
                          backward_ms=med["backward_kinematics"] - med["backward"],
                          vjp_ms=med["vjp_kinematics"] - med["vjp"]),
               in_scan_ms_per_call=smed, in_scan_all=scant, in_scan_calls=K,
               mjx_over_engine=dict(forward=smed["mjx_forward"] / smed["kinematics_forward"],
                                    forward_vjp=smed["mjx_forward_vjp"] / smed["kinematics_forward_vjp"],
                                    per_call_forward=med["mjx_forward"] / med["kinematics_forward"],
                                    per_call_forward_vjp=med["mjx_forward_vjp"] / med["kinematics_forward_vjp"]),
               bare_kernels=_bare_kernels(a, args.reps * 5, args.timing_repeats),
               note="ms_per_call: per call under jax.jit, each call dispatched from Python (its dispatch overhead "
                    "included: it dominates the kinematics' small calls and varies with the host's load); "
                    "in_scan_ms_per_call: K calls chained in one lax.scan (device time per call, the comparison with "
                    "MJX); the bare kernels back to back on the default stream; V100 numbers compare against MJX on "
                    "the same card only")
    res["pass"] = True  # reported, not gated
    print("timing", json.dumps({k: v for k, v in res.items() if k not in ("ms_all", "in_scan_all")}), flush=True)
    return res


def _gpu_name():
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                           os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                          stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()


RUN = dict(f64=sec_f64, mjx=sec_mjx, vjp=sec_vjp, step=sec_step, batch=sec_batch, timing=sec_timing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--section", default="", help="internal: run one section in this process, JSON to --out")
    ap.add_argument("--work", default="", help="internal: the run's work directory (f64 results, base build)")
    ap.add_argument("--f64-worlds", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=8, help="step: the remat scans' control steps")
    ap.add_argument("--remat-worlds", type=int, default=1024)
    ap.add_argument("--corpora", default=",".join(fixtures.MAIN_CORPORA), help="comma-separated corpus names")
    ap.add_argument("--kinematics-scan", action="store_true",
                    help="step, batch: also the scans through jax_ffi.kinematics alone (module docstring)")
    ap.add_argument("--reps", type=int, default=20, help="timing: calls per measurement")
    ap.add_argument("--timing-repeats", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--update", action="store_true",
                    help="replace only the run sections in an existing --out JSON (the others are kept)")
    ap.add_argument("--out", required=True)
    workdir.add_argument(ap)
    args = ap.parse_args()
    workdir.configure(args)
    CORPORA[:] = [c for c in args.corpora.split(",") if c]
    if args.section:
        r = RUN[args.section](args)
        Path(args.out).write_text(json.dumps(r, indent=1))
        return
    import jax
    work = workdir.work_dir("kinematics_check_")
    from nablaengine import jax_ffi as F
    lib = F.init()
    library = dict(path=provenance.rel(lib["path"]), key=lib["key"], arch=lib["arch"])
    res = dict(gpu=_gpu_name(), jax=jax.__version__, library=library, corpora=CORPORA, sections={})
    if args.update and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        assert old["gpu"] == res["gpu"], "--update: the existing results are from another GPU"
        res["sections"] = old["sections"]
        provenance.keep_sections(res, [(old, [s for s in old["sections"] if s not in args.sections.split(",")])])
    for sec in args.sections.split(","):
        part = work / f"{sec}.json"
        t = time.perf_counter()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--section", sec, "--out", str(part), "--work", str(work)]
        for k in ("f64_worlds", "horizon", "remat_worlds", "corpora", "reps", "timing_repeats", "timeout"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        cmd += ["--kinematics-scan"] if args.kinematics_scan else []
        env = dict(os.environ, JAX_ENABLE_X64="1") if sec == "f64" else None
        p = subprocess.run(cmd, timeout=args.timeout, env=env)
        if p.returncode or not part.exists():
            res["sections"][sec] = dict(error=f"exit code {p.returncode}", **{"pass": False})
            continue
        r = json.loads(part.read_text())
        r["seconds"] = time.perf_counter() - t
        res["sections"][sec] = r
        print(f"== {sec}: pass {r.get('pass')} ({r['seconds']:.0f} s)", flush=True)
    res["pass"] = bool(all(v.get("pass") for v in res["sections"].values()))
    res["provenance"] = provenance.provenance()  # commit, GPU, driver, nvcc, packages, MuJoCo Warp
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(pass_=res["pass"], sections={k: v.get("pass") for k, v in res["sections"].items()})))
    return 0 if res["pass"] else 1  # the exit status (a --section child exits 0: its parent reads the JSON)


if __name__ == "__main__":
    sys.exit(main())
