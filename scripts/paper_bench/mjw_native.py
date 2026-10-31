"""Parts `mjw_native`, `mjw_native_nw*`: native MuJoCo Warp forward (PyPI mujoco-warp 3.17.0) per control
step on the mjlab G1, from corpus states.

No nablaengine imports, only MuJoCo Warp's public API: mujoco.MjModel, mjw.put_model(batch_sizes=...), mjw.put_data, mjw.step, wp.ScopedCapture.
  - model: fixtures/mjlab/g1_train.mjb, opt.iterations = --iterations (10), sensors disabled (mjDSBL_SENSOR), every other
    option as stored (Newton, ls 8, tolerance as stored; MuJoCo Warp applies its own floor natively); the
    ITERATIONS / LS_ITERATIONS overflow warnings masked as in upstream's own G1 benchmark
    (m.opt.warn_overflow &= ~(ITERATIONS | LS_ITERATIONS)); graph_conditional as put_model sets it (True)
  - per-world DR written once on the host into the batched model arrays (put_model batch_sizes pair_friction /
    body_ipos: foot-floor pair friction [:2], torso body_ipos += com)
  - capacity: "procedure" (the documented sizing rule of benchmarks/, benchmarks/data/mjw_capacity.json:
    naconmax = ceil(nworld mu (1 + margin)) + c_max, nconmax = ceil(mu (1 + margin)), njmax = the smallest MuJoCo
    Warp valid size >= r_max, with mu the largest per-substep mean of max(broadphase candidates, contacts) per world,
    c_max the most contacts of one world and r_max the most constraint rows of one world over all corpora; read from
    that file when it exists, else the embedded values measured with MuJoCo Warp 3.17.0), "matched"
    nconmax 64 / njmax 192 per world (our engine's capacity), and "default" (put_data's own defaults); a capacity
    overflow (NEFC, NARROWPHASE, BROADPHASE, ...) is recorded and that capacity is not timed
  - timing: one CUDA graph holding [copy the corpus inputs into Data; 4 x mjw.step] (every replay starts from the same
    states), graph capture timed; warm-up, then rounds of N replays + one synchronize (common.timeit)
One process per nworld; all corpora in the same process (the inputs are copied into the graph's source arrays).

  python scripts/paper_bench/mjw_native.py --out <dir>/mjw_native_nw4096.json --nworld 4096
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CAPACITY_BITS = ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "NVMAX", "EPA_HORIZON")
# the documented capacity procedure's inputs for the mjlab G1 (benchmarks/data/mjw_capacity.json, problems.mjlab,
# measured with MuJoCo Warp 3.17.0 over mf_s0..mf_s3 at 10 iterations, 2026-10-29)
PROCEDURE_DEFAULT = dict(pool_mean_max=18.487548828125, contacts_world_max=30, rows_max=114, margin=0.25)
CAPACITY_FILE = "benchmarks/data/mjw_capacity.json"


def procedure_capacity(nworld):
    """nconmax / naconmax / njmax of the documented sizing rule at nworld (see the module docstring)."""
    import json
    import math
    src, v = "embedded (benchmarks/data/mjw_capacity.json of 2026-10-29)", dict(PROCEDURE_DEFAULT)
    f = C.REPO / CAPACITY_FILE
    if f.exists():
        try:
            pr = json.loads(f.read_text())["problems"]["mjlab"]
            v = {k: pr[k] for k in PROCEDURE_DEFAULT}
            src = CAPACITY_FILE
        except (KeyError, ValueError) as e:
            src += f"; {CAPACITY_FILE} unreadable ({e!r:.100})"
    valid = [(2 + i % 2) * 2 ** (i // 2 + 3) for i in range(19)]  # MuJoCo Warp's valid sizes 16, 24, 32, 48, ...
    mu = float(v["pool_mean_max"]) * (1 + float(v["margin"]))
    njmax = next((x for x in valid if x >= int(v["rows_max"])), int(v["rows_max"]))
    return dict(nconmax=int(math.ceil(mu)), naconmax=int(math.ceil(nworld * mu)) + int(v["contacts_world_max"]),
                njmax=njmax), dict(source=src, inputs=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--capacity", default="procedure,matched,default",
                    help="comma-separated: procedure (the documented sizing rule), matched (64/192), default")
    a = ap.parse_args()

    import mujoco
    import numpy as np
    import warp as wp
    import mujoco_warp as mjw
    try:
        OT = mjw.OverflowType
    except AttributeError:
        from mujoco_warp._src.types import OverflowType as OT

    res = C.Result(a.out, part="mjw_native", nworld=a.nworld, iterations=a.iterations, substeps=a.substeps,
                   env=C.basic_env(("mujoco", "mujoco-warp", "warp-lang", "numpy")),
                   mujoco_warp_file=C.rel(mjw.__file__), warp_version=wp.config.version)
    try:
        res.data["warp_cuda_toolkit"] = ".".join(map(str, wp.get_cuda_toolkit_version()))
        res.data["warp_cuda_driver"] = ".".join(map(str, wp.get_cuda_driver_version()))
    except Exception:  # noqa: BLE001
        pass
    meta = C.meta()
    mjm = mujoco.MjModel.from_binary_path(str(C.FIXTURES / "g1_train.mjb"))
    mjm.opt.iterations = a.iterations
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    torso = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, meta["torso_body"])
    foot = np.asarray(meta["foot_pairs"], int)
    n = a.nworld
    names = list(C.corpora(a.corpora))
    states = {k: C.load_states(k, nworld=n) for k in names}
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    t0 = time.perf_counter()
    m = mjw.put_model(mjm, batch_sizes={"pair_friction": n, "body_ipos": n})
    m.opt.warn_overflow = int(m.opt.warn_overflow) & ~int(OT.ITERATIONS | OT.LS_ITERATIONS)
    put_model_s = time.perf_counter() - t0
    res.data["opt"] = {k: str(getattr(m.opt, k)) for k in ("iterations", "ls_iterations", "tolerance", "ls_tolerance",
                                                           "solver", "cone", "integrator", "is_sparse",
                                                           "graph_conditional", "warn_overflow") if hasattr(m.opt, k)}
    from mujoco_warp._src import io as mjw_io  # the defaults put_data uses (recorded only)
    default_cap = dict(nconmax=int(mjw_io._default_nconmax(mjm, mjd)), njmax=int(mjw_io._default_njmax(mjm, mjd)))
    res.data["default_capacity"] = default_cap
    proc, proc_src = procedure_capacity(n)
    res.data["procedure_capacity"] = dict(proc, **proc_src)
    caps = {"procedure": proc, "matched": dict(nconmax=64, njmax=192), "default": dict(nconmax=None, njmax=None)}

    pf0 = mjm.pair_friction[None].astype(np.float32)
    bi0 = mjm.body_ipos[None].astype(np.float32)

    def set_dr(st):
        pf = np.tile(pf0, (n, 1, 1))
        pf[:, foot, 0] = st["friction"][:, None]
        pf[:, foot, 1] = st["friction"][:, None]
        bi = np.tile(bi0, (n, 1, 1))
        bi[:, torso] += st["com"]
        m.pair_friction.assign(pf)
        m.body_ipos.assign(bi)

    for cap in a.capacity.split(","):
        key0 = f"{cap}/nw{n}"
        try:
            kw = {k: v for k, v in caps[cap].items() if v is not None}
            d = mjw.put_data(mjm, mjd, nworld=n, **kw)
            used = dict(nconmax=kw.get("nconmax", default_cap["nconmax"]), njmax=kw.get("njmax", default_cap["njmax"]),
                        naconmax=kw.get("naconmax", n * kw.get("nconmax", default_cap["nconmax"])))
            iq, iv, iw, iu = (wp.zeros((n, k), dtype=float) for k in (mjm.nq, mjm.nv, mjm.nv, mjm.nu))

            def ctrl_step():
                wp.copy(d.qpos, iq)
                wp.copy(d.qvel, iv)
                wp.copy(d.qacc_warmstart, iw)
                wp.copy(d.ctrl, iu)
                for _ in range(a.substeps):
                    mjw.step(m, d)

            graph, capture_s, first_s = None, None, None
            for name in names:
                st = states[name]
                key = f"{name}/{key0}"
                try:
                    set_dr(st)
                    for arr, k in ((iq, "qpos"), (iv, "qvel"), (iw, "qacc_warmstart"), (iu, "ctrl")):
                        arr.assign(st[k])
                    if first_s is None:  # kernel compile (or Warp kernel cache load) + first run
                        t = time.perf_counter()
                        ctrl_step()
                        wp.synchronize()
                        first_s = time.perf_counter() - t
                    d.overflow.zero_()
                    ctrl_step()
                    wp.synchronize()
                    ov = d.overflow.numpy().reshape(-1).astype(np.int64)
                    bits = int(np.bitwise_or.reduce(ov)) if ov.size else 0
                    cap_bits = [b for b in CAPACITY_BITS if hasattr(OT, b) and bits & int(getattr(OT, b))]
                    stats = dict(overflow_bits_or=bits,
                                 overflow_flags=[f.name for f in OT if f.value and f.name != "ALL" and bits & f.value],
                                 capacity_overflow=cap_bits,
                                 capacity_overflow_worlds=int(sum(1 for x in ov if any(
                                     x & int(getattr(OT, b)) for b in CAPACITY_BITS if hasattr(OT, b)))))
                    for k in ("nefc", "solver_niter"):
                        if hasattr(d, k):
                            v = getattr(d, k).numpy().reshape(-1)
                            stats[k] = dict(mean=float(v.mean()), max=float(v.max()))
                    stats["nacon_total"] = int(d.nacon.numpy()[0])
                    stats["finite_worlds"] = int(np.isfinite(d.qvel.numpy()).all(1).sum())
                    cell = dict(corpus=st["_source"], capacity=cap, nconmax=used["nconmax"], njmax=used["njmax"],
                                naconmax=used["naconmax"], stats_last_substep=stats)
                    if cap_bits:
                        cell["skipped"] = f"capacity overflow {cap_bits} (not timed)"
                        res.cell(key, cell)
                        continue
                    if graph is None:
                        t = time.perf_counter()
                        with wp.ScopedCapture() as cp:
                            ctrl_step()
                        capture_s = time.perf_counter() - t
                        graph = cp.graph
                    r = C.timeit(lambda: wp.capture_launch(graph), wp.synchronize, latency=0 if C.QUICK else 10)
                    cell.update(graph_ctrlstep=r, first_call_s=round(first_s, 3), capture_s=round(capture_s, 3),
                                put_model_s=round(put_model_s, 3))
                    res.cell(key, cell)
                except Exception as e:  # noqa: BLE001
                    if C.is_oom(e):
                        C.record_oom(res, key, e, n, corpus=st["_source"], capacity=cap)
                        break
                    res.error(key, e)
            del d, graph
        except Exception as e:  # noqa: BLE001
            if C.is_oom(e):
                C.record_oom(res, key0, e, n)
            else:
                res.error(key0, e)
    return 1 if res.data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
