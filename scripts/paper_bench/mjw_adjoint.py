"""Parts `mjw_pr1535`, `mjw_pr1535_fix`: MuJoCo Warp forward and adjoint per control step with the analytic adjoint
of the unmerged google-deepmind/mujoco_warp pull request #1535, through nablaengine.reference.warp_physics._Sim (4 x
mjw.step per control step, the per-world DR of the benchmark problem, every capacity overflow warning masked).

PR #1535 as published (etaoxing/mujoco_warp @ 357a75d, the head of the pull request) and with etaoxing/mujoco_warp#1
(@ 4af093f, two public fixes): the tree on PYTHONPATH of a venv with mujoco 3.17.0 + warp-lang 1.18.0 and no
mujoco_warp installed (scripts/paper_bench/setup.sh).

Per corpus (one process per nworld; the inputs are copied into the graphs' source arrays), CUDA graphs captured once
(capture timed), then common.timeit rounds of graph replays + one synchronize:
  forward_graph       _Sim.forward: DR kernel, copies, 4 x mjw.step
  adjoint_graph       _Sim.backward: the taped forward recompute (wp.Tape over 4 x mjw.step under the PR's
                      backward_context) + tape.backward: the PR's adjoint-graph pattern
Cotangents: the corpus's stored cotangent 0 (ones where a corpus has none). --split: on the first corpus, the GPU time
(wp.timing) of the adjoint's taped forward and of tape.backward alone. --phase-graphs: the two halves of adjoint_graph
as two more graphs, timed like the others (same calls in the same order; the second replays on the tape of the first):
  adjoint_taped_forward   DR kernel, copies, wp.Tape over 4 x mjw.step under the PR's backward_context
  adjoint_tape_backward   tape.zero, the cotangent copies, tape.backward, the gradient copies and the ctrl sum

main(ext=...) takes an optional extension (more graphs on the same _Sim): an object with add_arguments(ap),
setup(sim, mjm, args, res) and graphs(sim, arrays, args) -> {name: (fn, name of a graph to run before it or None)}.

  PYTHONPATH=<tree>:src python scripts/paper_bench/mjw_adjoint.py --label mjw_pr1535 --out <dir>/x.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main(argv=None, ext=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--nconmax", type=int, default=64)
    ap.add_argument("--njmax", type=int, default=192)
    ap.add_argument("--ops", default="", help="comma-separated subset of the graphs (default: all that apply)")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--phase-graphs", action="store_true")
    if ext is not None:
        ext.add_arguments(ap)
    a = ap.parse_args(argv)

    import mujoco
    import numpy as np
    import warp as wp
    import mujoco_warp as mjw
    from nablaengine import fixtures, provenance
    from nablaengine.reference.warp_physics import SUBSTEPS, _accum, _Sim

    res = C.Result(a.out, part=a.label, nworld=a.nworld, iterations=a.iterations, substeps=SUBSTEPS,
                   nconmax=a.nconmax, njmax=a.njmax,
                   env=C.basic_env(("mujoco", "mujoco-warp", "warp-lang", "numpy")),
                   mujoco_warp=provenance.mujoco_warp(), provenance=provenance.provenance(),
                   mujoco_warp_file=C.rel(mjw.__file__), warp_version=wp.config.version)
    try:
        res.data["warp_cuda_toolkit"] = ".".join(map(str, wp.get_cuda_toolkit_version()))
    except Exception:  # noqa: BLE001
        pass
    mjm, mt = fixtures.load_model(C.FIXTURES)
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    mjm.opt.iterations = a.iterations
    n = a.nworld
    names = list(C.corpora(a.corpora))
    states = {k: C.load_states(k, nworld=n) for k in names}
    dev = wp.get_device("cuda:0")
    try:
        with wp.ScopedDevice(dev):
            mjw.enable_grad()
            t0 = time.perf_counter()
            sim = _Sim(mjm, n, np.asarray(mt["foot_pairs"]), fixtures.torso_id(mjm, mt), a.nconmax, a.njmax)
            if ext is not None:
                ext.setup(sim, mjm, a, res)
            res.data["setup_s"] = round(time.perf_counter() - t0, 3)
            z = lambda k: wp.zeros((n, k), dtype=float)  # noqa: E731
            a_q, a_v, a_w, a_u = z(mjm.nq), z(mjm.nv), z(mjm.nv), z(mjm.nu)
            fr, com = wp.zeros(n, dtype=float), z(3)
            o_q, o_v, o_w = z(mjm.nq), z(mjm.nv), z(mjm.nv)
            g_q, g_v = z(mjm.nq), z(mjm.nv)
            gq, gv, gu = z(mjm.nq), z(mjm.nv), z(mjm.nu)
            fns = {"forward_graph": lambda: sim.forward(a_q, a_v, a_w, a_u, fr, com, o_q, o_v, o_w),
                   "adjoint_graph": lambda: sim.backward(a_q, a_v, a_w, a_u, fr, com, g_q, g_v, gq, gv, gu)}
            before = {}  # graph -> the graph run (eagerly) before its first call and before its timing
            if a.phase_graphs:
                fns.update(phase_fns(sim, mjw, wp, SUBSTEPS, _accum, a_q, a_v, a_w, a_u, fr, com, g_q, g_v, gq, gv, gu))
                before["adjoint_tape_backward"] = "adjoint_taped_forward"
            if ext is not None:
                arrays = dict(qpos=a_q, qvel=a_v, warm=a_w, ctrl=a_u, friction=fr, com=com, out_qpos=o_q,
                              out_qvel=o_v, out_warm=o_w, cot_qpos=g_q, cot_qvel=g_v, grad_qpos=gq, grad_qvel=gv,
                              grad_ctrl=gu)
                for k, (fn, pre) in ext.graphs(sim, arrays, a).items():
                    fns[k] = fn
                    if pre:
                        before[k] = pre
            ops = list(fns)
            if a.ops:
                ops = [o for o in ops if o in a.ops.split(",")]
            graphs, first_s, capture_s = {}, {}, {}
            for ci, name in enumerate(names):
                st = states[name]
                key = f"{name}/nw{n}"
                try:
                    for arr, k in ((a_q, "qpos"), (a_v, "qvel"), (a_w, "qacc_warmstart"), (a_u, "ctrl"),
                                   (fr, "friction"), (com, "com")):
                        arr.assign(st[k])
                    if st["cot_qpos"] is not None:
                        g_q.assign(st["cot_qpos"])
                        g_v.assign(st["cot_qvel"])
                    else:
                        g_q.fill_(1.0)
                        g_v.fill_(1.0)
                    cell = dict(corpus=st["_source"], cotangent="corpus cot 0" if st["cot_qpos"] is not None else "ones")
                    for op in ops:
                        if op in before:  # the state this graph reads (for example stored per-substep data)
                            fns[before[op]]()
                            wp.synchronize()
                        if op not in graphs:
                            t = time.perf_counter()
                            fns[op]()  # kernel compile (or Warp cache load) + first run, eager
                            wp.synchronize()
                            first_s[op] = round(time.perf_counter() - t, 3)
                            t = time.perf_counter()
                            with wp.ScopedCapture() as cp:
                                fns[op]()
                            capture_s[op] = round(time.perf_counter() - t, 3)
                            graphs[op] = cp.graph
                        if op in before:  # this corpus's state for the graph
                            fns[before[op]]()
                        g = graphs[op]
                        cell[op] = C.timeit(lambda: wp.capture_launch(g), wp.synchronize,
                                            latency=0 if C.QUICK else 10)
                    cell["first_call_s"], cell["capture_s"] = dict(first_s), dict(capture_s)
                    wp.capture_launch(graphs[ops[0]])
                    wp.synchronize()
                    d = sim.df
                    stats = {}
                    for k in ("nefc", "solver_niter"):
                        if hasattr(d, k):
                            v = getattr(d, k).numpy().reshape(-1)
                            stats[k] = dict(mean=float(v.mean()), max=float(v.max()))
                    stats["finite_out_qvel_worlds"] = int(np.isfinite(o_v.numpy()).all(1).sum())
                    if any(op != "forward_graph" and op.startswith("adjoint") for op in graphs):
                        stats["finite_grad_worlds"] = int((np.isfinite(gq.numpy()).all(1)
                                                           & np.isfinite(gv.numpy()).all(1)
                                                           & np.isfinite(gu.numpy()).all(1)).sum())
                    cell["stats"] = stats
                    if a.split and ci == 0:
                        cell["adjoint_split_gpu_ms"] = split_ms(sim, mjw, wp, SUBSTEPS, a_q, a_v, a_w, a_u, fr,
                                                                com, g_q, g_v)
                    res.cell(key, cell)
                except Exception as e:  # noqa: BLE001
                    if C.is_oom(e):
                        C.record_oom(res, key, e, n, corpus=st["_source"])
                        break
                    res.error(key, e)
    except Exception as e:  # noqa: BLE001
        if C.is_oom(e):
            C.record_oom(res, f"nw{n}", e, n)
        else:
            res.error(f"nw{n}", e)
    return 1 if res.data["errors"] else 0


def phase_fns(sim, mjw, wp, nsub, accum, a_q, a_v, a_w, a_u, fr, com, g_q, g_v, gq, gv, gu):
    """_Sim.backward split into its taped forward recompute and its backward half (same calls, same order)."""
    D, holder = sim.D, {}

    def taped_forward():
        sim._dr(fr, com)
        wp.copy(D[0].qpos, a_q)
        wp.copy(D[0].qvel, a_v)
        wp.copy(D[0].qacc_warmstart, a_w)
        wp.copy(D[0].ctrl, a_u)
        tape = wp.Tape()
        with mjw.backward_context(sim.bc):
            with tape:
                for k in range(nsub):
                    mjw.step(sim.m, D[k], D[k + 1])
        holder["tape"] = tape

    def tape_backward():
        tape = holder["tape"]
        with mjw.backward_context(sim.bc):
            tape.zero()
            wp.copy(D[-1].qpos.grad, g_q)
            wp.copy(D[-1].qvel.grad, g_v)
            tape.backward()
        wp.copy(gq, D[0].qpos.grad)
        wp.copy(gv, D[0].qvel.grad)
        gu.zero_()
        for k in range(nsub):
            wp.launch(accum, dim=gu.shape, inputs=[D[k].ctrl.grad], outputs=[gu])

    return {"adjoint_taped_forward": taped_forward, "adjoint_tape_backward": tape_backward}


def split_ms(sim, mjw, wp, nsub, a_q, a_v, a_w, a_u, fr, com, g_q, g_v):
    """GPU time (sum of kernel times, wp.timing) of the adjoint's taped forward and of tape.backward alone."""
    D, holder = sim.D, {}

    def gpu_ms(fn):
        wp.synchronize()
        wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
        fn()
        wp.synchronize()
        r = wp.timing_end()
        return round(sum(x.elapsed for x in r), 3), len(r)

    def taped_forward():
        sim._dr(fr, com)
        wp.copy(D[0].qpos, a_q)
        wp.copy(D[0].qvel, a_v)
        wp.copy(D[0].qacc_warmstart, a_w)
        wp.copy(D[0].ctrl, a_u)
        with mjw.backward_context(sim.bc):
            tape = wp.Tape()
            with tape:
                for k in range(nsub):
                    mjw.step(sim.m, D[k], D[k + 1])
        holder["tape"] = tape

    def tape_backward():
        with mjw.backward_context(sim.bc):
            holder["tape"].zero()
            wp.copy(D[-1].qpos.grad, g_q)
            wp.copy(D[-1].qvel.grad, g_v)
            holder["tape"].backward()

    f, nf = gpu_ms(taped_forward)
    b, nb = gpu_ms(tape_backward)
    return dict(taped_forward=f, tape_backward_only=b, launches_forward=nf, launches_backward=nb)


if __name__ == "__main__":
    sys.exit(main())
