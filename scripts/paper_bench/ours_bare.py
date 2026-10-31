"""Part `ours_bare` (attribution): bare-kernel device times of our engine, without JAX (CUDA events around
back-to-back launches).

  step      scripts/engine_step_check.py (the forward unit's binary, split forward k_pre / k_post) at --nworld, 10
            iterations, 4 substeps, from the corpus: tape off and on (its T7 timing: three alternating runs of each,
            --reps launches per run) plus five more tape-off runs (--sweep-blocks-per-sm 0,..., the production launch,
            same binary); MuJoCo C check on 64 worlds only (sanity)
  adjoint   nablaengine.checks build / run (the adjoint check binary): ms_adjoint (k_adjoint on the tape, the
            corpus's stored cotangent 0) for the exact (clamp_kv 1, normal 0) and interface-default (clamp_kv 0,
            normal 3) variants, ROUNDS + 1 runs (round 0 dropped); its fused forward with tape (ms_forward_tape,
            k_substep: not the production forward) is recorded as such
Corpus: --corpus (default mf_s0; orchestrate.py passes the first corpus), which needs stored cotangents.

  PYTHONPATH=src python scripts/paper_bench/ours_bare.py --out <dir>/ours_bare.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

VARIANTS = {"exact": ["DK_ADJ_CLAMP_KV=1", "DK_ADJ_STAGE=3", "DK_ADJ_NORMAL=0"],
            "interface": ["DK_ADJ_CLAMP_KV=0", "DK_ADJ_STAGE=3", "DK_ADJ_NORMAL=3"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="mf_s0")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    import numpy as np
    from nablaengine import checks, cuda, fixtures, provenance, workdir

    out = Path(a.out)
    raw = out.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    res = C.Result(out, part="ours_bare", corpus=a.corpus, nworld=a.nworld, iterations=a.iterations, substeps=4,
                   reps=a.reps, env=C.basic_env(("numpy", "mujoco")), provenance=provenance.provenance(),
                   arch_flags=dict(forward=cuda.arch_flags("forward"), adjoint=cuda.arch_flags()))
    reps = max(5, a.reps // 5) if C.QUICK else a.reps
    # 1. the forward binary, tape off / on
    try:
        sj = raw / f"ours_bare_step_{a.corpus}.json"
        sweep = ",".join(["0"] * (C.ROUNDS))
        cmd = [sys.executable, str(C.REPO / "scripts" / "engine_step_check.py"), "--corpus", a.corpus, "--nworld",
               str(a.nworld), "--iterations", str(a.iterations), "--substeps", "4", "--check", "64", "--reps",
               str(reps), "--tape", "--sweep-blocks-per-sm", sweep, "--timeout", str(a.timeout), "--out", str(sj)]
        env = dict(os.environ, DK_FIXTURES=str(C.FIXTURES))
        p = subprocess.run(cmd, cwd=str(C.REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=a.timeout + 300)
        (raw / f"ours_bare_step_{a.corpus}.log").write_text(p.stdout)
        if p.returncode != 0:
            raise RuntimeError(f"engine_step_check exit {p.returncode}: {p.stdout[-1500:]}")
        r = json.loads(sj.read_text())
        t7 = (r.get("tape") or {}).get("T7_timing") or {}
        off = list(t7.get("ms_tape_off") or []) + [x["ms"] for x in r.get("blocks_per_sm_sweep") or []]
        on = list(t7.get("ms_tape_on") or [])
        res.cell("step", dict(
            tape0=dict(ms=C.stats_ms(np.asarray(off) / 1e3), runs=len(off), launches_per_run=reps),
            tape1=dict(ms=C.stats_ms(np.asarray(on) / 1e3), runs=len(on), launches_per_run=reps),
            main_run_ms_tape1=r.get("ms"), blocks_per_sm=r.get("blocks_per_sm"),
            blocks_per_sm_post=r.get("blocks_per_sm_post"), arch_flags=r.get("arch_flags"),
            forward_defines=r.get("forward_defines"), ours_vs_mjw_qvel=r.get("ours_vs_mjw_qvel"),
            supported_share=r.get("supported_share"), raw=C.rel(sj)))
    except Exception as e:  # noqa: BLE001
        res.error("step", e)
    # 2. the adjoint binary
    try:
        os.environ["DK_FIXTURES"] = str(C.FIXTURES)
        E = checks
        m, mt = fixtures.load_model(C.FIXTURES)
        z = fixtures.load_corpus(a.corpus, adjoint=True)
        work = workdir.work_dir("paper_bench_adj_")
        E.write_inputs(work, E.corpus_inputs(z))
        for name, defs in VARIANTS.items():
            try:
                binary = E.build(m, mt, work, f"adj_{name}", defs)
                rows = []
                for _ in range(C.ROUNDS + 1):
                    rr, _o = E.run(binary, work, m, a.nworld, a.iterations, 4, 1, reps, False, a.timeout)
                    rows.append(rr)
                rows = rows[1:]
                res.cell(f"adjoint/{name}", dict(
                    defines=defs, adjoint=dict(ms=C.stats_ms(np.asarray([x["ms_adjoint"] for x in rows]) / 1e3),
                                               runs=len(rows), launches_per_run=reps),
                    fused_forward_tape_not_production=dict(
                        ms=C.stats_ms(np.asarray([x["ms_forward_tape"] for x in rows]) / 1e3)),
                    worlds_per_sm_adjoint=rows[0].get("blocks_per_sm_adjoint"),
                    sizeof_adj=rows[0].get("sizeof_adj"), forward_build=rows[0].get("forward_build")))
            except Exception as e:  # noqa: BLE001
                res.error(f"adjoint/{name}", e)
    except Exception as e:  # noqa: BLE001
        res.error("adjoint", e)
    return 1 if res.data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
