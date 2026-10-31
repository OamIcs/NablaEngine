"""Markdown summary of a paper timing kit run (scripts/paper_bench/README.md).

  python scripts/paper_bench/summarize.py <out_dir>/paper_bench.json [--out <out_dir>/paper_bench.md]

table(job) is also called by orchestrate.py: the flat headline rows stored in paper_bench.json ("table"). Times are
ms per control step (4 substeps of 5 ms, 10 Newton iterations; MJX Menagerie: 5 x 4 ms at its own options), median
over the kept rounds with p10-p90; Jacobian rows also per world. A local_parts.py next to this file (common.LOCAL) may
add engine labels (ENGINE), rows (table_rows(job, rows, section, helpers)), repeat cells (REPEAT_CELLS) and an
upgrade(job) of older result files.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LOCAL = C.LOCAL
MODES = {"exact": "exact: clamp_kv=1, normal=0", "interface": "interface defaults: clamp_kv=0, normal=3"}
ENGINE = {
    "ours": "ours (NablaEngine, jax_ffi under jit)",
    "mjw": "MuJoCo Warp 3.17.0 (PyPI, native)",
    "mjw_pr1535": "MuJoCo Warp + unmerged PR #1535 @357a75d (as published)",
    "mjw_pr1535_fix": "MuJoCo Warp + PR #1535 + etaoxing/mujoco_warp#1 @4af093f (two public fixes)",
    "mjx": "MJX 3.17 (impl='jax', native)",
    "mjc": "MuJoCo C 3.17 (CPU, float64, native)",
    "genesis": "Genesis 1.4.5 (native)",
    **dict(getattr(LOCAL, "ENGINE", {}) or {}),
}


def _res(job, part):
    return ((job.get("parts") or {}).get(part) or {}).get("result") or {}


def _ms(x):
    if not isinstance(x, dict):
        return None
    m = x.get("ms") if isinstance(x.get("ms"), dict) else None
    return m


def _row(rows, section, engine, op, corpus, nworld, timing, note=None, **extra):
    m = _ms(timing)
    r = dict(section=section, engine=engine, op=op, corpus=corpus, nworld=nworld,
             ms_median=m["median"] if m else None, p10=m["p10"] if m else None, p90=m["p90"] if m else None)
    if note:
        r["note"] = note
    r.update(extra)
    rows.append(r)


def _upgrade(job):
    if LOCAL is not None and hasattr(LOCAL, "upgrade"):
        LOCAL.upgrade(job)
    return job


def _local_rows(job, rows, section):
    if LOCAL is not None and hasattr(LOCAL, "table_rows"):
        LOCAL.table_rows(job, rows, section, dict(row=_row, res=_res, ms=_ms))


def table(job):
    _upgrade(job)
    rows = []
    corpora = job.get("corpora") or []
    # ours
    o = _res(job, "ours").get("cells") or {}
    for c in corpora:
        cell = o.get(f"{c}/nw4096")
        if not cell:
            continue
        _row(rows, "forward", "ours", "fwd (no tape)", c, 4096, cell.get("fwd"))
        _row(rows, "forward", "ours", "fwd_tape (with tape)", c, 4096, cell.get("fwd_tape"))
        for mode, lab in MODES.items():
            mm = cell.get(mode) or {}
            _row(rows, "gradient", "ours", f"bwd ({mode})", c, 4096, mm.get("bwd"), lab)
            _row(rows, "gradient", "ours", f"vjp = fwd_tape + bwd, one jit ({mode})", c, 4096, mm.get("vjp"), lab)
            for k, what in (("jac", "fwd_tape + 71 bwd"), ("jac_bwd", "71 bwd on one tape")):
                if mm.get(k):
                    _row(rows, "jacobian", "ours", f"{what} ({mode})", c, 4096, mm.get(k), lab,
                         us_per_world=mm[k].get("us_per_world"))
    # MuJoCo Warp native
    w = _res(job, "mjw_native").get("cells") or {}
    for c in corpora:
        for cap in ("procedure", "matched", "default"):
            cell = w.get(f"{c}/{cap}/nw4096")
            if not cell:
                continue
            lab = (f"naconmax {cell.get('naconmax')} (nconmax {cell.get('nconmax')}) / njmax {cell.get('njmax')}"
                   + ("; the documented sizing rule (native headline)" if cap == "procedure" else "")
                   + ("; our engine's capacity" if cap == "matched" else "")
                   + ("; put_data's defaults" if cap == "default" else ""))
            extra = {}
            if cell.get("skipped"):
                st = cell.get("stats_last_substep") or {}
                lab += "; capacity overflow: not timed"
                extra["cell"] = (f"{'+'.join(st.get('capacity_overflow') or [])} overflow in "
                                 f"{st.get('capacity_overflow_worlds')} worlds")
            _row(rows, "forward", "mjw", f"forward graph ({cap} capacity)", c, 4096, cell.get("graph_ctrlstep"), lab,
                 **extra)
        if w.get(f"{c}/matched/nw4096"):
            rows.append(dict(section="gradient", engine="mjw", op="backward", corpus=c, nworld=4096,
                             ms_median=None, note="none in any release (no timing)"))
    # PR #1535
    for part in ("mjw_pr1535", "mjw_pr1535_fix"):
        cells = _res(job, part).get("cells") or {}
        for c in corpora:
            cell = cells.get(f"{c}/nw4096")
            if not cell:
                continue
            if cell.get("skipped"):
                rows.append(dict(section="gradient", engine=part, op="forward / adjoint", corpus=c, nworld=4096,
                                 ms_median=None, cell=f"skipped: {cell['skipped']}"))
                continue
            f, b = cell.get("forward_graph"), cell.get("adjoint_graph")
            _row(rows, "forward", part, "forward graph", c, 4096, f)
            _row(rows, "gradient", part, "adjoint graph (its own taped forward recompute + tape.backward)", c, 4096,
                 b, "the gradient from the input state; compare with ours vjp")
            if _ms(f) and _ms(b):
                rows.append(dict(section="gradient", engine=part, op="forward + adjoint (sum of medians)", corpus=c,
                                 nworld=4096, ms_median=round(_ms(f)["median"] + _ms(b)["median"], 5),
                                 note="a differentiated rollout: the forward, then the adjoint graph"))
    _local_rows(job, rows, "gradient")
    # MJX
    x = _res(job, "mjx").get("cells") or {}
    for c in corpora:
        cell = x.get(f"mjlab/{c}/nw4096")
        if cell:
            _row(rows, "forward", "mjx", "forward (jit(vmap(scan)))", c, 4096, cell.get("forward"))
    xt = _res(job, "mjx_triton").get("cells") or {}
    for c in corpora:
        cell = xt.get(f"mjlab/{c}/nw4096")
        if cell:
            _row(rows, "forward", "mjx", "forward, XLA_FLAGS=--xla_gpu_triton_gemm_any=true", c, 4096,
                 cell.get("forward"))
    for k, cell in x.items():
        if k.startswith("menagerie/"):
            _row(rows, "forward_menagerie", "mjx", "forward, Menagerie unitree_g1/scene_mjx.xml native options",
                 "native_states", 4096, cell.get("forward"),
                 f"{(cell.get('model') or {}).get('substeps')} substeps of {(cell.get('model') or {}).get('timestep')} s")
        if k.startswith("jacfwd/"):
            if cell.get("skipped"):
                rows.append(dict(section="jacobian", engine="mjx", op="forward-mode Jacobian (native)",
                                 corpus=k.split("/")[1], nworld=k.split("/")[2], ms_median=None,
                                 note=f"skipped: {cell['skipped']}"))
            else:
                _row(rows, "jacobian", "mjx", "forward-mode Jacobian (native), 99 jvp directions", cell["corpus"]["name"],
                     cell.get("B"), cell.get("jacobian"), f"chunk {cell.get('chunk')}",
                     us_per_world=round(1e3 * cell["ms_per_world"], 3) if cell.get("ms_per_world") else None)
    rv = x.get("reverse_mode")
    if rv:
        rows.append(dict(section="gradient", engine="mjx", op="reverse mode", corpus=None, nworld=rv.get("B"),
                         ms_median=None, note=("supported" if rv.get("supported") else
                                               f"not supported natively: {rv.get('type')}: {rv.get('message')}")))
    # Genesis
    g = _res(job, "genesis").get("cells") or {}
    for c in corpora:
        cell = g.get(f"{c}/nw4096")
        if cell:
            _row(rows, "forward", "genesis", "forward (state load + scene.step(), 4 substeps)", c, 4096,
                 cell.get("forward"))
        cell = g.get(f"{c}/performance/nw4096")
        if cell:
            _row(rows, "forward", "genesis", "forward, gs.init(performance_mode=True)", c, 4096, cell.get("forward"))
    # MuJoCo C
    mc = _res(job, "mjc")
    for c in corpora:
        cell = (mc.get("cells") or {}).get(f"{c}/nw4096")
        if not cell:
            continue
        _row(rows, "forward", "mjc", f"forward, Batch nthread {cell.get('nthread')}", c, 4096, cell.get("forward"))
        if cell.get("forward_1t"):
            _row(rows, "forward", "mjc", f"forward, 1 thread, {cell['forward_1t'].get('B')} worlds (not 4096)", c,
                 cell["forward_1t"].get("B"), cell["forward_1t"],
                 us_per_world_substep=cell["forward_1t"].get("us_per_world_substep"))
        for k, what in (("fd_forward", "forward"), ("fd_centered", "centered")):
            for sfx, how in (("_full", "measured"), ("", "per-world rate"), ("_1t", "1 thread")):
                r = cell.get(k + sfx)
                if not r:
                    continue
                note = f"nthread {r.get('nthread', cell.get('nthread'))}; {how}"
                if sfx == "":
                    note += (f"; at 4096 by linear extrapolation (not measured): "
                             f"{(r.get('extrapolated_ms_at_nworld') or {}).get('ms')} ms")
                _row(rows, "jacobian", "mjc", f"FD Jacobian, chained mjd_transitionFD, {what}, eps {r.get('eps')}", c,
                     r.get("B"), r, note,
                     us_per_world=round(1e3 * r["ms_per_world"], 3) if r.get("ms_per_world") else None)
    # sweeps
    first = corpora[0] if corpora else None
    for k, cell in o.items():
        if k.startswith("sweep/"):
            nw = int(k.split("nw")[-1])
            if cell.get("skipped"):
                rows.append(dict(section="sweep", engine="ours", op="fwd / vjp", corpus=first, nworld=nw,
                                 ms_median=None, note=f"skipped: {cell['skipped']}"))
                continue
            _row(rows, "sweep", "ours", "fwd", first, nw, cell.get("fwd"))
            _row(rows, "sweep", "ours", "vjp (exact)", first, nw, cell.get("vjp"))
    for name, part in (job.get("parts") or {}).items():
        res = part.get("result") or {}
        cells = res.get("cells") or {}
        nw = res.get("nworld")
        if name == "mjw_native" or name.startswith("mjw_native_nw"):
            for cap in ("procedure", "matched"):
                cell = cells.get(f"{first}/{cap}/nw{nw}")
                op = f"forward graph ({cap} capacity)"
                if cell and cell.get("skipped"):
                    rows.append(dict(section="sweep", engine="mjw", op=op, corpus=first,
                                     nworld=nw, ms_median=None, note=f"skipped: {cell['skipped']}"))
                elif cell:
                    _row(rows, "sweep", "mjw", op, first, nw, cell.get("graph_ctrlstep"))
                elif (cells.get(f"{cap}/nw{nw}") or {}).get("skipped"):
                    rows.append(dict(section="sweep", engine="mjw", op=op, corpus=first,
                                     nworld=nw, ms_median=None, note=f"skipped: {cells[f'{cap}/nw{nw}']['skipped']}"))
        if part.get("status") in ("timeout", "skipped") and name.startswith("mjw_native_nw"):
            rows.append(dict(section="sweep", engine="mjw", op="forward graph (procedure capacity)", corpus=first,
                             nworld=int(name.split("_nw")[-1]), ms_median=None,
                             note=f"{part.get('status')}: {part.get('reason', '')}"))
    _local_rows(job, rows, "sweep")
    # bare kernels
    b = _res(job, "ours_bare").get("cells") or {}
    bc = _res(job, "ours_bare").get("corpus")
    if b.get("step"):
        _row(rows, "bare", "ours", "forward kernels, tape off (CUDA events)", bc, 4096, b["step"].get("tape0"))
        _row(rows, "bare", "ours", "forward kernels, tape on (CUDA events)", bc, 4096, b["step"].get("tape1"))
    for mode in MODES:
        r = b.get(f"adjoint/{mode}")
        if r:
            _row(rows, "bare", "ours", f"k_adjoint ({mode}, CUDA events)", bc, 4096, r.get("adjoint"))
    return rows


REPEAT_CELLS = (  # (part, engine, op label, cell key pattern, path in the cell)
    ("ours", "ours", "fwd (no tape)", "{c}/nw4096", ("fwd",)),
    ("ours", "ours", "vjp (exact)", "{c}/nw4096", ("exact", "vjp")),
    ("ours", "ours", "vjp (interface)", "{c}/nw4096", ("interface", "vjp")),
    ("mjw_native", "mjw", "forward graph (procedure capacity)", "{c}/procedure/nw4096", ("graph_ctrlstep",)),
    ("mjw_native", "mjw", "forward graph (matched capacity)", "{c}/matched/nw4096", ("graph_ctrlstep",)),
    ("mjw_pr1535", "mjw_pr1535", "forward graph", "{c}/nw4096", ("forward_graph",)),
    ("mjw_pr1535", "mjw_pr1535", "adjoint graph", "{c}/nw4096", ("adjoint_graph",)),
    ("mjw_pr1535_fix", "mjw_pr1535_fix", "forward graph", "{c}/nw4096", ("forward_graph",)),
    ("mjw_pr1535_fix", "mjw_pr1535_fix", "adjoint graph", "{c}/nw4096", ("adjoint_graph",)),
    ("mjx", "mjx", "forward", "mjlab/{c}/nw4096", ("forward",)),
    ("mjc", "mjc", "forward (Batch)", "{c}/nw4096", ("forward",)),
    ("genesis", "genesis", "forward", "{c}/nw4096", ("forward",)),
    ("genesis", "genesis", "forward, performance_mode=True", "{c}/performance/nw4096", ("forward",)),
) + tuple(getattr(LOCAL, "REPEAT_CELLS", ()) or ())


def repeats(job):
    """Headline cells measured in several processes (the main part + rep<r>_<part>): per-process medians."""
    parts = job.get("parts") or {}
    heads = sorted({k.split("/")[0] for name, p in parts.items() if name.startswith("rep")
                    for k in ((p.get("result") or {}).get("cells") or {})} - {"mjlab"})
    heads += sorted({k.split("/")[1] for name, p in parts.items() if name.startswith("rep") and name.endswith("mjx")
                     for k in ((p.get("result") or {}).get("cells") or {})} - set(heads))
    out = []
    for c in heads:
        for part, eng, op, pat, path in REPEAT_CELLS:
            vals = []
            for name in [part] + sorted(n for n in parts if n.startswith("rep") and n.split("_", 1)[1] == part):
                x = ((parts.get(name) or {}).get("result") or {}).get("cells", {}).get(pat.format(c=c))
                for k in path:
                    x = x.get(k) if isinstance(x, dict) else None
                m = _ms(x)
                if m:
                    vals.append(m["median"])
            if len(vals) > 1:
                srt = sorted(vals)
                out.append(dict(engine=eng, op=op, corpus=c, processes=len(vals), medians=vals,
                                median=srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1]
                                                                                      + srt[len(srt) // 2]),
                                spread=round(srt[-1] / srt[0] - 1, 5)))
    return out


def fmt(r):
    if r.get("ms_median") is None:
        return r.get("cell") or "-"
    s = f"{r['ms_median']:.3f}"
    if r.get("p10") is not None:
        s += f" [{r['p10']:.3f}-{r['p90']:.3f}]"
    return s


def _corpus(c, kinds):
    k = kinds.get(c, "benchmark")
    return c if k == "benchmark" else f"{c} ({k})"


def markdown(job):
    _upgrade(job)
    rows = job.get("table") or table(job)
    sysi = job.get("system") or {}
    gpu = sysi.get("gpu") or {}
    cpu = sysi.get("cpu") or {}
    prov = job.get("provenance") or {}
    dk = prov.get("nablaengine") or {}
    setup = job.get("setup") or {}
    ck = job.get("corpus_kind") or {}
    corp_s = ", ".join(_corpus(c, ck) for c in job.get("corpora") or [])
    rev = (f"commit {dk['commit']} (dirty: {dk.get('dirty')})" if dk.get("commit")
           else f"commit: {dk.get('commit_note') or 'not recorded'}")
    L = [f"# Paper timing kit: {gpu.get('name')} (driver {gpu.get('driver')})", ""]
    if job.get("quick"):
        L += ["**QUICK run (PAPER_BENCH_QUICK=1): fewer rounds, not for publication.**", ""]
    L += [f"- started {job.get('started')}, finished {job.get('finished')}, {job.get('seconds')} s in all; "
          f"setup: {job.get('setup_status')}",
          f"- NablaEngine {dk.get('version')} {rev}; nvcc "
          f"{prov.get('nvcc')}; arch {prov.get('arch')}",
          f"- CPU {cpu.get('model')}: {cpu.get('allocated')} allocated of {cpu.get('logical_cpus')} logical "
          f"(affinity {cpu.get('affinity')})",
          f"- warp-lang wheel of the native venvs: {setup.get('warp_wheel')} "
          f"({((setup.get('venvs') or {}).get('venv_mjw') or {}).get('warp-lang')})",
          f"- packages of the main environment: {json.dumps(sysi.get('packages'))}",
          f"- corpora: {corp_s}"
          f"; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)",
          *([f"- **headline corpora missing: {', '.join(job['headline_missing'])}** (benchmarks/data)"]
            if job.get("headline_missing") else []),
          *([f"- **{job['warp_wheel_note']}**"] if job.get("warp_wheel_note") else []),
          *[f"- **{n}: {_res(job, n).get('version_warning')}**" for n in (job.get("parts") or {})
            if _res(job, n).get("version_warning")],
          f"- MuJoCo C threads: {(cpu.get('cpu_engine_threads') or {}).get('threads')} "
          f"({(cpu.get('cpu_engine_threads') or {}).get('rule')}); scheduler CPU counts "
          f"{json.dumps(cpu.get('scheduler_cpu_counts'))}", "",
          "What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = "
          "the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). "
          "MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph "
          "alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of "
          "a differentiated rollout (the forward, then the adjoint). Jacobian rows: "
          "the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 "
          "forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the "
          "per-world-rate rows is a linear extrapolation, the measured rows are labelled).", "",
          "ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.", ""]
    corpora = job.get("corpora") or []
    for sec, title in (("forward", "Forward, 4096 worlds (mjlab G1)"), ("gradient", "Gradient, 4096 worlds"),
                       ("jacobian", "Jacobian 71 x 99 (per world: us/world)"),
                       ("forward_menagerie", "MJX on Menagerie unitree_g1 (its native options)")):
        sel = [r for r in rows if r["section"] == sec]
        if not sel:
            continue
        L += [f"## {title}", ""]
        if sec in ("forward", "gradient"):
            L += ["| engine | op | " + " | ".join(_corpus(c, ck) for c in corpora) + " | note |",
                  "|---|---|" + "---|" * len(corpora) + "---|"]
            keys = []
            for r in sel:
                if (r["engine"], r["op"]) not in keys:
                    keys.append((r["engine"], r["op"]))
            for e, op in keys:
                cells, notes = [], set()
                for c in corpora:
                    rr = [r for r in sel if r["engine"] == e and r["op"] == op and r.get("corpus") == c]
                    cells.append(fmt(rr[0]) if rr else "")
                    notes |= {r.get("note") for r in rr if r.get("note")}
                if not any(cells):
                    rr = [r for r in sel if r["engine"] == e and r["op"] == op]
                    notes |= {r.get("note") for r in rr if r.get("note")}
                L.append(f"| {ENGINE.get(e, e)} | {op} | "
                         + " | ".join(cells) + f" | {'; '.join(sorted(notes))} |")
        else:
            L += ["| engine | op | corpus | B | ms per call | us per world | note |", "|---|---|---|---|---|---|---|"]
            for r in sel:
                L.append(f"| {ENGINE.get(r['engine'], r['engine'])} | {r['op']} | {r.get('corpus')} | {r.get('nworld')}"
                         f" | {fmt(r)} | {r.get('us_per_world', '')} | {r.get('note', '')} |")
        L.append("")
    sw = [r for r in rows if r["section"] == "sweep"]
    if sw:
        nws = sorted({int(r["nworld"]) for r in sw if r.get("nworld") is not None})
        L += [f"## nworld sweep ({sw[0].get('corpus')})", "",
              "| engine | op | " + " | ".join(map(str, nws)) + " |", "|---|---|" + "---|" * len(nws)]
        keys = []
        for r in sw:
            if (r["engine"], r["op"]) not in keys:
                keys.append((r["engine"], r["op"]))
        for e, op in keys:
            cells = []
            for n in nws:
                rr = [r for r in sw if r["engine"] == e and r["op"] == op and r.get("nworld") == n]
                cells.append(fmt(rr[0]) if rr and rr[0].get("ms_median") is not None
                             else (rr[0].get("note", "") if rr else ""))
            L.append(f"| {ENGINE.get(e, e)} | {op} | " + " | ".join(cells) + " |")
        L.append("")
    rep = repeats(job)
    if rep:
        L += ["## Headline cells over P processes (ms per control step, 4096 worlds)", "",
              "The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; "
              "median over the kept rounds per process.", "",
              "| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {ENGINE.get(r['engine'], r['engine'])} | {r['op']} | {r['corpus']} | {r['processes']} | "
              f"{', '.join(f'{v:.3f}' for v in r['medians'])} | {r['median']:.3f} | {100 * r['spread']:.1f}% |"
              for r in rep]
        L.append("")
    bare = [r for r in rows if r["section"] == "bare"]
    if bare:
        L += ["## Bare kernels (attribution; device time, CUDA events)", "", "| op | ms |", "|---|---|"]
        L += [f"| {r['op']} | {fmt(r)} |" for r in bare]
        L.append("")
    oi = _res(job, "ours").get("init") or {}
    if oi:
        L += [f"jax_ffi.init: {oi.get('seconds')} s (built now: {oi.get('built')}), arch {oi.get('arch')}, "
              f"flags {json.dumps(oi.get('arch_flags'))}, forward defines {oi.get('forward_defines')}", ""]
    L += ["## Parts", "", "| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs |"
          " note |", "|---|---|---|---|---|---|---|---|"]
    for name, p in (job.get("parts") or {}).items():
        g = p.get("gpu_samples") or {}
        o1, o2 = p.get("other_gpu_processes"), p.get("other_gpu_processes_after")
        other = max((x for x in (o1, o2) if isinstance(x, list)), key=len, default=None)
        note = p.get("reason") or ("; ".join(f"{k}: {v}" for k, v in (p.get("error") or {}).items())[:300]
                                   if p.get("error") else "")
        L.append(f"| {name} | {p.get('status')} | {p.get('seconds', '')} | {g.get('sm_mhz_median')} "
                 f"({g.get('sm_mhz_min')}) | {g.get('power_w_max')} | {g.get('hardware_slowdown_flagged')} | "
                 f"{len(other) if isinstance(other, list) else ''} | {note} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default=None, help="default: <json>.md next to it")
    a = ap.parse_args()
    job = json.loads(Path(a.json).read_text())
    job["table"] = table(job)
    md = markdown(job)
    out = Path(a.out) if a.out else Path(a.json).with_suffix(".md")
    out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
