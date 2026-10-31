"""Markdown tables from benchmark result files (dk-bench/1). Standard library only.

  python -m benchmarks.bench report runs/bench/<machine>/                       # every *.json there, to stdout
  python -m benchmarks.bench report runs/bench/a/ runs/bench/b/x.json --out runs/bench/report/   # report.md

Per machine and job:
  timing     one row per (problem, corpus, nworld, iterations, engine [~variant]), one column per op: the median
             milliseconds of one call in the throughput mode (N back-to-back calls, one sync; latency mode with
             --mode latency), or the cell's status where it has no number (n/s = not_supported, oom, time = over the
             unit's time budget, err = error; a hardware throttle flag is appended as "!"), and the control steps per
             second of fwd. Speed-ups are taken against the job's own dk anchor cells (job `anchor`: dk, mjlab, 4096
             worlds, 10 iterations), never across jobs or machines.
  forward    the forward accuracy entries the accuracy scorer wrote (python -m benchmarks.bench.accuracy.job
             <file> --write): per engine, corpus and iterations, qvel relative error and the qpos rotation error
             (rad) against the MuJoCo C float64 truth, median and p99, on all states and on the model-ok states.
  gradient   the gradient entries (group G0, subset all, object vjp): relative VJP error per input block (qpos
             tangent, qvel, ctrl), median and p99, floored at the truth's own uncertainty.
Labelled rows (adapter attribute native = "labelled") are marked with "*". A file that is not a dk-bench/1 document is
skipped with a note. --out never overwrites a file tracked by git.
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.bench import prov, schema

OPS = ("fwd", "fwd_train", "bwd", "vjp", "jac")
STATUS_SHORT = {"not_supported": "n/s", "skipped:oom": "oom", "skipped:time": "time", "error": "err"}


def _docs(paths):
    out, notes = [], []
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        try:
            doc = schema.read(f)
        except Exception as e:  # noqa: BLE001  (not a result file, or an invalid one)
            notes.append(f"skipped {prov.rel(f)}: {type(e).__name__}: {str(e)[:160]}")
            continue
        out.append((f, doc))
    return out, notes


def _status(c):
    st = c["status"]
    kind = st.split(":", 1)[0]
    if kind in ("ok", "flagged"):
        return None
    return STATUS_SHORT.get(st, STATUS_SHORT.get(kind, kind))


def _fmt_ms(v):
    if v is None:
        return "-"
    return f"{v:.3g}" if v < 100 else f"{v:.0f}"


def _engine_name(doc, c):
    e = c["engine"] if c.get("variant", "default") == "default" else f"{c['engine']}~{c['variant']}"
    native = c.get("native") or ((doc.get("engines") or {}).get(e, {}).get("adapter") or {}).get("native")
    return e + (" *" if native == "labelled" else "")


def timing_table(doc, mode="throughput"):
    """Markdown rows of a job's timing cells (module docstring)."""
    rows = {}
    anchor = {}
    for c in doc.get("cells", []):
        if c.get("kind") != "timing" or c.get("mode") != mode:
            continue
        key = (c["model"], c["corpus"] + (f".s{c['subset']}" if c.get("subset") else ""), c["nworld"],
               str(c["iterations"]), _engine_name(doc, c), c.get("process", 0))
        r = rows.setdefault(key, {})
        short = _status(c)
        if short is None:
            flag = "!" if c["status"].startswith("flagged") else ""
            r[c["op"]] = _fmt_ms(c.get("median_ms")) + flag
            if c["op"] == "fwd":
                r["steps"] = c.get("control_steps_per_s")
        else:
            r[c["op"]] = short
    for cid in doc.get("anchor") or []:
        c = next((x for x in doc.get("cells", []) if x["id"] == cid), None)
        if c is not None and c.get("median_ms") is not None:
            anchor[c["op"]] = c["median_ms"]
    head = "| problem | corpus | nworld | it | engine | p | " + " | ".join(f"{o} ms" for o in OPS) + " | fwd steps/s |"
    lines = [head, "|" + "---|" * (7 + len(OPS))]
    for key in sorted(rows, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5])):
        r = rows[key]
        st = r.get("steps")
        lines.append("| " + " | ".join(str(x) for x in key) + " | " + " | ".join(r.get(o, "") for o in OPS)
                     + f" | {'' if st is None else f'{st:.3g}'} |")
    if anchor:
        lines.append("")
        lines.append("dk anchor of this job (mjlab, 4096 worlds, 10 iterations): " + ", ".join(
            f"{k} {v:.3g} ms" for k, v in sorted(anchor.items())))
    return lines


def forward_table(doc):
    ents = [e for e in doc.get("accuracy", []) if e.get("kind") == "forward" and e.get("subset") in ("all", "model_ok")
            and "qvel" in e and not e.get("run_to_run")]
    if not ents:
        return []
    lines = ["| engine | corpus | nworld | it | subset | states | qvel rel. median | qvel rel. p99 | qpos rad median "
             "| qpos rad p99 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(ents, key=lambda e: (e["corpus"], str(e.get("iterations")), str(e.get("engine")),
                                         str(e.get("variant")), e.get("nworld") or 0, e["subset"])):
        eng = e.get("engine", "?") + ("" if e.get("variant") in (None, "default") else f"~{e['variant']}")
        q, r = e["qvel"], e["qpos_rad"]
        lines.append(f"| {eng} | {e['corpus']} | {e.get('nworld') or '-'} | {e.get('iterations')} | {e['subset']} | "
                     f"{e.get('states')} | "
                     f"{q.get('median', float('nan')):.2e} | {q.get('p99', float('nan')):.2e} | "
                     f"{r.get('median', float('nan')):.2e} | {r.get('p99', float('nan')):.2e} |")
    return lines


def gradient_table(doc, group="G0", subset="all"):
    ents = [e for e in doc.get("gradient", []) if e.get("kind") == "gradient" and e.get("object") == "vjp"
            and e.get("group") == group and e.get("subset") == subset and "median" in e]
    if not ents:
        return []
    rows = {}
    for e in ents:
        eng = e.get("engine", "?") + ("" if e.get("variant") in (None, "default") else f"~{e['variant']}")
        rows.setdefault((e["corpus"], str(e.get("iterations")), eng, e.get("nworld") or 0), {})[e["block"]] = e
    lines = ["| engine | corpus | nworld | it | states | qpos median / p99 | qvel median / p99 | ctrl median / p99 |",
             "|---|---|---|---|---|---|---|---|"]
    for (corpus, it, eng, nw), b in sorted(rows.items()):
        cells = []
        for blk in ("qpos", "qvel", "ctrl"):
            e = b.get(blk)
            cells.append("-" if e is None else f"{e['median']:.2e} / {e['p99']:.2e}")
        n = next(iter(b.values())).get("usable_states")
        lines.append(f"| {eng} | {corpus} | {nw or '-'} | {it} | {n} | " + " | ".join(cells) + " |")
    return lines


def render(docs, notes=(), mode="throughput"):
    out = ["# Benchmark report", ""]
    for f, doc in docs:
        job, m = doc["job"], doc["machine"]
        gpu = (m.get("gpu") or {}).get("name")
        out += [f"## {m.get('tag')}: job {job['name']}" + (" (quick)" if job.get("quick") else ""), "",
                f"File `{prov.rel(f)}`; GPU {gpu}; {len(doc['cells'])} cells; command `{job.get('cmd')}`.", ""]
        warn = job.get("warnings") or []
        if warn:
            out += [f"Warnings: {'; '.join(warn)}", ""]
        out += [f"### Timing ({mode} mode)", ""] + timing_table(doc, mode) + [""]
        ft = forward_table(doc)
        if ft:
            out += ["### Forward accuracy against the MuJoCo C float64 truth", ""] + ft + [""]
        gt = gradient_table(doc)
        if gt:
            out += ["### Gradient accuracy (VJP, group G0, all sampled states)", ""] + gt + [""]
    if notes:
        out += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    out.append("`*`: a labelled row (not a native configuration of that engine). Timing cells: median ms per call; "
               "n/s not supported natively (the result file quotes the reason), oom, time, err; `!` hardware "
               "throttling during the timed rounds.")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m benchmarks.bench report", description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="result files or directories of them")
    ap.add_argument("--out", help="directory for report.md (default: print)")
    ap.add_argument("--mode", default="throughput", choices=schema.MODES, help="timing mode of the timing table")
    a = ap.parse_args(argv)
    docs, notes = _docs(a.paths)
    if not docs:
        print("report: no result files found", file=sys.stderr)
        return 1
    text = render(docs, notes, a.mode)
    if a.out:
        out = Path(a.out) / "report.md"
        if prov.is_committed(out):
            print(f"report: {prov.rel(out)} is committed: choose another --out", file=sys.stderr)
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(json.dumps(dict(report=prov.rel(out), files=len(docs), skipped=len(notes))))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
