"""A diagnostic of one engine's variants on the same states (e.g. Genesis float32 against float64, or against another
integrator): how far each variant's forward outputs and gradients are from the MuJoCo C float64 truth, and how far
the variants are from each other. The difference between a float32 and a float64 run of the same engine is that
engine's own numerical error; its distance to the truth that remains in float64 is the model difference. Reads
finished harness jobs (dump cells of `python -m benchmarks.bench run ... --variant <engine>=<v1>,<v2>`); numpy and
mujoco, no GPU.

Forward (every state of the corpus): per variant against the truth (accuracy/forward.py: qvel relative, qpos tangent
rad, all and model-ok states), and per variant against the reference variant (the same metrics with the reference's
outputs in place of the truth). Gradient (the blocks the variants have): per variant against the truth on its sampled
states (accuracy/gradient.py, floored relative error per group), and per variant against the reference on every
state and cotangent: |g - g_ref| / max(|g_ref|, 1e-9) and the cosine, per block.

  python -m benchmarks.bench.report.variant_diag --jobs runs/bench/<dir>/ --engine genesis \
      --variants default,fp64,approx --reference fp64 --corpus mf_s0 --iterations 10 \
      --out-json docs/results/<name>.json --out-md docs/results/<name>.md
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench import prov, schema
from benchmarks.bench.accuracy import forward as AF
from benchmarks.bench.accuracy import gradient as AG
from benchmarks.bench.accuracy import stats
from benchmarks.bench.report import paper_accuracy as PA


def _summ(x, cos=None):
    s = stats.summary(x, cos)
    return PA._round(s)


def find_cells(docs, engine, variants, corpus, iterations, process=0):
    out = {}
    for path, doc in docs:
        for c in doc["cells"]:
            if c.get("kind") != "dump" or schema.parse_status(c.get("status", "error"))[0] != "ok":
                continue
            if (c["engine"], c["corpus"], c["iterations"], c["process"]) != (engine, corpus, iterations, process):
                continue
            if c["variant"] in variants:
                if c["variant"] in out:
                    raise ValueError(f"two dumps of variant {c['variant']}")
                out[c["variant"]] = (path, doc, c)
    return out


def grad_diff(d, ref, sampled):
    """{block: summaries of |g - g_ref| / |g_ref| and cosines over every state and over the sampled ones} of two dumps
    on the same states (the blocks both have)."""
    blocks = [b for b in AG.BLOCKS if f"grad0_{b}" in d and f"grad0_{b}" in ref]
    C = 0
    while blocks and f"grad{C}_{blocks[0]}" in d:
        C += 1
    if not np.array_equal(d["index"], ref["index"]):
        raise ValueError("the dumps cover different states")
    per = {}
    for b in blocks:
        g = np.stack([d[f"grad{c}_{b}"] for c in range(C)], 1)
        r = np.stack([ref[f"grad{c}_{b}"] for c in range(C)], 1)
        nr = np.linalg.norm(r, axis=2)
        e = np.linalg.norm(g - r, axis=2) / np.maximum(nr, 1e-9)
        cos = np.sum(g * r, 2) / np.maximum(np.linalg.norm(g, axis=2) * nr, 1e-30)
        m = np.isin(d["index"], sampled)
        per[b] = dict(all_states=_summ(e, cos), sampled_states=_summ(e[m], cos[m]),
                      bitwise_equal=bool(np.array_equal(g, r)))
    return dict(states=int(len(d["index"])), cotangents=C, blocks=per)


def build(docs, engine, variants, reference, corpus, iterations):
    from benchmarks.bench.truth import store
    cells = find_cells(docs, engine, variants, corpus, iterations)
    missing = [v for v in variants if v not in cells]
    if missing:
        raise ValueError(f"no dump for variants {missing} ({engine}, {corpus}, {iterations} iterations)")
    tr = store.load(corpus)
    model = AF._model(tr)
    masks = AF.subset_masks(tr, ("all", "model_ok"))
    GT = AG.truth_from_store(tr) if tr.has_gradient else None
    dumps = {v: PA.read_dump(cells[v][2]) for v in variants}
    res = dict(forward=dict(vs_truth={}, vs_reference={}), gradient=dict(vs_truth={}, vs_reference={}))
    ref = dumps[reference]
    ref_src = AF.source_from_dump(ref)
    ref_src["name"] = f"{engine}~{reference}"
    for v in variants:
        d = dumps[v]
        src = AF.source_from_dump(d)
        res["forward"]["vs_truth"][v], _ = PA.score_forward(src, tr, model, masks)
        if v != reference:
            res["forward"]["vs_reference"][v], _ = PA.score_forward(src, tr, model, masks, reference=ref_src)
        if GT is not None and any(k.startswith("grad0_") for k in d):
            cell = cells[v][2]
            gs = AG.source_from_dump(d, engine=engine, variant=v, not_supported=cell.get("gradient_not_supported"))
            ent, _ = AG.score(gs, GT)
            res["gradient"]["vs_truth"][v] = PA.compact_grad(ent, brief=True)
            if v != reference:
                res["gradient"]["vs_reference"][v] = grad_diff(d, ref, GT["index"])
    # run to run of the first variant (process 1 against process 0), when the job ran it in several processes
    later = find_cells(docs, engine, variants[:1], corpus, iterations, process=1)
    if variants[0] in later:
        v = variants[0]
        d1 = PA.read_dump(later[v][2])
        r0 = AF.source_from_dump(dumps[v])
        r0["name"] = f"{engine}~{v} process 0"
        res["run_to_run"] = dict(variant=v, cell=later[v][2]["id"],
                                 forward=PA.score_forward(AF.source_from_dump(d1), tr, model, masks, reference=r0)[0])
        if GT is not None and any(k.startswith("grad0_") for k in d1):
            res["run_to_run"]["gradient"] = grad_diff(d1, dumps[v], GT["index"])
    prov_ = dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 jobs=sorted({prov.rel(p) for p, _, _ in cells.values()}),
                 dumps={v: cells[v][2]["raw"] for v in variants},
                 cells={v: cells[v][2]["id"] for v in variants},
                 truth=dict(file=f"benchmarks/data/truth/{corpus}.npz",
                            sha256=prov.sha256(P.REPO / f"benchmarks/data/truth/{corpus}.npz")),
                 gpu=(cells[reference][1].get("machine") or {}).get("gpu"),
                 engine_provenance={v: {k: ((cells[v][1].get("engines") or {}).get(engine) or {}).get(
                     "provenance", {}).get(k) for k in ("precision", "rigid_options", "rigid_options_grad")}
                     for v in variants})
    return dict(schema="dk-variant-diag/1", engine=engine, variants=list(variants), reference=reference,
                corpus=corpus, iterations=iterations, provenance=prov_, **PA._round(res))


def render(res, path):
    f = PA.f
    e, ref = res["engine"], res["reference"]
    L = [f"# {e}: variants against the truth and against `{ref}` ({res['corpus']}, {res['iterations']} iterations)",
         "", f"Every number is in `{path}` (generated {res['provenance']['date']} by "
         "`python -m benchmarks.bench.report.variant_diag`). Forward: all 4096 states; qvel relative error, qpos "
         "tangent (rad). Gradient: VJPs of the 3 stored cotangents; against the truth on its sampled states "
         "(floored relative error, median), against the reference variant on every state.", "",
         "## Forward", "",
         "| variant | vs | qvel median | p90 | p99 | max | qpos (rad) median | p99 | model-ok qvel median | p99 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(v, "truth", res["forward"]["vs_truth"][v]) for v in res["variants"]]
    rows += [(v, ref, res["forward"]["vs_reference"][v]) for v in res["variants"] if v != ref]
    for v, vs, x in rows:
        a, m = x["all"], x["model_ok"]
        L.append(f"| {v} | {vs} | {f(a['qvel'].get('median'))} | {f(a['qvel'].get('p90'))} | "
                 f"{f(a['qvel'].get('p99'))} | {f(a['qvel'].get('max'))} | {f(a['qpos_rad'].get('median'))} | "
                 f"{f(a['qpos_rad'].get('p99'))} | {f(m['qvel'].get('median'))} | {f(m['qvel'].get('p99'))} |")
    rr = res.get("run_to_run")
    if rr:
        a, m = rr["forward"]["all"], rr["forward"]["model_ok"]
        L.append(f"| {rr['variant']} process 1 | process 0 | {f(a['qvel'].get('median'))} | "
                 f"{f(a['qvel'].get('p90'))} | "
                 f"{f(a['qvel'].get('p99'))} | {f(a['qvel'].get('max'))} | {f(a['qpos_rad'].get('median'))} | "
                 f"{f(a['qpos_rad'].get('p99'))} | {f(m['qvel'].get('median'))} | {f(m['qvel'].get('p99'))} |")
    L += ["", "## Gradient", "", "Against the truth (floored relative error, median / p99 of the supported blocks, all "
          "sampled states of each group):", "", "| variant | block | G0 | G4 | G5 | G6 |", "|---|---|---|---|---|---|"]
    for v, g in res["gradient"]["vs_truth"].items():
        for b in AG.BLOCKS:
            cs = []
            for grp in PA.TABLE_GROUPS:
                x = (g.get(grp) or {}).get("all", {}).get(b, {})
                cs.append("n/s" if x.get("status") else f"{f(x.get('median'))} / {f(x.get('p99'))}")
            if all(c == "n/s" for c in cs):
                continue
            L.append(f"| {v} | {b} | " + " | ".join(cs) + " |")
    L += ["", f"Against `{ref}` (every state and cotangent): |g - g_ref| / |g_ref|", "",
          "| variant | block | median | p90 | p99 | max | min cos | sampled states: median | p99 |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = list(res["gradient"]["vs_reference"].items())
    if rr and rr.get("gradient"):
        rows.append((f"{rr['variant']} process 1 vs process 0", rr["gradient"]))
    for v, x in rows:
        for b, y in x["blocks"].items():
            a, s = y["all_states"], y["sampled_states"]
            L.append(f"| {v} | {b} | {f(a.get('median'))} | {f(a.get('p90'))} | {f(a.get('p99'))} | "
                     f"{f(a.get('max'))} | {a.get('cos_min')} | {f(s.get('median'))} | {f(s.get('p99'))} |")
    if res.get("notes"):
        L += ["", "## Notes", ""] + [f"- {n}" for n in res["notes"]]
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--variants", required=True, help="comma-separated, e.g. default,fp64")
    ap.add_argument("--reference", required=True, help="the variant the others are compared with")
    ap.add_argument("--corpus", default="mf_s0")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--note", action="append", default=[], help="a note for the JSON and the markdown (repeat)")
    ap.add_argument("--attach", action="append", default=[], metavar="NAME=JSON",
                    help="embed another result file (another diagnostic's JSON) under attached.NAME (repeat)")
    a = ap.parse_args(argv)
    for o in (a.out_json, a.out_md):
        if prov.is_committed(Path(o)):
            print(f"{o} is committed: choose another path", file=sys.stderr)
            return 1
    variants = a.variants.split(",")
    if a.reference not in variants:
        variants.append(a.reference)
    res = build(PA.load_docs(a.jobs), a.engine, variants, a.reference, a.corpus, a.iterations)
    res["file"] = prov.rel(a.out_json)
    res["notes"] = list(a.note)
    for item in a.attach:
        name, _, path = item.partition("=")
        pth = Path(path)
        res.setdefault("attached", {})[name] = dict(file=prov.rel(pth), sha256=prov.sha256(pth),
                                                    content=json.loads(pth.read_text()))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(schema.clean(res), indent=1) + "\n")
    Path(a.out_md).write_text(render(res, res["file"]) + "\n")
    print(json.dumps(dict(json=res["file"], md=prov.rel(a.out_md))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
