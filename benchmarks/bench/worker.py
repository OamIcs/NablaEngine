"""Runs ONE benchmark unit inside the interpreter its engine needs and writes a JSON record: the unit is (engine,
variant, problem, corpus, nworld, iterations, process) with lists of state subsets, ops and timing modes. It imports
only that engine's adapter module (adapters.load), so it runs in every engine environment; the driver (protocol.py)
starts a fresh worker process per unit. Standard library + numpy (+ the adapter's own dependencies).

  python -m benchmarks.bench.worker --spec unit.json --out result.json   # what the driver runs
  python -m benchmarks.bench.worker --probe ENGINE                        # env-check: import + check_env, JSON

Per state subset of the unit (one after the other, a fresh adapter each): select the states (problem.select: a fixed
seeded permutation), adapter.setup (timed as a whole and by the parts the adapter reports), then for each op:
prepare(op) (timed), one call + sync (first_call_s), a warm-up of at least warmup_calls calls and warmup_s seconds
(back-to-back groups of doubling size, one sync each), then per mode
  throughput  `rounds` rounds of N back-to-back calls and one sync each (time / N per round)
  latency     `rounds` rounds of N individually synchronised calls (the median call per round; calls_ms keeps all)
The first drop_rounds rounds are dropped from the statistics (kept in rounds_ms). N (reps_per_round) comes from the
warm-up's per-call time: at least round_min_s per round, at least min_reps_gpu / min_reps_cpu calls, at most
round_max_s per round; the adapter's timing_hints(op) override reps / rounds / warm-up. When the unit's time budget
(deadline_unix) cannot hold the rounds, N shrinks (recorded in `budget`), down to 1, else the op is skipped:time.
Wall time of each mode's kept rounds goes into window_unix, so the driver can attach the GPU clock samples.
After the ops: outputs() (non-finite worlds), diagnostics(), memory(), and the requested raw dumps (outputs,
gradient of the 3 stored cotangents, jacobian) into <raw_dir>/<unit key>.npz.
The result file is rewritten after every op, so a unit killed at its timeout keeps the ops it finished.
"""

import argparse
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from benchmarks.bench import adapters, prov, schema
from benchmarks.bench import problem as P
from benchmarks.bench.adapters import base

DUMPS = ("outputs", "gradient", "jacobian")
MSG_LIMIT = 4000


def _msg(exc):
    """Exception text for a cell message: type, message and the innermost frames (paths scrubbed)."""
    tb = traceback.extract_tb(exc.__traceback__)[-3:]
    where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(tb))
    return prov.scrub(f"{type(exc).__name__}: {exc}" + (f" [at {where}]" if where else ""))[:MSG_LIMIT]


def _status_of(exc):
    if isinstance(exc, base.NotSupported):
        return schema.status("not_supported"), prov.scrub(str(exc))[:MSG_LIMIT]
    if base.is_oom(exc):
        return schema.status("skipped", "oom"), _msg(exc)
    return schema.status("error", schema.short(type(exc).__name__)), _msg(exc)


def _iters(it):
    return "native" if it is None else int(it)


class Worker:
    def __init__(self, spec, out):
        self.spec, self.out = spec, Path(out)
        self.unit = spec["unit"]
        self.prot = spec["protocol"]
        self.deadline = float(spec.get("deadline_unix") or (time.time() + 1e9))
        self.t_start = time.time()
        self.rec = dict(status="running", unit=self.unit["key"], pid=os.getpid(), started=schema.now(),
                        python=platform.python_version(), cells=[], message=None)

    # ------------------------------------------------------------------ records
    def save(self):
        self.rec["elapsed_s"] = round(time.time() - self.t_start, 3)
        tmp = self.out.with_name(self.out.name + ".tmp")
        tmp.write_text(json.dumps(schema.clean(self.rec), indent=1, allow_nan=False))
        os.replace(tmp, self.out)

    def cid(self, op, mode, subset=None):
        u = self.unit
        return schema.cell_id(u["engine"], u["variant"], u["model"]["id"], u["corpus"],
                              self.subset if subset is None else subset, u["nworld"], _iters(u["iterations"]), op,
                              mode, u["process"])

    def cell(self, op, mode, status, message=None, subset=None, **fields):
        u = self.unit
        subset = self.subset if subset is None else subset
        c = dict(id=self.cid(op, mode, subset), kind="timing" if op is not None else "dump", engine=u["engine"],
                 variant=u["variant"], label=self.cls_info.get("label"), native=self.cls_info.get("native"),
                 device=self.cls_info.get("device"), env=self.spec.get("env"), model=u["model"]["id"],
                 track=u["model"]["track"], corpus=u["corpus"], subset=subset, nworld=u["nworld"],
                 iterations=_iters(u["iterations"]), substeps=u["model"]["substeps"], op=op, mode=mode,
                 process=u["process"], status=status, message=message, headline=bool(u.get("headline")),
                 started=schema.now(), cmd=self.spec.get("cmd", ""), unit=u["key"])
        if u.get("opts"):
            c["opts"] = u["opts"]
        c.update(fields)
        self.rec["cells"].append(c)
        self.save()
        return c

    def fail_all(self, status, message, subsets=None):
        """Write `status` for every cell of the given subsets (default: all of the unit) not written yet."""
        have = {c["id"] for c in self.rec["cells"]}
        for s in self.unit["subsets"] if subsets is None else subsets:
            for op in list(self.unit["ops"]) + ([None] if self.unit.get("dump") else []):
                for mode in (self.unit["modes"] if op is not None else [None]):
                    if self.cid(op, mode, s) not in have:
                        self.cell(op, mode, status, message, subset=s)

    # ------------------------------------------------------------------ main
    def run(self):
        u = self.unit
        self.cls_info, self.subset = {}, u["subsets"][0]
        try:
            cls = adapters.load(u["engine"])
        except Exception as e:  # noqa: BLE001
            self.rec.update(status="error", message=_msg(e))
            self.fail_all(schema.status("error", "adapter"), _msg(e))
            return 0
        self.cls_info = cls.describe()
        self.rec["adapter"] = self.cls_info
        self.rec["software"] = prov.software()
        prob = P.Problem.from_dict(u["model"])
        if prob.track not in cls.tracks:
            msg = f"{cls.id} does not run track {prob.track} problems ({prob.id})"
            self.rec.update(status="done", message=msg)
            self.fail_all(schema.status("not_supported", "track"), msg)
            return 0
        if u["variant"] not in cls.variants:
            msg = f"{cls.id}: unknown variant {u['variant']!r} (one of {sorted(cls.variants)})"
            self.rec.update(status="error", message=msg)
            self.fail_all(schema.status("error", "variant"), msg)
            return 0
        opts = {**cls.variants[u["variant"]], **(u.get("opts") or {})}
        try:
            corpus = P.load_corpus(u["corpus"])
            self.rec["corpus"] = corpus.describe()
            self.rec["problem"] = prob.describe()
        except Exception as e:  # noqa: BLE001
            self.rec.update(status="error", message=_msg(e))
            self.fail_all(schema.status("error", "corpus"), _msg(e))
            return 0
        self.rec.update(setup={}, states={}, provenance=None, semantics=None)
        for k, s in enumerate(u["subsets"]):
            self.subset = s
            self.run_subset(cls, prob, corpus, opts, s, len(u["subsets"]) - k - 1)
        self.rec["status"] = "done"
        self.save()
        return 0

    def run_subset(self, cls, prob, corpus, opts, s, subsets_after):
        """setup, the ops, finish for one state subset with a fresh adapter."""
        u = self.unit
        ops = list(u["ops"])
        try:
            states = P.select(corpus, u["nworld"], s, self.prot.get("seed", P.SELECT_SEED))
        except Exception as e:  # noqa: BLE001
            self.fail_all(schema.status("error", "states"), _msg(e), [s])
            return
        self.rec["states"][str(s)] = states.describe()
        self.save()
        a = cls()
        t0 = time.perf_counter()
        try:
            setup = a.setup(prob, states, u["nworld"], u["iterations"], **opts) or {}
        except Exception as e:  # noqa: BLE001
            st, msg = _status_of(e)
            self.rec["setup"][str(s)] = dict(error=msg, total_s=round(time.perf_counter() - t0, 4))
            self.rec["message"] = msg
            self.fail_all(st, msg, [s])
            return
        self.setup = dict(setup, total_s=round(time.perf_counter() - t0, 4))
        self.rec["setup"][str(s)] = self.setup
        self._engine_info(a)
        self.save()
        first = len(self.rec["cells"])
        try:
            fns = a.ops()
            missing = a.not_supported()
        except Exception as e:  # noqa: BLE001
            st, msg = _status_of(e)
            self.fail_all(st, msg, [s])
            self._close(a)
            return
        timed = {}  # op -> (callable, {mode: cell})
        n_timed = sum(1 for o in ops if o in fns)
        for i, op in enumerate(ops):
            if op not in fns:
                msg = missing.get(op) or f"{cls.id}: no native {op}"
                for mode in u["modes"]:
                    self.cell(op, mode, schema.status("not_supported"), prov.scrub(msg))
                continue
            alias = next((o for o, (f, _) in timed.items() if f == fns[op]), None)
            if alias is not None:
                for mode, src in timed[alias][1].items():
                    c = {k: v for k, v in src.items() if k not in ("id", "op", "started", "alias_of")}
                    self.rec["cells"].append(dict(c, id=self.cid(op, mode), op=op, alias_of=alias,
                                                  started=schema.now()))
                self.save()
                continue
            n_left = sum(1 for o in ops[i:] if o in fns) + n_timed * subsets_after
            timed[op] = (fns[op], self.time_op(a, op, fns[op], n_left))
        self.finish(a, states, self.rec["cells"][first:])
        self._close(a)

    @staticmethod
    def _close(a):
        try:
            a.close()
        except Exception:  # noqa: BLE001
            pass

    def _engine_info(self, a):
        for k in ("provenance", "semantics"):
            try:
                self.rec[k] = getattr(a, k)()
            except Exception as e:  # noqa: BLE001
                self.rec[k] = dict(error=_msg(e))

    # ------------------------------------------------------------------ timing
    def remaining(self):
        return self.deadline - time.time()

    def time_op(self, a, op, fn, n_left):
        """Time one op in every mode; returns {mode: cell} of the cells written."""
        u, p = self.unit, self.prot
        modes = list(u["modes"])
        try:
            hints = dict(a.timing_hints(op) or {})
        except Exception as e:  # noqa: BLE001
            hints = dict(error=_msg(e))
        unknown = set(hints) - set(base.HINT_KEYS) - {"error"}
        if unknown:
            msg = f"timing_hints({op}) has unknown keys {sorted(unknown)} (allowed: {base.HINT_KEYS})"
            return {m: self.cell(op, m, schema.status("error", "hints"), msg) for m in modes}
        rounds = int(hints.get("rounds", p["rounds"]))
        drop = min(int(p["drop_rounds"]), rounds - 1)
        warm_calls = int(hints.get("warmup_calls", p["warmup_calls"]))
        warm_s = float(hints.get("warmup_s", p["warmup_s"]))
        try:
            t0 = time.perf_counter()
            prep = a.prepare(op) or {}
            prepare_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn()
            a.sync()
            first_call_s = time.perf_counter() - t0
            # warm-up: groups of k back-to-back calls + one sync, k doubling while a group is short
            calls, k, t_w0, per = 0, 1, time.perf_counter(), first_call_s
            while True:
                if self.remaining() < 0:
                    raise TimeoutError(f"warm-up of {op} did not finish within the unit's budget")
                tg = time.perf_counter()
                for _ in range(k):
                    fn()
                a.sync()
                dt = time.perf_counter() - tg
                calls += k
                per = dt / k
                if calls >= warm_calls and time.perf_counter() - t_w0 >= warm_s:
                    break
                if dt < 0.05:
                    k = min(2 * k, 4096)
            warm = dict(calls=calls, s=round(time.perf_counter() - t_w0, 4), per_call_ms=per * 1e3)
            # repetitions per round
            if "reps" in hints:
                n_want = int(hints["reps"])
            else:
                min_reps = p["min_reps_gpu"] if self.cls_info.get("device") == "gpu" else p["min_reps_cpu"]
                n_want = max(math.ceil(p["round_min_s"] / per), int(min_reps))
                n_want = min(n_want, max(1, math.floor(p["round_max_s"] / per)), int(p.get("max_reps", 100000)))
            share = max(0.0, self.remaining() - 5.0) / max(1, n_left)
            n_fit = math.floor(share / (per * rounds * len(modes) * 1.2)) if per > 0 else n_want
            budget = dict(reps_wanted=n_want, share_s=round(share, 2), reduced=n_fit < n_want)
            if n_fit < 1:
                msg = (f"{op}: {rounds} rounds x {len(modes)} modes at {per * 1e3:.3g} ms per call do not fit the "
                       f"{share:.1f} s left for it (per-unit timeout)")
                return {m: self.cell(op, m, schema.status("skipped", "time"), msg, budget=budget) for m in modes}
            N = min(n_want, n_fit)
        except Exception as e:  # noqa: BLE001
            st, msg = _status_of(e)
            if isinstance(e, TimeoutError):
                st = schema.status("skipped", "time")
            return {m: self.cell(op, m, st, msg) for m in modes}
        out = {}
        for mode in modes:
            try:
                rounds_ms, calls_ms, window = self._rounds(a, fn, mode, rounds, N, drop)
            except Exception as e:  # noqa: BLE001
                st, msg = _status_of(e)
                out[mode] = self.cell(op, mode, st, msg, budget=budget)
                continue
            kept = rounds_ms[drop:] if mode == "throughput" else calls_ms
            med = statistics.median(kept)
            q = np.percentile(kept, [10, 90])
            sps = u["nworld"] / (med * 1e-3)
            out[mode] = self.cell(
                op, mode, schema.status("ok"), None, reps_per_round=N, rounds=rounds, drop_rounds=drop,
                rounds_ms=rounds_ms, median_ms=med, p10_ms=float(q[0]), p90_ms=float(q[1]), min_ms=min(kept),
                max_ms=max(kept), control_steps_per_s=sps, world_substeps_per_s=sps * u["model"]["substeps"],
                warmup=warm, window_unix=window, first_call_s=first_call_s, prepare_s=prepare_s, clocks=None,
                setup=dict(self.setup, **{f"prepare_{k}": v for k, v in prep.items()}), hints=hints or None,
                budget=budget, **({"calls_ms": calls_ms} if mode == "latency" else {}))
        return out

    def _rounds(self, a, fn, mode, rounds, N, drop):
        """(per-call ms of every round, kept individual calls (latency), [unix start, end] of the kept rounds)."""
        rounds_ms, calls, window = [], [], [None, None]
        pc = time.perf_counter
        for r in range(rounds):
            if r == drop:
                window[0] = time.time()
            if mode == "throughput":
                t = pc()
                for _ in range(N):
                    fn()
                a.sync()
                rounds_ms.append((pc() - t) / N * 1e3)
            else:
                cs = []
                for _ in range(N):
                    t = pc()
                    fn()
                    a.sync()
                    cs.append((pc() - t) * 1e3)
                rounds_ms.append(statistics.median(cs))
                if r >= drop:
                    calls += cs
        window[1] = time.time()
        return rounds_ms, calls, window

    # ------------------------------------------------------------------ after the ops
    def finish(self, a, states, cells):
        u = self.unit
        diag, outs = {}, None
        try:
            outs = a.outputs()
            bad = ~np.all(np.isfinite(np.concatenate([np.asarray(outs["qpos"], np.float64),
                                                      np.asarray(outs["qvel"], np.float64)], 1)), 1)
            nonfinite = int(bad.sum())
        except Exception as e:  # noqa: BLE001
            nonfinite = None
            diag["outputs_error"] = _msg(e)
        try:
            d = dict(a.diagnostics() or {})
            diag.update({k: d.pop(k) for k in list(d) if k in base.DIAG_KEYS})
            if d:
                diag.setdefault("extra", {}).update(d)
        except Exception as e:  # noqa: BLE001
            diag["diagnostics_error"] = _msg(e)
        if diag.get("nonfinite_worlds") is None:
            diag["nonfinite_worlds"] = nonfinite
        try:
            mem = dict(a.memory() or {})
        except Exception as e:  # noqa: BLE001
            mem = dict(error=_msg(e))
        mem["rss_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        mib = prov.process_gpu_mib() if self.cls_info.get("device") == "gpu" else None
        mem["nvml_process_bytes"] = None if mib is None else int(mib * 2 ** 20)
        if mem.get("alloc_peak_bytes"):
            mem["bytes_per_world"] = mem["alloc_peak_bytes"] / u["nworld"]
        for c in cells:
            c["diag"], c["mem"] = diag, mem
        self.rec.setdefault("diag", {})[str(states.subset)] = diag
        self.rec.setdefault("mem", {})[str(states.subset)] = mem
        self._engine_info(a)  # after the run: adapters may record what setup only learned later
        if u.get("dump"):
            self.dump(a, states, outs, diag, mem)
        self.save()

    def dump(self, a, states, outs, diag, mem):
        u = self.unit
        want = [d for d in u["dump"] if d in DUMPS]
        raw_dir = self.spec.get("raw_dir")
        arrays, contents, notes, extra = dict(index=states.index), [], [], {}
        if "outputs" in want and outs is not None:
            arrays.update({f"out_{k}": np.asarray(v, np.float64) for k, v in outs.items()})
            contents.append("outputs")
        if "gradient" in want:
            if states.cot_qpos is None:
                notes.append("gradient: the corpus has no stored cotangents")
            else:
                try:
                    blocks = None
                    for c in range(states.cot_qpos.shape[1]):
                        g = a.gradient(*states.cotangent(c))
                        have = tuple(k for k in base.GRAD_BLOCKS if k in g)
                        if not have or (blocks is not None and have != blocks):
                            raise ValueError(f"gradient() returned the blocks {have} (cotangent {c})")
                        blocks = have
                        for k in have:
                            arrays[f"grad{c}_{k}"] = np.asarray(g[k], np.float64)
                    if blocks != base.GRAD_BLOCKS:  # a partial gradient: which blocks, and the engine's reasons
                        arrays["grad_blocks"] = np.array(blocks)
                        missing = [k for k in base.GRAD_BLOCKS if k not in blocks]
                        why = dict(a.gradient_not_supported() or {})
                        extra["gradient_not_supported"] = {k: prov.scrub(str(why.get(k, "not supported")))[:MSG_LIMIT]
                                                           for k in missing}
                    report = a.gradient_report()
                    if report:
                        extra["gradient_report"] = report
                    contents.append("gradient")
                except Exception as e:  # noqa: BLE001
                    notes.append(f"gradient: {_status_of(e)[1]}")
        if "jacobian" in want:
            try:
                arrays["jacobian"] = np.asarray(a.jacobian(), np.float64)
                contents.append("jacobian")
            except Exception as e:  # noqa: BLE001
                notes.append(f"jacobian: {_status_of(e)[1]}")
        if raw_dir is None:
            self.cell(None, None, schema.status("error", "no_raw_dir"), "dump requested without a raw directory",
                      diag=diag, mem=mem)
            return
        path = Path(raw_dir) / f"{u['key'].replace('/', '__')}{f'__s{states.subset}' if states.subset else ''}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        st = schema.status("ok") if contents else schema.status("not_supported")
        self.cell(None, None, st, "; ".join(notes) or None if contents else ("; ".join(notes) or "nothing to dump"),
                  raw=prov.rel(path), contents=contents, states=states.describe(), diag=diag, mem=mem, **extra)


def probe(engine):
    """env-check inside an environment: import the adapter, run its check_env; JSON on stdout (last line)."""
    out = dict(engine=engine, python=platform.python_version(), executable=prov.rel(sys.executable))
    try:
        cls = adapters.load(engine)
        out["adapter"] = cls.describe()
        out["check"] = cls.check_env() or {}
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out.update(ok=False, error=_msg(e))
    out["packages"] = prov.packages()
    print("\n" + json.dumps(schema.clean(out)))
    return 0 if out["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", help="the unit's JSON spec (written by protocol.py)")
    ap.add_argument("--out", help="where to write the result JSON")
    ap.add_argument("--probe", metavar="ENGINE", help="import ENGINE's adapter and run its check_env (env-check)")
    a = ap.parse_args(argv)
    if a.probe:
        return probe(a.probe)
    if not (a.spec and a.out):
        ap.error("--spec and --out are required (or --probe)")
    spec = json.loads(Path(a.spec).read_text())
    w = Worker(spec, a.out)
    try:
        return w.run()
    except BaseException as e:
        w.rec.update(status="error", message=_msg(e))
        w.save()
        raise


if __name__ == "__main__":
    sys.exit(main())
