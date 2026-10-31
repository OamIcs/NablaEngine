"""Command line of the benchmark harness (run from the repository root, in the dk environment):

  python -m benchmarks.bench env-check [--engines E,...] [--strict]
  python -m benchmarks.bench run --out DIR [--machine TAG] [--part P,...] [--engines E,...] [--quick] ...
  python -m benchmarks.bench report ...        (benchmarks/bench/report)
  python -m benchmarks.bench fetch-models ...  (benchmarks/bench/models)
  python -m benchmarks.bench gen-corpus ...    (benchmarks/bench/corpus)
  python -m benchmarks.bench truth ...         (benchmarks/bench/truth)
  python -m benchmarks.bench validate FILE...  (check result files against dk-bench/1)
  python -m benchmarks.bench list              (engines with adapters / environments, parts)

report, fetch-models, gen-corpus and truth hand their arguments to the main(argv) -> int of the first importable
module of DELEGATES[<subcommand>]; until that module exists the subcommand says so and exits with status 2.
`run` is protocol.run (its docstring has the protocol, the result files and the environment variables).
"""

import argparse
import importlib
import importlib.util
import json
import sys

DELEGATES = {
    "report": ("benchmarks.bench.report.cli", "benchmarks.bench.report.build", "benchmarks.bench.report"),
    "fetch-models": ("benchmarks.bench.models.fetch", "benchmarks.bench.models"),
    "gen-corpus": ("benchmarks.bench.corpus.gen_mf", "benchmarks.bench.corpus.cli", "benchmarks.bench.corpus"),
    "truth": ("benchmarks.bench.truth.cli", "benchmarks.bench.truth.make", "benchmarks.bench.truth"),
}


def _delegate(name, argv):
    for mod in DELEGATES[name]:
        try:
            found = importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError:  # a parent package does not exist
            found = False
        if not found:
            continue
        m = importlib.import_module(mod)
        if callable(getattr(m, "main", None)):
            return int(m.main(argv) or 0)
    print(f"`{name}` is not implemented yet: it is provided by main(argv) of one of {', '.join(DELEGATES[name])}",
          file=sys.stderr)
    return 2


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _kv(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v


def _kv_ref(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected ENGINE=REF, got {s!r}")
    e, ref = (x.strip() for x in s.split("=", 1))
    if not e or not ref:
        raise argparse.ArgumentTypeError(f"expected ENGINE=REF, got {s!r}")
    return e, ref


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def run_parser(sub):
    ap = sub.add_parser("run", help="run benchmark units (protocol.py)",
                        description="Run a job: plan its units, run each in a fresh worker process, write "
                                    "<out>/<job>.json after every unit (resumable).")
    ap.add_argument("--out", required=True, help="output directory (e.g. runs/bench/<machine>/); a committed result "
                                                 "file is never overwritten")
    ap.add_argument("--machine", help="machine tag recorded in the results (default: from the GPU name; never the "
                                      "host name)")
    ap.add_argument("--part", type=_csv, action="extend", help="job parts: " + ", ".join(
        __import__("benchmarks.bench.protocol", fromlist=["PARTS"]).PARTS))
    ap.add_argument("--engines", type=_csv, action="extend", help="engine ids (filter the parts, or the engines of a "
                                                                  "custom job)")
    ap.add_argument("--models", type=_csv, action="extend", help="problems: mjlab (Track A), menagerie_g1 (Track B)")
    ap.add_argument("--corpora", type=_csv, action="extend", help="corpora ($DK_BENCH_DATA/<name>.npz, default "
                                                                  "benchmarks/data; a+b concatenates)")
    ap.add_argument("--nworld", type=_csv, action="extend", help="batch sizes")
    ap.add_argument("--iterations", type=_csv, action="extend", help="Newton iterations (10, 50; 'native' = the "
                                                                     "model's own, Track B only)")
    ap.add_argument("--ops", type=_csv, action="extend", help="ops: fwd, fwd_train, bwd, vjp, jac ('' = none, dump "
                                                              "only)")
    ap.add_argument("--modes", type=_csv, help="timing modes (default throughput,latency)")
    ap.add_argument("--variant", type=_kv, action="append", help="ENGINE=V1,V2: run these variants of an engine")
    ap.add_argument("--add-engine", type=_kv_ref, action="append", help="ENGINE=REF: also run ENGINE (an adapter "
                                                                         "outside the parts) wherever the parts run "
                                                                         "engine REF")
    ap.add_argument("--opt", type=_kv, action="append", help="ENGINE.KEY=VALUE: an adapter setup option (JSON value)")
    ap.add_argument("--model-path", type=_kv, action="append", help="PROBLEM=PATH: the model file of a problem")
    ap.add_argument("--dump", type=_csv, action="extend", help="raw dumps per unit: outputs, gradient, jacobian")
    ap.add_argument("--processes", type=int, help="processes per unit (default: 3 for headline units, else 1)")
    ap.add_argument("--subsets", type=int, help="disjoint state subsets per small nworld (default: 4 at <= 256, 2 at "
                                                "<= 1024, else 1)")
    ap.add_argument("--rounds", type=int, help="timed rounds (default 5; --quick 3)")
    ap.add_argument("--warmup-calls", type=int, help="minimum warm-up calls (default 5)")
    ap.add_argument("--warmup-s", type=float, help="minimum warm-up seconds (default 1; --quick 0.5)")
    ap.add_argument("--timeout", dest="timeout_s", type=float, help="seconds per unit (default 1800; --quick 600)")
    ap.add_argument("--quick", action="store_true", help="smoke test: nworld 256 and 4096, one corpus, P = 1, one "
                                                         "subset, 3 rounds, op jac only at 256 worlds")
    ap.add_argument("--job", help="job name (default: the parts joined by '+', 'quick' or 'custom')")
    ap.add_argument("--fresh", action="store_true", help="start the job file over instead of resuming it")
    ap.add_argument("--retry-errors", action="store_true", help="re-run cells with status error / skipped:time")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--no-anchor", action="store_true", help="no dk anchor unit")
    ap.add_argument("--no-sampler", action="store_true", help="no nvidia-smi clock sampling")
    ap.add_argument("--skip-missing", action="store_true", help="skip engines without an adapter instead of failing")
    ap.add_argument("--keep-logs", action="store_true", help="keep every worker log in <out>/logs (default: failed "
                                                             "units only)")
    ap.add_argument("--keep-clock-samples", action="store_true", help="keep the raw clock samples in <out>/raw/clocks")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in DELEGATES:
        return _delegate(argv[0], argv[1:])
    ap = argparse.ArgumentParser(prog="python -m benchmarks.bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ec = sub.add_parser("env-check", help="check the machine and every engine's environment")
    ec.add_argument("--engines", type=_csv, action="extend")
    ec.add_argument("--strict", action="store_true", help="an environment that is not installed is a failure")
    run_parser(sub)
    va = sub.add_parser("validate", help="validate result files (dk-bench/1)")
    va.add_argument("files", nargs="+")
    va.add_argument("--portable", action="store_true")
    sub.add_parser("list", help="engines (adapters, environments) and parts")
    for name in DELEGATES:
        sub.add_parser(name, help=f"(delegated to {DELEGATES[name][0]})", add_help=False)
    a = ap.parse_args(argv)
    from benchmarks.bench import protocol
    if a.cmd == "env-check":
        return protocol.env_check(a)
    if a.cmd == "validate":
        from benchmarks.bench import schema
        return schema.main(a.files + (["--portable"] if a.portable else []))
    if a.cmd == "list":
        from benchmarks.bench import adapters
        reg = adapters.discover()
        envs = protocol.Envs()
        for e in sorted(set(reg) | set(envs.engines())):
            print(f"{e:16s} adapter {reg.get(e, '-'):40s} env {envs.engine_env(e)[0] or '-'}")
        for p, blocks in protocol.PARTS.items():
            print(f"part {p}: " + "; ".join(f"{','.join(b.engines)} x {','.join(b.corpora)} x nworld "
                                            f"{','.join(map(str, b.nworld))} x it {','.join(map(str, b.iterations))}"
                                            for b in blocks))
        return 0
    if a.cmd == "run":
        a.variant = {k: _csv(v) for k, v in (a.variant or [])}
        opts = {}
        for k, v in a.opt or []:
            if "." not in k:
                ap.error(f"--opt {k}={v}: expected ENGINE.KEY=VALUE")
            e, key = k.split(".", 1)
            opts.setdefault(e, {})[key] = _value(v)
        a.opt = opts
        a.model_path = dict(a.model_path or [])
        try:
            return protocol.run(a, argv)
        except protocol.PlanError as e:
            print(f"run: {e}", file=sys.stderr)
            return 2
        except KeyboardInterrupt as e:
            print(f"run: interrupted ({e or 'Ctrl-C'}); the finished units are saved: the same command resumes",
                  file=sys.stderr)
            return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
