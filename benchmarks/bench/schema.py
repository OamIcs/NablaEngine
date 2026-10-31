"""The result schema dk-bench/1: one JSON file per (machine, job), rewritten after every benchmark unit so a run can
resume, validated on write and on read. Standard library only (it imports in every engine environment).

  from benchmarks.bench import schema
  doc = schema.new_job(job=..., machine=..., software=..., protocol=..., provenance=...)
  schema.write(path, doc)   # clean (numpy -> Python, non-finite -> null), validate (+ portability), atomic replace
  doc = schema.read(path)   # parse + validate

Top level: schema, created, updated, job {name, machine, parts, quick, cmd}, machine {tag, gpu, cpu, os,
scheduler_cpus}, software (the driver's environment), provenance (NablaEngine's, plus the harness commit),
environments {env name: software of that interpreter}, engines {engine id: the adapter's provenance}, models {problem
id: model file, sha256, options}, corpora {name: path, sha256, n, kind}, protocol {warm-up, rounds, ...}, anchor (the
dk anchor cell ids or null), cells [...], accuracy [...], gradient [...] (the last two are filled by the accuracy
scorers; only their shape is checked here).

A cell is one (engine, variant, model, corpus, subset, nworld, iterations, op, mode, process). Its id is
  <engine>[~<variant>]/<model>/<corpus>[.s<subset>]/<nworld>/it<iterations>/<op>/<mode>/p<process>
(kind "dump": .../dump/p<process>). Status vocabulary (STATUS_KINDS, parse_status):
  ok                       timed; every timing field present
  not_supported[:<short>]  the engine has no native way to do the op (message: the quoted reason)
  skipped:oom | skipped:time   out of memory / over the unit's time budget (message: details)
  error[:<short>]          anything else (message: the exception and the worker's stderr tail)
  flagged:<reasons>        timed, but the GPU reported a hardware throttle reason during the timed rounds
                           (hw_slowdown, hw_thermal, power_brake, joined by '+'); timing fields present
<short> is a single token without spaces (at most 64 characters); the full text goes into `message`.
Final statuses (resume skips them): ok, not_supported, skipped:oom, flagged; error and skipped:time are re-run by
`run --retry-errors`.

Portability (on write): no string may contain the home directory, the host name, or an absolute path under /home,
/tmp, /global, /scratch, /Users (repository-relative paths, no host / user / job identifiers).
"""

import datetime
import json
import math
import os
import re
import socket
import tempfile
from pathlib import Path

SCHEMA = "dk-bench/1"
STATUS_KINDS = ("ok", "not_supported", "skipped", "error", "flagged")
SKIP_REASONS = ("oom", "time")
THROTTLE_FLAGS = ("hw_slowdown", "hw_thermal", "power_brake")
OPS = ("fwd", "fwd_train", "bwd", "vjp", "jac")
MODES = ("throughput", "latency")
NATIVE_KINDS = ("native", "ours", "unmerged-upstream", "labelled")
DEVICES = ("gpu", "cpu")
CELL_KINDS = ("timing", "dump")
TRACKS = ("A", "B")

NUM = (int, float)
OPT_NUM = (int, float, type(None))
OPT_STR = (str, type(None))
OPT_DICT = (dict, type(None))

# field: (types, when) -- when: "all" (every cell), "timed" (timing cells with status ok / flagged), "optional"
CELL_FIELDS = {
    "id": (str, "all"), "kind": (str, "all"), "engine": (str, "all"), "variant": (str, "all"),
    "label": (OPT_STR, "all"), "native": (OPT_STR, "all"), "device": (OPT_STR, "all"), "env": (OPT_STR, "all"),
    "model": (str, "all"), "track": (OPT_STR, "all"), "corpus": (str, "all"), "subset": (int, "all"),
    "nworld": (int, "all"), "iterations": ((int, str), "all"), "substeps": ((int, type(None)), "all"),
    "op": (OPT_STR, "all"), "mode": (OPT_STR, "all"), "process": (int, "all"), "status": (str, "all"),
    "message": (OPT_STR, "all"), "headline": (bool, "all"), "started": (OPT_STR, "all"), "cmd": (str, "all"),
    "unit": (str, "all"),
    # timing (status ok / flagged)
    "reps_per_round": (int, "timed"), "rounds": (int, "timed"), "drop_rounds": (int, "timed"),
    "rounds_ms": (list, "timed"), "median_ms": (NUM, "timed"), "p10_ms": (NUM, "timed"), "p90_ms": (NUM, "timed"),
    "min_ms": (NUM, "timed"), "max_ms": (NUM, "timed"), "control_steps_per_s": (NUM, "timed"),
    "world_substeps_per_s": (OPT_NUM, "timed"), "warmup": (dict, "timed"), "window_unix": (list, "timed"),
    "first_call_s": (OPT_NUM, "timed"), "prepare_s": (OPT_NUM, "timed"), "clocks": (OPT_DICT, "timed"),
    # optional everywhere
    "calls_ms": (list, "optional"), "setup": (OPT_DICT, "optional"), "mem": (OPT_DICT, "optional"),
    "diag": (OPT_DICT, "optional"), "load": (OPT_DICT, "optional"), "raw": (OPT_STR, "optional"),
    "alias_of": (OPT_STR, "optional"), "hints": (OPT_DICT, "optional"), "elapsed_s": (OPT_NUM, "optional"),
    "budget": (OPT_DICT, "optional"), "states": (OPT_DICT, "optional"), "contents": (list, "optional"),
    "opts": (OPT_DICT, "optional"), "gpu_procs_before": ((int, type(None)), "optional"),
    "gradient_not_supported": (OPT_DICT, "optional"), "gradient_report": (OPT_DICT, "optional"),
}
TOP_FIELDS = {
    "schema": str, "created": str, "updated": str, "job": dict, "machine": dict, "software": dict,
    "provenance": OPT_DICT, "environments": dict, "engines": dict, "models": dict, "corpora": dict, "protocol": dict,
    "anchor": (list, type(None)), "cells": list, "accuracy": list, "gradient": list,
}
JOB_FIELDS = {"name": str, "machine": str, "parts": list, "quick": bool, "cmd": str}
MACHINE_FIELDS = {"tag": str, "gpu": OPT_DICT, "cpu": dict, "os": str, "scheduler_cpus": int}
PROTOCOL_FIELDS = {"warmup_calls": int, "warmup_s": NUM, "rounds": int, "drop_rounds": int, "modes": list,
                   "round_min_s": NUM, "round_max_s": NUM, "min_reps_gpu": int, "min_reps_cpu": int,
                   "processes_headline": int, "timer": str, "clock_sample_ms": (int, type(None)),
                   "timeout_s": NUM, "seed": int}
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
SHORT_RE = re.compile(r"^[A-Za-z0-9_.+:=<>-]{1,64}$")
ABS_PATH_RE = re.compile(r"(^|[\s'\"=:(\[,])/(home|tmp|global|scratch|Users|var/tmp)/")


class SchemaError(ValueError):
    """A document that is not valid dk-bench/1; .problems lists every violation found."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems[:20]) + (f" (+{len(self.problems) - 20} more)"
                                                          if len(self.problems) > 20 else ""))


def now():
    """UTC time as 2026-11-02T10:00:00Z."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------- status vocabulary

def status(kind, detail=None):
    """The status string of kind (STATUS_KINDS) with an optional short detail (see the module docstring)."""
    s = kind if not detail else f"{kind}:{detail}"
    parse_status(s)
    return s


def short(text, limit=64):
    """A status detail from free text: the first token-like run of characters (e.g. an exception class name)."""
    t = re.sub(r"[^A-Za-z0-9_.+:=<>-]+", "_", str(text)).strip("_")[:limit]
    return t or None


def parse_status(s):
    """(kind, detail or None); ValueError for a status outside the vocabulary."""
    if not isinstance(s, str) or not s:
        raise ValueError(f"status must be a non-empty string, got {s!r}")
    kind, _, detail = s.partition(":")
    detail = detail or None
    if kind not in STATUS_KINDS:
        raise ValueError(f"unknown status kind {kind!r} in {s!r} (one of {STATUS_KINDS})")
    if kind == "ok" and detail:
        raise ValueError(f"status ok takes no detail: {s!r}")
    if kind == "skipped" and detail not in SKIP_REASONS:
        raise ValueError(f"status skipped needs a reason in {SKIP_REASONS}: {s!r}")
    if kind == "flagged":
        flags = (detail or "").split("+")
        if not detail or any(f not in THROTTLE_FLAGS for f in flags):
            raise ValueError(f"status flagged needs reasons from {THROTTLE_FLAGS} joined by '+': {s!r}")
    if detail is not None and not SHORT_RE.match(detail):
        raise ValueError(f"status detail must be a short token (no spaces, <= 64 characters): {s!r}")
    return kind, detail


def is_timed(s):
    """True for the statuses that carry timing numbers (ok, flagged)."""
    return parse_status(s)[0] in ("ok", "flagged")


def is_final(s):
    """True for statuses a resumed run keeps (ok, not_supported, skipped:oom, flagged); error and skipped:time are
    re-run with --retry-errors."""
    kind, detail = parse_status(s)
    return kind in ("ok", "not_supported", "flagged") or (kind == "skipped" and detail == "oom")


# ------------------------------------------------------------------------------------------------- ids

def cell_id(engine, variant, model, corpus, subset, nworld, iterations, op, mode, process):
    """The cell id (module docstring); op None = a dump cell."""
    e = engine if variant in (None, "", "default") else f"{engine}~{variant}"
    c = corpus if not subset else f"{corpus}.s{subset}"
    tail = "dump" if op is None else f"{op}/{mode}"
    return f"{e}/{model}/{c}/{nworld}/it{iterations}/{tail}/p{process}"


# ------------------------------------------------------------------------------------------------- cleaning

def clean(obj):
    """JSON-ready copy: numpy scalars and arrays -> Python, tuples -> lists, non-finite floats -> None, Paths -> str."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    tolist = getattr(obj, "tolist", None)  # numpy arrays and scalars, without importing numpy
    if callable(tolist):
        return clean(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        return clean(item())
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# ------------------------------------------------------------------------------------------------- validation

def _type_name(t):
    ts = t if isinstance(t, tuple) else (t,)
    return "|".join("null" if x is type(None) else x.__name__ for x in ts)


def _check_type(val, types):
    ts = types if isinstance(types, tuple) else (types,)
    if isinstance(val, bool) and bool not in ts:
        return False  # True is an int in Python, not in the schema
    return isinstance(val, ts)


def _check_fields(obj, fields, where, out):
    if not isinstance(obj, dict):
        out.append(f"{where}: expected an object")
        return
    for k, t in fields.items():
        if k not in obj:
            out.append(f"{where}: missing {k}")
        elif not _check_type(obj[k], t):
            out.append(f"{where}.{k}: expected {_type_name(t)}, got {type(obj[k]).__name__}")


def _finite_list(v):
    return isinstance(v, list) and all(_check_type(x, NUM) and math.isfinite(x) for x in v)


def validate_cell(c, where="cell"):
    """Problems of one cell (empty list: valid)."""
    out = []
    if not isinstance(c, dict):
        return [f"{where}: expected an object"]
    where = f"{where} {c.get('id', '?')}"
    for k, (t, when) in CELL_FIELDS.items():
        if when == "all" and k not in c:
            out.append(f"{where}: missing {k}")
        elif k in c and not _check_type(c[k], t):
            out.append(f"{where}.{k}: expected {_type_name(t)}, got {type(c[k]).__name__}")
    unknown = sorted(set(c) - set(CELL_FIELDS))
    if unknown:
        out.append(f"{where}: unknown fields {unknown}")
    if out:
        return out
    try:
        kind, _ = parse_status(c["status"])
    except ValueError as e:
        return [f"{where}: {e}"]
    if c["kind"] not in CELL_KINDS:
        out.append(f"{where}: kind {c['kind']!r} not in {CELL_KINDS}")
    if c["native"] is not None and c["native"] not in NATIVE_KINDS:
        out.append(f"{where}: native {c['native']!r} not in {NATIVE_KINDS}")
    if c["device"] is not None and c["device"] not in DEVICES:
        out.append(f"{where}: device {c['device']!r} not in {DEVICES}")
    if c["track"] is not None and c["track"] not in TRACKS:
        out.append(f"{where}: track {c['track']!r} not in {TRACKS}")
    if isinstance(c["iterations"], str) and c["iterations"] != "native":
        out.append(f"{where}: iterations must be an int or 'native'")
    if c["nworld"] < 1 or c["process"] < 0 or c["subset"] < 0:
        out.append(f"{where}: nworld >= 1, process >= 0, subset >= 0")
    if c["kind"] == "timing":
        if c["op"] not in OPS:
            out.append(f"{where}: op {c['op']!r} not in {OPS}")
        if c["mode"] not in MODES:
            out.append(f"{where}: mode {c['mode']!r} not in {MODES}")
    elif c["op"] is not None or c["mode"] is not None:
        out.append(f"{where}: a dump cell has op and mode null")
    expect = cell_id(c["engine"], c["variant"], c["model"], c["corpus"], c["subset"], c["nworld"], c["iterations"],
                     c["op"] if c["kind"] == "timing" else None, c["mode"], c["process"])
    if c["id"] != expect:
        out.append(f"{where}: id does not match its fields (expected {expect})")
    if kind != "ok" and not c.get("message") and kind != "flagged":
        out.append(f"{where}: status {c['status']} needs a message")
    if c["kind"] == "timing" and kind in ("ok", "flagged"):
        for k, (_, when) in CELL_FIELDS.items():
            if when == "timed" and k not in c:
                out.append(f"{where}: missing {k} (status {c['status']})")
        if not out:
            if not (_finite_list(c["rounds_ms"]) and len(c["rounds_ms"]) == c["rounds"]):
                out.append(f"{where}: rounds_ms must hold `rounds` finite numbers")
            if not 0 <= c["drop_rounds"] < c["rounds"]:
                out.append(f"{where}: 0 <= drop_rounds < rounds")
            if c["reps_per_round"] < 1:
                out.append(f"{where}: reps_per_round >= 1")
            for k in ("median_ms", "p10_ms", "p90_ms", "min_ms", "max_ms", "control_steps_per_s"):
                if not (math.isfinite(c[k]) and c[k] > 0):
                    out.append(f"{where}: {k} must be finite and > 0")
            if not (len(c["window_unix"]) == 2 and _finite_list(c["window_unix"])
                    and c["window_unix"][0] <= c["window_unix"][1]):
                out.append(f"{where}: window_unix = [start, end]")
            if c["mode"] == "latency" and not _finite_list(c.get("calls_ms")):
                out.append(f"{where}: a latency cell carries calls_ms")
            for k in ("calls", "s"):
                if k not in c["warmup"]:
                    out.append(f"{where}: warmup.{k} missing")
    if c["kind"] == "dump" and kind == "ok" and not c.get("raw"):
        out.append(f"{where}: an ok dump cell names its raw file")
    return out


def portability_problems(doc):
    """Strings with the home directory, the host name or a site path (only meaningful on the machine that wrote it)."""
    home = os.path.expanduser("~")
    host = socket.gethostname().split(".")[0]
    host_re = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(host)}(?![A-Za-z0-9_-])") if len(host) >= 4 else None
    tmp = tempfile.gettempdir()
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            if (home not in ("", "/") and home in o) or ABS_PATH_RE.search(o) or (tmp != "/tmp" and tmp in o):
                out.append(f"{path}: machine-specific path in {o[:120]!r}")
            elif host_re is not None and host_re.search(o):
                out.append(f"{path}: host name in {o[:120]!r}")

    walk(doc, "$")
    return out


def validate(doc, portable=False):
    """Raise SchemaError if doc is not valid dk-bench/1 (portable=True: also the portability check, on write)."""
    out = []
    if not isinstance(doc, dict):
        raise SchemaError(["document: expected an object"])
    if doc.get("schema") != SCHEMA:
        out.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    _check_fields(doc, TOP_FIELDS, "$", out)
    if out:
        raise SchemaError(out)
    _check_fields(doc["job"], JOB_FIELDS, "job", out)
    _check_fields(doc["machine"], MACHINE_FIELDS, "machine", out)
    _check_fields(doc["protocol"], PROTOCOL_FIELDS, "protocol", out)
    if isinstance(doc["machine"].get("tag"), str) and not TAG_RE.match(doc["machine"]["tag"]):
        out.append(f"machine.tag {doc['machine']['tag']!r}: lower-case letters, digits, '_', '.', '-'")
    seen = set()
    for i, c in enumerate(doc["cells"]):
        out += validate_cell(c, f"cells[{i}]")
        cid = c.get("id") if isinstance(c, dict) else None
        if cid in seen:
            out.append(f"cells[{i}]: duplicate id {cid}")
        seen.add(cid)
    for sec in ("accuracy", "gradient"):
        for i, r in enumerate(doc[sec]):
            if not isinstance(r, dict) or not isinstance(r.get("kind", sec), str) or "engine" not in r:
                out.append(f"{sec}[{i}]: expected an object with an engine")
    for name, e in doc["environments"].items():
        if not isinstance(e, dict):
            out.append(f"environments.{name}: expected an object")
    if doc["anchor"] is not None and not all(a in seen for a in doc["anchor"]):
        missing = [a for a in doc["anchor"] if a not in seen]
        # an anchor names cells that are planned; before they ran they are not in cells yet
        if doc["job"].get("complete"):
            out.append(f"anchor cells missing: {missing}")
    if portable:
        out += portability_problems(doc)
    if out:
        raise SchemaError(out)


# ------------------------------------------------------------------------------------------------- documents

def new_job(*, job, machine, software, protocol, provenance=None):
    """An empty job document (no cells yet)."""
    t = now()
    return dict(schema=SCHEMA, created=t, updated=t, job=job, machine=machine, software=software,
                provenance=provenance, environments={}, engines={}, models={}, corpora={}, protocol=protocol,
                anchor=None, cells=[], accuracy=[], gradient=[])


def write(path, doc):
    """Clean, validate (with the portability check) and atomically replace path; returns the cleaned document."""
    doc = clean(doc)
    doc["updated"] = now()
    validate(doc, portable=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)  # mkstemp's 0600 -> the usual file mode
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return doc


def read(path):
    """Parse and validate a job document."""
    doc = json.loads(Path(path).read_text())
    validate(doc)
    return doc


def main(argv=None):
    """python -m benchmarks.bench.schema FILE... : validate result files (exit 1 on the first invalid one)."""
    import argparse
    ap = argparse.ArgumentParser(description="Validate dk-bench/1 result files.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--portable", action="store_true", help="also the portability check (no site paths)")
    a = ap.parse_args(argv)
    bad = 0
    for f in a.files:
        try:
            doc = json.loads(Path(f).read_text())
            validate(doc, portable=a.portable)
            print(f"{f}: valid {SCHEMA}, {len(doc['cells'])} cells")
        except (OSError, ValueError) as e:
            bad += 1
            print(f"{f}: INVALID: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
