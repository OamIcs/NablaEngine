"""The benchmark corpus file format (benchmarks/data/<name>.npz) and its deterministic writer.

A corpus holds S independent states, each the input of one control step, with stored output cotangents, the
generator's metadata and per-state labels. Keys (nq, nv, nu of the model; every array is C-contiguous):

  qpos (S, nq), qvel (S, nv), qacc_warmstart (S, nv), ctrl (S, nu)  float32  inputs of one control step
  friction (S,), com (S, 3)                                         float32  per-state domain randomization (only for
                                                                             a model with DR: Track A): foot-floor
                                                                             pair friction, torso COM offset (m)
  cot_qpos (S, C, nq), cot_qvel (S, C, nv)                          float32  C output cotangents per state (C = 3)
  metadata                                                          0-d str  JSON (format, generator, seed, options,
                                                                             model and its sha256, physics, labels:
                                                                             a description of every label_ key, ...)
  label_<name> (S,)                                                 bool or int32, per-state labels (metadata.labels)

The file is a zip written with fixed member timestamps, so that re-running a generator that reproduces the arrays bit
for bit also reproduces the file's sha256 (as long as zlib compresses alike); content_digest() hashes the arrays
themselves, independent of the zip and of zlib. np.load reads the file without pickle.

  from benchmarks.bench.corpus import format as corpus_format
  z = corpus_format.read("benchmarks/data/mf_s0.npz")   # dict of arrays; z["metadata"] is the parsed dict
  corpus_format.write(path, arrays, metadata)          # validate(), then write; returns the file's sha256
  corpus_format.sha256(path), corpus_format.content_digest(path)
"""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "dk-bench-corpus/1"
INPUTS = ("qpos", "qvel", "qacc_warmstart", "ctrl")
DR_KEYS = ("friction", "com")
COT_KEYS = ("cot_qpos", "cot_qvel")
LABEL_PREFIX = "label_"
MAX_BYTES = 8 * 2**20  # the size budget of one corpus file (<= 8 MiB per corpus)


def data_provenance():
    """Where and with what a data file was made, for the index files next to it (corpora.json, corpus_stats.json):
    UTC date, Python, the packages the numbers depend on (mujoco, numpy) and the CPU model. No paths, host names or
    source commits: the files themselves are identified by their sha256 and the generator's version."""
    import datetime
    import importlib.metadata
    import platform
    pk = {}
    for name in ("mujoco", "numpy"):
        try:
            pk[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pk[name] = None
    cpu = None
    try:
        from benchmarks.bench import prov
        cpu = (prov.cpu() or {}).get("model")
    except Exception:  # noqa: BLE001  (best effort)
        pass
    return dict(date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                python=platform.python_version(), packages=pk, cpu=cpu)
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def validate(arrays, metadata):
    """Raise ValueError unless arrays (dict name -> ndarray, without 'metadata') and metadata (dict) form a corpus."""
    a = arrays
    if "metadata" in a:
        raise ValueError("pass the metadata separately, not as an array")
    missing = [k for k in INPUTS + COT_KEYS if k not in a]
    if missing:
        raise ValueError(f"missing arrays {missing}")
    has_dr = [k in a for k in DR_KEYS]
    if any(has_dr) and not all(has_dr):
        raise ValueError("friction and com come together")
    extra = [k for k in a if k not in INPUTS + COT_KEYS + DR_KEYS and not k.startswith(LABEL_PREFIX)]
    if extra:
        raise ValueError(f"unknown arrays {extra}")
    model = metadata.get("model") or {}
    nq, nv, nu = (int(model[k]) for k in ("nq", "nv", "nu"))
    S = len(a["qpos"])
    C = a["cot_qpos"].shape[1] if a["cot_qpos"].ndim == 3 else -1
    shapes = dict(qpos=(S, nq), qvel=(S, nv), qacc_warmstart=(S, nv), ctrl=(S, nu), friction=(S,), com=(S, 3),
                  cot_qpos=(S, C, nq), cot_qvel=(S, C, nv))
    for k, shape in shapes.items():
        if k not in a:
            continue
        if a[k].dtype != np.float32 or a[k].shape != shape:
            raise ValueError(f"{k}: want float32 {shape}, got {a[k].dtype} {a[k].shape}")
        if not np.isfinite(a[k]).all():
            raise ValueError(f"{k}: non-finite values")
    described = (metadata.get("labels") or {})
    for k in a:
        if k.startswith(LABEL_PREFIX):
            if a[k].shape != (S,) or a[k].dtype not in (np.bool_, np.int32):
                raise ValueError(f"{k}: want bool or int32 ({S},), got {a[k].dtype} {a[k].shape}")
            if k not in described:
                raise ValueError(f"{k}: no description in metadata['labels']")
    if metadata.get("format") != FORMAT:
        raise ValueError(f"metadata['format'] must be {FORMAT!r}")


def write(path, arrays, metadata, max_bytes=MAX_BYTES):
    """Validate and write the corpus to path (atomically: a temporary file next to it, then a rename); raises if the
    file would exceed max_bytes. Returns its sha256."""
    validate(arrays, metadata)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    items = [(k, np.ascontiguousarray(v)) for k, v in arrays.items()]
    items.append(("metadata", np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":")))))
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            for k, v in items:
                buf = io.BytesIO()
                np.lib.format.write_array(buf, v, allow_pickle=False)
                info = zipfile.ZipInfo(f"{k}.npy", date_time=_ZIP_TIME)
                info.external_attr = 0o644 << 16
                zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        size = tmp.stat().st_size
        if size > max_bytes:
            raise ValueError(f"{path.name}: {size} bytes, over the budget of {max_bytes}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return sha256(path)


def read(path):
    """The corpus as a dict of arrays, with 'metadata' parsed from its JSON."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["metadata"] = json.loads(str(out["metadata"]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(path_or_arrays):
    """sha256 over every array's name, dtype, shape and bytes in sorted key order (the metadata as its JSON string):
    equal for equal contents however the zip was compressed."""
    if isinstance(path_or_arrays, (str, os.PathLike)):
        with np.load(path_or_arrays, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        arrays = dict(path_or_arrays)
        if isinstance(arrays.get("metadata"), dict):
            arrays["metadata"] = np.array(json.dumps(arrays["metadata"], sort_keys=True, separators=(",", ":")))
    h = hashlib.sha256()
    for k in sorted(arrays):
        v = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
    return h.hexdigest()
