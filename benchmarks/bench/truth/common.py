"""Shared, dependency-light helpers of the truth package (numpy and the standard library only, so that the native
MuJoCo Warp labelling process, truth/mjw_labels.py, imports them too): digests of corpus inputs and cotangents, the
problem a corpus belongs to, the cache directory and its keyed npz files.

  from benchmarks.bench.truth import common
  prob = common.problem_of(corpus)                 # problem.Problem of a problem.Corpus (Track A or B)
  common.inputs_digest(corpus.arrays), common.cot_digest(corpus)
  common.save_cached(path, meta, arrays); common.load_cached(path, meta)   # None unless the meta matches
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.bench import problem as P

MAXC = 64  # contact slots per substep in the packed contact arrays (truth/fdlib.MAXC)
CACHE_DEFAULT = "runs/bench/truth"  # --cache-dir of the truth tools: forward truth, labels, FD Jacobians
TRUTH_SUBDIR = "truth"  # truth files: <data dir>/truth/<corpus>.npz (data dir: $DK_BENCH_DATA, default benchmarks/data)
VERSION = 1  # of the cached arrays' semantics (part of every cache key)


def inputs_digest(arrays):
    """sha256 (hex) of the float32 bytes of the inputs present, in problem.INPUTS order, each preceded by its name."""
    h = hashlib.sha256()
    for k in P.INPUTS:
        if k in arrays:
            h.update(k.encode())
            h.update(np.ascontiguousarray(arrays[k], np.float32).tobytes())
    return h.hexdigest()


def cot_digest(cot_qpos, cot_qvel):
    """sha256 (hex) of the stored output cotangents (S, C, nq), (S, C, nv) as float32."""
    h = hashlib.sha256()
    for a in (cot_qpos, cot_qvel):
        h.update(np.ascontiguousarray(a, np.float32).tobytes())
    return h.hexdigest()


def problem_of(corpus, model_path=None):
    """The problem.Problem of a problem.Corpus: Track B for a benchmark corpus whose metadata names track B (or a
    Menagerie model), else Track A (the mjlab G1; also every fixture-format corpus)."""
    meta = corpus.meta if isinstance(corpus.meta, dict) else {}
    track = (meta.get("options") or {}).get("track") if isinstance(meta.get("options"), dict) else None
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    if track == "B" or "menagerie" in str(model.get("repo", "")).lower():
        return P.get("menagerie_g1", model_path=model_path)
    return P.get("mjlab", model_path=model_path)


def check_model(corpus, prob):
    """The problem's model description (problem.Problem.describe) after checking that its sha256 is the one the corpus
    metadata records (when it records one); ValueError otherwise."""
    desc = prob.describe()
    meta = corpus.meta if isinstance(corpus.meta, dict) else {}
    want = (meta.get("model") or {}).get("sha256") if isinstance(meta.get("model"), dict) else None
    if not desc.get("exists"):
        raise FileNotFoundError(f"{prob.id}: model file {desc['path']} not found")
    if want and desc.get("sha256") != want:
        raise ValueError(f"corpus {corpus.name} was generated with model sha256 {want}, {desc['path']} has "
                         f"{desc.get('sha256')}")
    return desc


def cache_dir(path=None):
    p = Path(path or CACHE_DEFAULT)
    p = p if p.is_absolute() else P.REPO / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def truth_dir():
    """<data dir>/truth: $DK_BENCH_DATA (default benchmarks/data) + /truth."""
    return P.data_dir() / TRUTH_SUBDIR


def truth_path(corpus_name):
    return truth_dir() / f"{corpus_name}.npz"


def save_cached(path, meta, arrays):
    """np.savez_compressed with the meta as a 0-d JSON string under `meta`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta, sort_keys=True), **arrays)
    tmp.replace(path)


def load_cached(path, meta):
    """The arrays of a save_cached file whose meta equals `meta` (JSON round trip), else None."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if json.loads(str(z["meta"])) != json.loads(json.dumps(meta, sort_keys=True)):
            return None
        return {k: z[k] for k in z.files if k != "meta"}
