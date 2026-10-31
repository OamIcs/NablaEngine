"""The benchmark harness (result schema dk-bench/1): one driver (protocol.py) that starts a fresh worker process
(worker.py) per benchmark unit in the Python environment the engine needs (benchmarks/envs/envs.toml), and adapters
(adapters/) that wrap each engine behind one interface (adapters/base.py).

  python -m benchmarks.bench env-check | run | report | fetch-models | gen-corpus | truth

The modules imported by the worker (schema, prov, problem, worker, adapters.base, adapters/__init__) use only the
standard library and numpy (mujoco is imported lazily by problem.py), so they import in every engine environment.
"""

from pathlib import Path

SCHEMA = "dk-bench/1"
REPO = Path(__file__).resolve().parents[2]  # the checkout this harness runs from
