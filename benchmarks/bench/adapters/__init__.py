"""The adapter registry: every module benchmarks/bench/adapters/<name>.py (not base, not starting with '_') that
defines, at top level, a dict literal with string-literal keys ADAPTERS = {"<engine id>": <Adapter subclass>, ...}.

The keys are read from the module's source (ast), so discovering engines imports nothing: the worker then imports only
the one module of the engine it runs, in that engine's interpreter (base.py documents the interface).

  from benchmarks.bench import adapters
  adapters.discover()        # {engine id: module name}
  cls = adapters.load("mjw") # imports adapters/<its module>.py only, checks the class
"""

import ast
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = ("__init__", "base")


def _keys(path):
    """The string keys of the top-level ADAPTERS dict literal of a module's source; None without one."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as e:
        raise ImportError(f"adapters/{path.name}: cannot parse ({e})") from None
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if any(isinstance(t, ast.Name) and t.id == "ADAPTERS" for t in targets):
            if not isinstance(value, ast.Dict) or not all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                                          for k in value.keys):
                raise ImportError(f"adapters/{path.name}: ADAPTERS must be a dict literal with string-literal keys")
            return [k.value for k in value.keys]
    return None


def discover():
    """{engine id: module name (benchmarks.bench.adapters.<name>)}; ImportError for an engine id defined twice."""
    out = {}
    for p in sorted(HERE.glob("*.py")):
        if p.stem in SKIP or p.stem.startswith("_"):
            continue
        for k in _keys(p) or []:
            mod = f"{__name__}.{p.stem}"
            if k in out:
                raise ImportError(f"engine {k!r} is defined in both {out[k]} and {mod}")
            out[k] = mod
    return out


def module_of(engine_id):
    reg = discover()
    if engine_id not in reg:
        raise KeyError(f"no adapter for engine {engine_id!r} (known: {', '.join(sorted(reg)) or 'none'}); an adapter "
                       f"is a file benchmarks/bench/adapters/<name>.py with ADAPTERS = {{{engine_id!r}: <class>}}")
    return reg[engine_id]


def load(engine_id):
    """The adapter class of an engine: imports its module only and checks the class (base.validate_class)."""
    from benchmarks.bench.adapters import base
    mod = importlib.import_module(module_of(engine_id))
    cls = getattr(mod, "ADAPTERS", {}).get(engine_id)
    problems = base.validate_class(cls, engine_id)
    if problems:
        raise TypeError(f"adapter {engine_id}: " + "; ".join(problems))
    return cls
