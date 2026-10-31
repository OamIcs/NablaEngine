"""Temporary work directories of the check and benchmark scripts (scripts/).

work_dir(prefix) makes a directory under $TMPDIR (tempfile.mkdtemp) and removes it when the Python process exits,
also after an exception, unless keeping it was asked for: --keep-work on the scripts that take it (add_argument,
configure), or DK_KEEP_WORK=1 in the environment (inherited by the scripts' own subprocesses, such as the sections of
jax_ffi_check.py). A kept directory's path is printed to stderr. A process killed by a signal (SIGKILL, or SIGTERM
from `timeout`) leaves its directory behind. A check binary's work directory holds its inputs and outputs (an
engine_step_check.py run 15-110 MB, an adjoint check binary's about 300 MB), so leaving every one behind fills a disk.
"""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ENV = "DK_KEEP_WORK"


def keep_requested():
    """$DK_KEEP_WORK set to anything but '' or '0'."""
    return os.environ.get(ENV, "") not in ("", "0")


def work_dir(prefix, dir=None):
    """A new directory (tempfile.mkdtemp(prefix=prefix, dir=dir or $TMPDIR)), removed at exit unless keep_requested()."""
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=dir or os.environ.get("TMPDIR")))
    if keep_requested():
        print(f"work directory kept ({ENV}): {d}", file=sys.stderr, flush=True)
    else:
        atexit.register(shutil.rmtree, d, True)
    return d


def add_argument(ap):
    """--keep-work on an argparse parser (see configure)."""
    ap.add_argument("--keep-work", action="store_true",
                    help=f"keep the temporary work directories (also {ENV}=1; by default they are removed at exit)")


def configure(args):
    """After parse_args: --keep-work sets DK_KEEP_WORK=1, so this process and its subprocesses keep theirs."""
    if getattr(args, "keep_work", False):
        os.environ[ENV] = "1"
