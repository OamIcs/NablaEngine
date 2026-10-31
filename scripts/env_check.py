"""Environment smoke test for a new machine: versions, GPU, nvcc and target architecture, the fixture model, a trivial
CUDA kernel compiled and run (with the device's shared-memory limits), JAX's devices, and, when installed, Warp and
MuJoCo Warp. Those two are optional: the engine, the JAX interface and the checks do not need them (the checks read the
committed MuJoCo Warp references, fixtures/mjlab/mjw317_<corpus>.npz); only regenerating a reference does
(scripts/make_mjw_reference.py, in an environment with mujoco-warp 3.17.0).

  PYTHONPATH=src python scripts/env_check.py

Exit status 1 when a required part fails: no nvcc (or one older than release 12), the fixture model does not load, the
test kernel does not compile or computes a wrong result, JAX is missing or sees no usable device; else 0.
"""

import re
import subprocess
import sys

import mujoco

from nablaengine import cuda, fixtures, workdir

KERNEL = r"""
#include <cstdio>
__global__ void k(float* x) { x[threadIdx.x] = threadIdx.x * 2.f; }
int main() {
  float* d; cudaMalloc(&d, 32 * 4); k<<<1, 32>>>(d); float h[32]; cudaMemcpy(h, d, 128, cudaMemcpyDeviceToHost);
  int smem; cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int smsm; cudaDeviceGetAttribute(&smsm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  printf("kernel ok: %s; SMs %d, shared memory per block (opt-in) %d B, per SM %d B\n",
         h[31] == 62.f ? "yes" : "NO", sms, smem, smsm);
  return 0;
}
"""


def step(failures, name, f):
    """Run one part of the check; an exception is a failure of that part."""
    try:
        return f()
    except Exception as e:  # noqa: BLE001
        failures.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
        print(f"{name}: FAILED ({type(e).__name__}: {str(e)[:300]})", flush=True)
        return None


def main():
    failures = []
    try:  # optional: only regenerating the MuJoCo Warp references needs them
        import warp as wp
        import mujoco_warp as mjw
        ref_error = None
    except ImportError as e:
        wp = mjw = None
        ref_error = repr(e)
    try:
        import jax
    except ImportError:
        jax = None
        failures.append("jax: not installed (the JAX interface and its checks need it)")
    print("mujoco", mujoco.__version__, "| warp", wp.config.version if wp else "not installed",
          "| jax", jax.__version__ if jax else "not installed",
          "| mujoco_warp", getattr(mjw, "__version__", "installed") if mjw else "not installed", flush=True)
    gpu = ["-i", cuda.visible_gpu()] if cuda.visible_gpu() else []  # the first visible GPU, else all
    print(step(failures, "nvidia-smi", lambda: subprocess.run(
        ["nvidia-smi", *gpu, "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip()), flush=True)

    def nvcc_check():
        path = cuda.nvcc()
        out = subprocess.run([path, "--version"], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
        rel = re.search(r"release (\d+)\.(\d+)", out)
        print("nvcc", path, "| arch", cuda.arch(), "|", out.strip().split("\n")[-2], flush=True)
        if not rel or int(rel.group(1)) < 12:
            raise RuntimeError(f"nvcc release {rel.group(0) if rel else '?'}: CUDA 12 or newer is needed")
        return path

    have_nvcc = step(failures, "nvcc", nvcc_check) is not None
    mm = step(failures, "fixture model", fixtures.load_model)
    if mm is not None:
        m, mt = mm
        print(f"fixture model: nq {m.nq} nv {m.nv} nbody {m.nbody} npair {m.npair}; foot pairs {mt['foot_pairs']}",
              flush=True)

    def kernel():
        work = workdir.work_dir("dk_env_")
        (work / "k.cu").write_text(KERNEL)
        cuda.compile_cuda(work / "k.cu", work / "k")
        out = subprocess.run([str(work / "k")], stdout=subprocess.PIPE, universal_newlines=True,
                             check=True).stdout.strip()
        print(out, flush=True)
        if "kernel ok: yes" not in out:
            raise RuntimeError("the test kernel computed a wrong result")

    if have_nvcc:
        step(failures, "test kernel", kernel)
    print("(the engine's shared memory per world: engine_step_check.py's smem / smem_post, jax_ffi.init()'s sizeof_sim)")
    if jax is not None:
        devs = step(failures, "jax devices", jax.devices)
        if devs is not None:
            print("jax devices:", devs, flush=True)
            if not any(d.platform == "gpu" for d in devs):
                failures.append("jax: no GPU device (the CUDA plugin, jax-cuda12-plugin, is needed)")
    if mjw is None:
        print("Warp / MuJoCo Warp not installed (" + ref_error + "): not needed by the engine, the JAX interface and "
              "the checks; scripts/make_mjw_reference.py needs them to regenerate a MuJoCo Warp reference")
    elif mm is not None:
        print("warp device:", wp.get_device("cuda:0").name, "| mujoco_warp put_model:", mjw.put_model(mm[0]) is not None)
    print("env check: " + ("PASS" if not failures else "FAIL: " + "; ".join(failures)), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
