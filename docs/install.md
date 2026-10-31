# Installation

## With `setup_env.sh`

```bash
git clone https://github.com/OamIcs/NablaEngine.git NablaEngine && cd NablaEngine
scripts/setup_env.sh --dry-run     # print the plan first
scripts/setup_env.sh               # environment in .dk_env/ (or --env DIR)
source .dk_env/activate.sh
python scripts/env_check.py
```

`setup_env.sh` creates a Python 3.12 venv; installs `requirements-lock.txt` (the tested versions: numpy, mujoco
3.17.0, JAX 0.12 with its CUDA 12 plugins, MJX, SciPy) and NVIDIA's CUDA 12 build of Warp 1.18.0 (checked by its
sha256); installs MuJoCo Warp 3.17.0 from PyPI (step `mjw`: only the comparison tools need it, such as
`scripts/make_mjw_reference.py`; the engine, the JAX interface and the checks against the committed references do
not; skip it with `--steps venv,deps,nvcc,activate`); finds a CUDA 12.x nvcc (`$NVCC`, `PATH`, `$CUDA_HOME`,
`/usr/local/cuda`) or installs conda's `cuda-nvcc` into the environment; and writes `activate.sh` (`PATH`, `NVCC`,
`PYTHONPATH`). `scripts/setup_env.sh --help` lists the options (`--cuda 12.9.1` picks the toolkit).

## By hand

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
pip install "warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
pip install --no-deps mujoco-warp==3.17.0   # optional: the comparison tools
export NVCC=/path/to/cuda-12.x/bin/nvcc     # the pip wheel nvidia-cuda-nvcc-cu12 has no nvcc, only ptxas
export PYTHONPATH=$PWD/src
```

The package needs no build step: `PYTHONPATH=<checkout>/src`, a `.pth` file or `pip install -e <checkout>`. The model
and the reference data are found in the checkout (`fixtures/mjlab`, `benchmarks/data`); outside a source checkout set
`DK_FIXTURES=<checkout>/fixtures/mjlab`, or pass the model: `jax_ffi.init(model=<MjModel>, foot_pairs=<foot-floor pair
indices>)`.

## PyTorch only

The PyTorch interface needs neither JAX nor MuJoCo Warp:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install "numpy>=2.2" mujoco==3.17.0 torch      # on a V100: torch from https://download.pytorch.org/whl/cu126
export NVCC=/path/to/cuda-12.x/bin/nvcc PYTHONPATH=$PWD/src
python -m nablaengine.torch_ffi build               # the library for this GPU's architecture, into the cache
```

The build needs mujoco and nvcc; a process that only runs the interface needs PyTorch, NumPy and the NVIDIA driver,
so the artifact can be built once and copied (`DK_TORCH_ARTIFACT`, `docs/api_torch.md`).

## Platform notes

- Exactly `mujoco==3.17.0`: the model file is MuJoCo's binary format of that version.
- The CUDA library is compiled by nvcc on first use (about 15 s) and cached per architecture and content hash
  (`DK_CACHE_DIR`, `docs/configuration.md`); a host C++ compiler is needed too. CUDA 13 dropped sm_70 (V100), and
  sm_120 (RTX PRO 6000) needs CUDA 12.8 or newer.
- `scripts/verify.sh` (about 15 minutes on a V100) runs every check; `--quick` a few minutes. `docs/verification.md`
  describes them.
