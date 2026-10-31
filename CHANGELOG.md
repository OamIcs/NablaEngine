# Changelog

## 0.1.0

First release.

- The fused CUDA engine for the Unitree G1 (mjlab model): one control step for thousands of worlds per call, with
  MuJoCo's physics and an analytic adjoint of the step.
- JAX interface (`nablaengine.jax_ffi`, `docs/api.md`, extra `nablaengine[jax]`) and PyTorch interface
  (`nablaengine.torch_ffi`, `docs/api_torch.md`, extra `nablaengine[torch]`), with bitwise the same results; the
  PyTorch one needs no JAX and runs a library built once per GPU architecture.
- Diagnostics on the device, including worlds beyond the contact (64) or constraint-row (192) capacity, whose extra
  contacts / rows are dropped: counted and flagged, with optional warnings (`set_overflow_warnings`).
- Verification against MuJoCo Warp, float64 MuJoCo and MJX (`scripts/verify.sh`, `docs/verification.md`) and the
  benchmark harness (`benchmarks/`).
