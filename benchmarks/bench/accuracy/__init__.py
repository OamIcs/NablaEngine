"""Accuracy scorers of the benchmark against the engine-independent MuJoCo C truth (benchmarks/bench/truth):
  forward.py   control-step outputs per state subset, with MuJoCo C's solver-truncation floors and run-to-run noise
  gradient.py  VJPs and Jacobians per group and input block, floored at the truth's own uncertainty; a paired view
  fd_sweep.py  MuJoCo C finite differences across step sizes, and their cost
  job.py       fills a job document's accuracy and gradient lists from its raw dumps
  stats.py     summary statistics
"""
