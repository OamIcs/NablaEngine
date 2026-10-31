# PhysX baseline (Isaac Lab 3.0 / Isaac Sim 6.2), Unitree G1, physics only

GPU NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.58.03), CPU AMD EPYC 9555 64-Core Processor; isaaclab 3.0.0, isaacsim 6.2.0.0, torch 2.11.0+cu130; mode full, 2026-10-29T00:33:33Z

Control step = 4 physics steps of 5 ms (20 ms simulated); random PD targets held per control step; robots reset at the start of each round and fall. NOT the same physics problem as an MJCF model: see README.

| configuration | backend | asset | self-collision | N | ms / control step: median (p10-p90 of rounds) | env-steps/s | sim.step ms (synced per op) | GPU MiB | startup s |
|---|---|---|---|---|---|---|---|---|---|
| physx_task_n1024 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 1024 | 18.09 (17.87-18.11) | 56591 | 16.42 | 3376 | 18 |
| physx_task_n4096 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 4096 | 27.52 (27.22-27.61) | 148824 | 25.60 | 4764 | 64 |
| physx_task_n8192 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 8192 | 36.31 (36.27-36.53) | 225595 | 34.21 | 6670 | 137 |
| physx_full_selfcoll_n4096 | PhysX (isaacsim_physx) | G1_CFG | on | 4096 | 54.14 (54.09-54.63) | 75661 | 48.03 | 6008 | 84 |
| newton_task_n1024 | Newton / MuJoCo Warp (newton_mjwarp) | G1_MINIMAL_CFG | off (task default) | 1024 | 7.73 (7.71-7.74) | 132465 | 7.86 | 930 | 7 |
| newton_task_n4096 | Newton / MuJoCo Warp (newton_mjwarp) | G1_MINIMAL_CFG | off (task default) | 4096 | 12.86 (12.84-12.87) | 318562 | 12.96 | 1554 | 9 |
| official_runtime_physx_n4096 (Isaac Lab runtime benchmark, full env.step) | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 4096 | 34.2 (mean env.step) | 119740 | 6.80 per physics step | | |
| physx_task_n16384 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 16384 | 53.02 (52.84-53.35) | 308995 | 50.58 | 10328 | 311 |
| physx_full_selfcoll_n8192 | PhysX (isaacsim_physx) | G1_CFG | on | 8192 | 84.94 (84.19-85.22) | 96446 | 78.62 | 8276 | 178 |
