# PhysX baseline (Isaac Lab 3.0 / Isaac Sim 6.2), Unitree G1, physics only

GPU NVIDIA GeForce RTX 3090 (driver 565.57.01), CPU Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz; isaaclab 3.0.0, isaacsim 6.2.0.0, torch 2.11.0+cu128; mode full, 2026-10-29T13:26:44Z

Control step = 4 physics steps of 5 ms (20 ms simulated); random PD targets held per control step; robots reset at the start of each round and fall. NOT the same physics problem as an MJCF model: see README.

| configuration | backend | asset | self-collision | N | ms / control step: median (p10-p90 of rounds) | env-steps/s | sim.step ms (synced per op) | GPU MiB | startup s |
|---|---|---|---|---|---|---|---|---|---|
| physx_task_n1024 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 1024 | 35.04 (35.00-35.51) | 29225 | 60.93 | 2992 | 73 |
| physx_task_n4096 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 4096 | 45.16 (44.17-70.21) | 90697 | 37.34 | 4380 | 123 |
| physx_task_n8192 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 8192 | 63.65 (62.62-82.87) | 128696 | 53.90 | 6286 | 295 |
| physx_full_selfcoll_n4096 | PhysX (isaacsim_physx) | G1_CFG | on | 4096 | 87.45 (85.33-157.00) | 46836 | 74.71 | 5230 | 157 |
| newton_task_n1024 | Newton / MuJoCo Warp (newton_mjwarp) | G1_MINIMAL_CFG | off (task default) | 1024 | 13.81 (13.62-14.03) | 74124 | 14.25 | 556 | 97 |
| newton_task_n4096 | Newton / MuJoCo Warp (newton_mjwarp) | G1_MINIMAL_CFG | off (task default) | 4096 | 35.15 (34.73-35.49) | 116530 | 35.90 | 1180 | 15 |
| official_runtime_physx_n4096 (Isaac Lab runtime benchmark, full env.step) | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 4096 | 95.5 (mean env.step) | 42905 | 9.14 per physics step | | |
| physx_task_n16384 | PhysX (isaacsim_physx) | G1_MINIMAL_CFG | off (task default) | 16384 | 106.67 (106.22-164.74) | 153598 | 99.00 | 9944 | 663 |
| physx_full_selfcoll_n8192 | PhysX (isaacsim_physx) | G1_CFG | on | 8192 | 145.75 (144.17-146.62) | 56207 | 131.94 | 7498 | 324 |
