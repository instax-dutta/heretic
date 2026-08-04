#!/usr/bin/env python3
"""Multi-device without SPMD: classic all_reduce across 8 chips in one process."""
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

print("devices:", xr.global_device_count(), flush=True)
x = torch.ones(4).to(xm.xla_device())
y = xm.all_reduce(xm.REDUCE_SUM, x)
print("trace ok", flush=True)
xm.mark_step(wait=True)
print("all_reduce sum:", y.sum().item(), flush=True)
print("ALLREDUCE OK", flush=True)
