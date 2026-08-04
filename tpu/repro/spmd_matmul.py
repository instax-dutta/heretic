#!/usr/bin/env python3
"""Tiny SPMD matmul: if this crashes, SPMD execution is fundamentally broken."""
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as spmd

print(f"cores={xr.global_device_count()}", flush=True)
device = xm.xla_device()
xr.use_spmd()
mesh = spmd.get_1d_mesh("data")

x = torch.randn(8, 16).to(device)
spmd.mark_sharding(x, mesh, ("data", None))
w = torch.randn(16, 16).to(device)
y = torch.matmul(x, w)
print("trace ok", flush=True)
xm.mark_step(wait=True)
print("mark_step ok", flush=True)
print("sum:", y.sum().item(), flush=True)
print("SPMD_MATMUL OK", flush=True)
