#!/usr/bin/env python3
"""Plain SPMD (no FSDP): 0.5B on 8 cores, input sharded, one forward.
If this crashes too, SPMD execution itself is broken on this build."""
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as spmd

print(f"cores={xr.global_device_count()} processes={xr.process_count()}", flush=True)
device = xm.xla_device()
xr.use_spmd()
mesh = spmd.get_1d_mesh("data")
print("device:", device, "mesh:", mesh.axis_names, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype=torch.bfloat16
).to(device)
print(f"load+move {time.time()-t0:.1f}s", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
ids = tok(["Hello", "World"], return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
spmd.mark_sharding(ids.input_ids, mesh, ("data", None))
spmd.mark_sharding(ids.attention_mask, mesh, ("data", None))
print("inputs sharded", flush=True)

t0 = time.time()
out = model(**ids)
print(f"forward trace {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
xm.mark_step(wait=True)
print(f"mark_step(wait=True) {time.time()-t0:.1f}s", flush=True)

try:
    v = out.logits.sum().item()
    print(f"fetch sum={v:.1f}", flush=True)
except Exception as e:
    print(f"fetch failed: {e}", flush=True)

print("PLAIN_SPMD OK", flush=True)
