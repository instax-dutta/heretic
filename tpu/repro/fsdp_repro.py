#!/usr/bin/env python3
"""Minimal FSDP/SPMD repro: 0.5B model, 8-core SPMD FSDP, one forward.
Isolates the smoke-test segfault (ExecuteReplicated during first forward).
Usage: PJRT_DEVICE=TPU python3 fsdp_repro.py [wait|nowait] [spmd|classic]"""
import sys, time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

mode_wait = len(sys.argv) > 1 and sys.argv[1] == "wait"

print(f"cores={xr.global_device_count()} processes={xr.process_count()}", flush=True)
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as spmd

device = xm.xla_device()  # MUST be before use_spmd: registers TPU:0 in bridge map
print("device:", device, flush=True)
xr.use_spmd()
print("spmd:", xr.is_spmd(), flush=True)
print("device2:", xm.xla_device(), flush=True)

mesh = spmd.get_1d_mesh("fsdp")
print("mesh:", mesh.mesh_shape, mesh.axis_names, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct", torch_dtype=torch.bfloat16
)
print(f"load {time.time()-t0:.1f}s", flush=True)

from functools import partial
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

layer_cls = {type(m) for m in model.modules() if "DecoderLayer" in type(m).__name__}
policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)

from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel,
)


def shard_output(output, mesh):
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", output)
    spmd.mark_sharding(logits, mesh, (None, None, "fsdp"))


t0 = time.time()
model = SpmdFullyShardedDataParallel(
    model, mesh=mesh, shard_output=shard_output, auto_wrap_policy=policy
)
print(f"wrap {time.time()-t0:.1f}s", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
ids = tok(
    ["Hello", "World"], return_tensors="pt", padding="max_length",
    max_length=128, truncation=True,
).to(xm.xla_device())
spmd.mark_sharding(ids.input_ids, mesh, ("fsdp", None))
spmd.mark_sharding(ids.attention_mask, mesh, ("fsdp", None))
print("inputs on:", ids.input_ids.device, flush=True)

t0 = time.time()
try:
    out = model(**ids)
    print(f"forward trace {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"forward failed: {e}", flush=True)
    raise SystemExit(1)

t0 = time.time()
xm.mark_step(wait=mode_wait)
print(f"mark_step(wait={mode_wait}) {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
try:
    v = out.logits.sum().item()
    print(f"fetch {time.time()-t0:.1f}s sum={v:.1f}", flush=True)
except Exception as e:
    print(f"fetch failed: {e}", flush=True)

print("FSDP_REPRO OK", flush=True)
