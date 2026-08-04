import os
import sys

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, "/root/heretic/src")
from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt

s = Settings(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    dtypes=["bfloat16"],
    quantization="none",
    device_map="auto",
    tpu_cores=1,
    tpu_use_fsdp=False,
    batch_size=2,
    max_response_length=20,
)
m = Model(s)
print("MODEL READY", flush=True)

prompts = [Prompt(system="", user=f"Repro prompt {i} with varying words" + " more words here" * (i % 2)) for i in range(4)]

def get_logits_wait(batch, wait):
    _, outputs = m.forward(batch, use_cache=False)
    logits = outputs.logits[:, -1, :]
    xm.mark_step(wait=wait)
    return logits

b0, b1 = prompts[:2], prompts[2:]

print("wait=True then .cpu()", flush=True)
l = get_logits_wait(b0, True)
c = l.cpu()
print("  ok", tuple(c.shape), c.dtype, flush=True)

print("wait=False then .cpu()", flush=True)
l = get_logits_wait(b1, False)
c = l.cpu()
print("  ok", tuple(c.shape), c.dtype, flush=True)

print("wait=True then cat of two", flush=True)
l1 = get_logits_wait(b0, True)
l2 = get_logits_wait(b1, True)
t = torch.cat([l1, l2], dim=0)
print("  ok", tuple(t.shape), flush=True)

print("wait=False then cat of two", flush=True)
l1 = get_logits_wait(b0, False)
l2 = get_logits_wait(b1, False)
t = torch.cat([l1, l2], dim=0)
print("  ok", tuple(t.shape), flush=True)

print("ALL OK", flush=True)
