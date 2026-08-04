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
from heretic.utils import load_prompts

s = Settings(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    dtypes=["bfloat16"],
    quantization="none",
    device_map="auto",
    tpu_cores=1,
    tpu_use_fsdp=False,
    batch_size=2,
    max_response_length=20,
    good_prompts={"dataset": "mlabonne/harmless_alpaca", "split": "train[:10]", "column": "text"},
    bad_prompts={"dataset": "mlabonne/harmful_behaviors", "split": "train[:10]", "column": "text"},
)
m = Model(s)
print("MODEL READY", flush=True)

good = load_prompts(s, s.good_prompts)
bad = load_prompts(s, s.bad_prompts)
probes = good + bad
print(f"probes {len(probes)}", flush=True)

def mem(tag):
    try:
        info = xm.get_memory_info(xm.xla_device())
        print(f"  MEM[{tag}] used={info['bytes_used']/1e6:.0f}MB", flush=True)
    except Exception as e:
        print(f"  MEM[{tag}] err {e}", flush=True)

print("STEP1: prefix-check generation (4 prompts, ONE exec)", flush=True)
mem("pre")
resp = m.get_responses_batched(good[:2] + bad[:2])
print("  prefix OK", len(resp), flush=True)
mem("post-prefix")

print("STEP2: KL baseline get_logits (20 prompts, ONE exec)", flush=True)
l = m.get_logits_batched(probes)
mem("post-logits")
print("  logits OK", tuple(l.shape), l.dtype, flush=True)

print("STEP3: generation for 20 prompts (ONE exec)", flush=True)
resp = m.get_responses_batched(probes)
mem("post-gen")
print("  gen OK", len(resp), flush=True)

print("STEP4: residual means (10+10 prompts, 2 execs)", flush=True)
gm = m.get_residuals_mean(good)
mem("post-resid")
print("  resid OK", tuple(gm.shape), flush=True)

print("ALL OK", flush=True)
