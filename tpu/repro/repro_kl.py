import os
import sys

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

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
    batch_size=1,
    max_response_length=20,
    good_prompts={"dataset": "mlabonne/harmless_alpaca", "split": "train[:10]", "column": "text"},
    bad_prompts={"dataset": "mlabonne/harmful_behaviors", "split": "train[:10]", "column": "text"},
)
m = Model(s)
print("MODEL READY", flush=True)

good = load_prompts(s, s.good_prompts)
bad = load_prompts(s, s.bad_prompts)
print(f"good {len(good)} bad {len(bad)}", flush=True)

pp = good[:2] + bad[:2]
resp = m.get_responses_batched(pp)
print("prefix-check responses OK", len(resp), flush=True)

probes = good + bad
print(f"KL probe prompts: {len(probes)}", flush=True)
import torch_xla.core.xla_model as xm
def mem(tag):
    try:
        info = xm.get_memory_info(xm.xla_device())
        print(f"  MEM[{tag}] used={info['bytes_used']/1e6:.0f}MB limit={info['bytes_limit']/1e6:.0f}MB", flush=True)
    except Exception as e:
        print(f"  MEM[{tag}] err {e}", flush=True)
mem("after-prefix")
logits_list = []
for i, batch in enumerate([probes[j : j + s.batch_size] for j in range(0, len(probes), s.batch_size)]):
    mem(f"b{i}-before")
    l = m.get_logits(batch)
    mem(f"b{i}-after")
    print(f"  batch {i} raw {tuple(l.shape)}", flush=True)
    logits_list.append(l)
    del l

print(f"cat of {len(logits_list)} ...", flush=True)
t = torch.cat(logits_list, dim=0)
print("cat OK", tuple(t.shape), flush=True)

print("CPU listcomp variant ...", flush=True)
logits_list = []
for i, batch in enumerate([probes[j : j + 2] for j in range(0, len(probes), 2)]):
    l = m.get_logits(batch).cpu()
    print(f"  batch {i} cpu {tuple(l.shape)} {l.dtype}", flush=True)
    logits_list.append(l)
t = torch.cat(logits_list, dim=0)
print("cpu-cat OK", tuple(t.shape), flush=True)
print("ALL OK", flush=True)
