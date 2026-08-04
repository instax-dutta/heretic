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
print("MODEL READY")

prompts = [{"role": "user", "content": f"Repro prompt number {i} with some padding words to vary length" + " extra words here" * (i % 3)} for i in range(10)]
from heretic.utils import Prompt
prompts = [Prompt(system="", user=p["content"]) for p in prompts]

print("running generation first (like common-prefix check)...")
resp = m.get_responses(prompts[:2])
print("generation OK:", resp[:1])

logits_list = []
for i, batch in enumerate([prompts[j : j + 2] for j in range(0, len(prompts), 2)]):
    _, outputs = m.forward(batch, use_cache=False)
    logits = outputs.logits[:, -1, :]
    xm.mark_step()
    logits_list.append(logits)
    print(f"batch {i} done, logits {tuple(logits.shape)}")

print(f"calling torch.cat on {len(logits_list)} tensors ...")
t = torch.cat(logits_list, dim=0)
print("cat OK:", tuple(t.shape))
xm.mark_step()
print("ALL OK")
