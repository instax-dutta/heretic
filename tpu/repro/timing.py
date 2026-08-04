#!/usr/bin/env python3
"""Per-op timing breakdown for the TPU trial cost (~40s/trial measured).
Loads the model once, then times: abliterate+reset, keyword generation,
KL logits, at 20 and 64 prompts, to find where trial time goes."""
import sys, time

sys.path.insert(0, "/root/heretic/src")
from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt, load_prompts
from heretic.config import DatasetSpecification
from heretic.system import mark_step

settings = Settings(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    dtypes=["bfloat16"],
    quantization="none",
    device_map="auto",
    tpu_cores=1,
    tpu_use_fsdp=False,
    batch_size=2,
    n_trials=1,
    n_startup_trials=1,
    max_response_length=20,
    row_normalization="full",
    full_normalization_lora_rank=3,
    orthogonalize_direction=True,
    winsorization_quantile=1.0,
)

t0 = time.time()
m = Model(settings)
print(f"MODEL LOAD {time.time()-t0:.1f}s", flush=True)

spec = DatasetSpecification(dataset="mlabonne/harmless_alpaca", split="test[:20]", column="text")
prompts = load_prompts(settings, spec)

t0 = time.time()
m.get_logits(prompts)
mark_step()
print(f"LOGITS 20 prompts: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
m.get_responses(prompts)
print(f"GENERATE 20 prompts x20tok: {time.time()-t0:.1f}s", flush=True)

for n in ():
    spec2 = DatasetSpecification(dataset="mlabonne/harmless_alpaca", split="test[:128]", column="text")
    p2 = load_prompts(settings, spec2)[:n]
    t0 = time.time()
    m.get_logits(p2)
    mark_step()
    print(f"LOGITS {n} prompts: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
m.get_logits(prompts)
mark_step()
print(f"LOGITS 20 again (cached compile): {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
m.get_responses(prompts)
print(f"GENERATE 20 again (cached): {time.time()-t0:.1f}s", flush=True)

import torch
from heretic.model import AbliterationParameters
t0 = time.time()
m.reset_model()
mark_step()
print(f"RESET MODEL: {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
params = {k: AbliterationParameters(max_weight=1.0, max_weight_position=17, min_weight=0.3, min_weight_distance=6) for k in ("attn.o_proj", "mlp.down_proj")}
m.abliterate(torch.zeros(25, 896), 16.5, params)
mark_step()
print(f"ABLITERATE: {time.time()-t0:.1f}s", flush=True)
print("DONE", flush=True)
