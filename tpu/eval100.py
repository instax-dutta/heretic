#!/usr/bin/env python3
"""One-shot x/100 evaluation of the exported abliterated model vs the base.

Loads the exported model from exported_model/, scores keyword refusals on the
FULL 100-prompt harmful_behaviors test split (bypasses the TPU/20 clamp), and
prints refusals for base vs abliterated.
"""
import sys

sys.path.insert(0, "/root/heretic/src" if "/root/heretic" in __import__("os").getcwd() else "")
sys.path.insert(0, "/root/heretic")

import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
sys.path.insert(0, "/root/heretic")

from heretic.config import Settings, DatasetSpecification
from heretic.model import Model
from heretic.scorers.keyword_rate import DEFAULT_KEYWORD_MARKERS
from heretic.utils import load_prompts

MODEL_DIR = "/root/heretic/exported_model"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SPLIT = "test[:100]"


def make_model(model_id: str) -> Model:
    return Model(Settings(model=model_id, max_response_length=20, batch_size=2))


def make_prompts(model_id: str) -> list:
    settings = Settings(model=model_id, max_response_length=20)
    return load_prompts(
        settings,
        DatasetSpecification(dataset="mlabonne/harmful_behaviors", split=SPLIT, column="text"),
    )


def refusal_stats(model, prompts: list) -> dict:
    refusals = 0
    chunk = 20
    for i in range(0, len(prompts), chunk):
        part = prompts[i : i + chunk]
        responses = model.get_responses(part)
        for prompt, response in zip(part, responses):
            text = response.strip()
            if not text:
                refusals += 1
                continue
            text = text.lower().replace("*", "").replace("\u2019", "'")
            text = " ".join(text.split())
            if any(marker.lower() in text for marker in DEFAULT_KEYWORD_MARKERS):
                refusals += 1
    return {"refusals": refusals, "total": len(prompts)}


if __name__ == "__main__":
    print(f"### Base model: {BASE_MODEL}")
    prompts = make_prompts(BASE_MODEL)
    print(f" * {len(prompts)} harmful prompts loaded ({SPLIT})")
    base = make_model(BASE_MODEL)
    s1 = refusal_stats(base, prompts)
    print(f" * BASE   : {s1['refusals']}/{s1['total']} refusals")
    del base
    ablated = make_model(MODEL_DIR)
    s2 = refusal_stats(ablated, prompts)
    print(f" * ABLATED: {s2['refusals']}/{s2['total']} refusals")
    print(f"RESULT base={s1['refusals']} ablated={s2['refusals']} /100")