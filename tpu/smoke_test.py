"""Headless TPU smoke test for Heretic - runs over SSH, no notebook needed.

Mirrors notebooks/tpu_env_check.ipynb as plain asserts. Prints PASS/FAIL per
stage, exits non-zero if any stage fails.

Usage:
    python tpu/smoke_test.py                          # full suite, 0.5B model
    python tpu/smoke_test.py --model X --skip-model  # detection-only stages
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable

STAGES: list[tuple[str, Callable[["Ctx"], bool | None]]] = []
PASSED: list[str] = []
FAILED: list[str] = []


class Ctx:
    """Shared state between stages (holds the loaded Model)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Any = None


def stage(name: str):
    def deco(fn: Callable[["Ctx"], bool | None]):
        STAGES.append((name, fn))
        return fn
    return deco


def report(title: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {title}{f' - {detail}' if detail else ''}")
    (PASSED if ok else FAILED).append(title)


@stage("TPU basics")
def tpu_basics(ctx: Ctx) -> bool:
    import os

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    # Must run before any torch_xla device/tensor access: on single-process
    # multi-chip TPUs this enables SPMD mode (required for sharding). The
    # flag must be set BEFORE the first client/device call below, so derive
    # the chip count from TPU env vars rather than from the XLA client.
    from heretic.system import _get_tpu_core_count_from_env

    if _get_tpu_core_count_from_env() > 1:
        os.environ.setdefault("XLA_USE_SPMD", "1")
    from heretic.system import get_xla_device_count

    cores = get_xla_device_count()
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  torch_xla: {torch_xla.__version__}")
    device = xm.xla_device()
    hw = xm.xla_device_hw(device)
    print(f"  Device:    {device}")
    print(f"  HW:        {hw}")
    print(f"  Cores:     {cores}")
    assert hw == "TPU", f"HW is {hw}, expected TPU"
    assert cores >= 1, "No XLA devices"
    x = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    y = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    z = x @ y
    torch_xla.sync()
    assert z.shape == (128, 128), f"Unexpected matmul shape {z.shape}"
    report("TPU basics", True, f"{hw} {cores} cores")
    return True


@stage("Heretic TPU detection")
def detection(ctx: Ctx) -> bool:
    from heretic.system import detect_tpu, get_accelerator_info, setup_tpu_environment

    setup_tpu_environment()
    assert detect_tpu(), "detect_tpu() returned False"
    info = get_accelerator_info()
    print(f"  {info}")
    assert "TPU" in info, f"Accelerator report missing TPU: {info!r}"
    report("Heretic TPU detection", True)
    return True


@stage("Settings validation")
def settings_check(ctx: Ctx) -> bool:
    from heretic.config import Settings

    sys.argv = [
        "heretic",
        "--model=Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "--dtypes=bfloat16",
        "--quantization=none",
        "--device-map=auto",
        "--n-trials=1",
        "--n-startup-trials=1",
        "--max-response-length=10",
        "--row-normalization=full",
        "--full-normalization-lora-rank=3",
        "--orthogonalize-direction",
        "--winsorization-quantile=1.0",
    ]
    settings = Settings()
    print(f"  Model:       {settings.model}")
    print(f"  Dtypes:      {settings.dtypes}")
    print(f"  Quantization:{settings.quantization}")
    print(f"  TPU cores:   {settings.tpu_cores}")
    assert settings.quantization == "none", "Quantization must be none on TPU"
    assert "bfloat16" in settings.dtypes, "bfloat16 must be in dtypes on TPU"
    report("Settings validation", True)
    return True


@stage("Model load + forward")
def model_load(ctx: Ctx) -> bool | None:
    from heretic.config import Settings
    from heretic.model import Model
    from heretic.system import get_xla_device_count
    from heretic.utils import Prompt

    cores = get_xla_device_count()
    settings = Settings(
        model=ctx.args.model,
        dtypes=["bfloat16"],
        quantization="none",
        device_map="auto",
        tpu_cores=cores,
        tpu_use_fsdp=cores > 1,
        n_trials=1,
        n_startup_trials=1,
        max_response_length=10,
        batch_size=2,
        row_normalization="full",
        full_normalization_lora_rank=3,
        orthogonalize_direction=True,
        winsorization_quantile=1.0,
    )
    m = Model(settings)
    ctx.model = m
    print(f"  Loaded:      {type(m.model).__name__}")
    print(f"  Device:      {next(m.model.parameters()).device}")
    print(f"  Dtype:       {next(m.model.parameters()).dtype}")
    print(f"  Layers:      {len(m.get_layers())}")
    assert m._is_tpu, "Model not on TPU"

    prompts = [
        Prompt(system="You are a helpful assistant.", user="What is 2+2?"),
        Prompt(system="You are a helpful assistant.", user="Hello!"),
    ]
    inputs, outputs = m.forward(prompts, use_cache=False)
    print(f"  Logits:      {outputs.logits.shape} on {outputs.logits.device}")
    assert outputs.logits.device.type in ("xla", "tpu"), (
        f"Logits on {outputs.logits.device}, expected XLA"
    )
    report("Model load + forward", True, f"{len(m.get_layers())} layers")
    return None  # keeps ctx.model for later stages


@stage("get_logits")
def logits(ctx: Ctx) -> bool:
    from heretic.utils import Prompt

    prompts = [
        Prompt(system="You are a helpful assistant.", user="What is 2+2?"),
        Prompt(system="You are a helpful assistant.", user="Hello!"),
    ]
    out = ctx.model.get_logits(prompts)
    print(f"  Shape: {out.shape}, dtype: {out.dtype}")
    assert out.shape[0] == 2, f"Expected batch 2, got {out.shape[0]}"
    assert out.shape[1] > 0, "Empty vocab"
    report("get_logits", True)
    return True


@stage("get_residuals")
def residuals(ctx: Ctx) -> bool:
    from heretic.utils import Prompt

    prompts = [
        Prompt(system="You are a helpful assistant.", user="What is 2+2?"),
        Prompt(system="You are a helpful assistant.", user="Hello!"),
    ]
    out = ctx.model.get_residuals(prompts)
    print(f"  Shape: {out.shape} (prompt, layer, hidden), dtype: {out.dtype}")
    assert out.shape[0] == 2, f"Expected batch 2, got {out.shape[0]}"
    assert out.shape[1] > 0, "No layers"
    report("get_residuals", True)
    return True


@stage("get_responses (XLA greedy loop)")
def responses(ctx: Ctx) -> bool:
    from heretic.utils import Prompt

    prompts = [
        Prompt(system="You are a helpful assistant.", user="What is 2+2?"),
        Prompt(system="You are a helpful assistant.", user="Hello!"),
    ]
    t0 = time.perf_counter()
    out = ctx.model.get_responses(prompts, max_new_tokens=ctx.args.max_response_length)
    elapsed = time.perf_counter() - t0
    for i, r in enumerate(out):
        print(f"  [{i}] {r[:80]!r}")
    assert len(out) == 2 and all(isinstance(r, str) for r in out), "Bad responses"
    report("get_responses", True, f"{elapsed:.1f}s for {ctx.args.max_response_length} tokens")
    return True


@stage("XLA re-trace check")
def retrace(ctx: Ctx) -> bool:
    from heretic.utils import Prompt

    prompts = [Prompt(system="You are a helpful assistant.", user="What is 2+2?")]
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ctx.model.get_logits(prompts)
        times.append(time.perf_counter() - t0)
    print(f"  Iterations: {[f'{t:.2f}s' for t in times]}")
    ok = times[1] < times[0] * 0.8
    report("XLA re-trace check", ok)
    return ok


@stage("KLDivergence scorer")
def kl_scorer(ctx: Ctx) -> bool:
    from heretic.plugin import Context
    from heretic.scorers.kl_divergence import KLDivergence

    scorer = KLDivergence(
        heretic_settings=ctx.model.settings,
        settings=KLDivergence.get_settings_model()(prompts={
            "dataset": "mlabonne/harmless_alpaca",
            "split": "test[:2]",
            "column": "text",
        }),
    )
    c = Context(settings=ctx.model.settings, model=ctx.model)
    scorer.init(c)
    baseline = scorer.get_baseline_score(c)
    score = scorer.get_score(c)
    print(f"  Baseline: {baseline.rich_display}")
    print(f"  Current:  {score.rich_display}")
    report("KLDivergence scorer", True)
    return True


@stage("Memory report")
def memory(ctx: Ctx) -> bool:
    try:
        import torch_xla.core.xla_model as xm
        info = xm.get_memory_info(xm.xla_device())
        used = info["bytes_used"] / 1e9
        total = info["bytes_limit"] / 1e9
        print(f"  TPU HBM: {used:.2f} / {total:.2f} GB in use")
        assert used < total, "HBM overcommit"
    except Exception as e:
        print(f"  (HBM info unavailable: {e})")
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) / 1e6
                print(f"  RAM available: {avail:.1f} GB")
    report("Memory report", True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--max-response-length", type=int, default=20)
    parser.add_argument("--skip-model", action="store_true",
                        help="run detection-only stages (no model download)")
    args = parser.parse_args()

    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ctx = Ctx(args)
    for name, fn in STAGES:
        if args.skip_model and name in (
            "Model load + forward", "get_logits", "get_residuals",
            "get_responses (XLA greedy loop)", "XLA re-trace check",
            "KLDivergence scorer",
        ):
            print(f"\n== {name} == (skipped)")
            continue
        print(f"\n== {name} ==")
        try:
            fn(ctx)
        except Exception as e:
            report(name, False, str(e)[:200])
        if name == "Memory report":
            break  # last stage

    print(f"\n{'=' * 60}")
    if FAILED:
        print(f"SMOKE TEST FAILED: {', '.join(FAILED)}")
        return 1
    print(f"SMOKE TEST PASSED ({len(PASSED)} stages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
