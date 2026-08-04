# Heretic -> TPU v5e-8: Execution Plan

Status of the port as of the last session (Jul 28 2026): the XLA-compatible
inference paths, TPU detection, config, notebooks, and efficiency tweaks are
done and committed. The port was never validated end-to-end on real hardware.
This document is the runbook for finishing it the moment a v5e-8 VM is
reachable.

Execution model: **SSH into the Kaggle TPU VM, run commands directly. No
notebooks.** All validation is plain scripts (`tpu/smoke_test.py`) and the
`heretic` CLI. Notebooks already in the repo are docs only.

## Current state (brief)

- Done: `detect_tpu`, XLA device mgmt, `mark_step`, XLA `empty_cache`,
  bfloat16 forcing, quantization disabled, `forward()` paths for
  logits/residuals, fixed-shape XLA greedy generation loop, fixed-length
  tokenization (512), TPU prompt-count reductions, Colab + Kaggle notebooks,
  `configs/config.tpu.toml`, `[tpu]` pip extra.
- Missing / known issues:
  1. FSDP (`fsdp_utils.py`) is dead code - `wrap_model_fsdp` is never called;
     `tpu_use_fsdp`/`tpu_cores` are config-only. Multi-core is unimplemented.
  2. `get_accelerator_info_dict` is defined twice in `system.py` (line 162 with
     TPU support, line 443 without). The second shadows the first, so TPU
     reporting on the VM will lie ("No accelerator detected").
  3. `save_merged_model` (xm.save path) is dead code; main.py calls
     `get_merged_model()` + `save_pretrained()` directly. On XLA this merges
     on-device (risky with FSDP). Need a TPU-safe merge path.
  4. `svd_lowrank` runs on the module device (XLA) for
     `row_normalization=full` - likely unsupported on TPU; needs CPU fallback.
  5. Kaggle notebook installs from `p-e-w/heretic` (upstream, no TPU code).

## Phase 0 - VM bootstrap (~10 min)

1. SSH in (Kaggle TPU v5e-8 VM: `kaggle` user, TPU attached, torch +
   torch_xla preinstalled; internet on).
2. Check `nproc`, RAM, disk, Python version (3.10-3.12 OK; repo targets 3.12).
3. Verify TPU: `PJRT_DEVICE=TPU`, `xm.xla_device_count()` == 8, HW == "TPU v5e".
4. Clone/pull this repo, run `tpu/bootstrap.sh`:
   - Uses the VM's preinstalled torch + torch_xla - never lets pip clobber them.
   - `pip install -e ".[tpu]"` with torch/torchvision excluded from upgrade.
5. Sanity: `python -c "from heretic.system import detect_tpu; print(detect_tpu())"`
   must print True.

## Phase 1 - Local fixes already applied (no TPU needed)

- [x] Remove shadowing `get_accelerator_info_dict` -> TPU core report works.
- [x] Kaggle notebook: point install at `instax-dutta/heretic`.

## Phase 2 - Headless smoke test on 1 core (~20-30 min)

Run `python tpu/smoke_test.py` (plain-script port of `tpu_env_check.ipynb`,
all cells converted to asserts, prints PASS/FAIL per stage):

1. TPU basics: device, cores, bf16 matmul, mark_step/sync.
2. `detect_tpu()` + `get_accelerator_info()` shows 8 TPU cores (validates the
   Phase-1 fix).
3. `Settings()` with TPU flags: bfloat16 forced, quantization none.
4. Load `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bf16, device_map=auto,
   max_memory cpu=64GB). Confirm model lands on `xla:0`.
5. `forward()`: logits shape/device assert.
6. `get_logits()` / `get_residuals()` shapes + devices.
7. `get_responses()` (XLA greedy loop) - 2 prompts, max 20 tokens, timing.
8. XLA re-trace check: 3x `get_logits` - second call must be much faster.
9. KLDivergence scorer baseline on `test[:2]`.
10. Memory report: HBM + RAM after a few iterations.

Expected issues to handle here:
- **SVD on XLA** (Phase-1 risk #4): if `row_normalization=full` fails in
  `apply_abliteration`, implement CPU fallback: move W to CPU for the
  `svd_lowrank` call, bring U/S/Vh back. This is a 5-line change in
  `model.py` around line 679.
- **PJRT cleanup errors at exit**: known-harmless, already documented; ignore.

If all stages pass on 1 core, proceed. If not, fix and re-run before touching
FSDP - single-core is the ground truth for correctness.

## Phase 3 - Wire FSDP (the main remaining feature, ~1-2 h)

Goal: >8B models sharded across 8 cores (v5e-8 = 128 GB HBM aggregate).

### Design

Order matters for PEFT + FSDP. Chosen pattern:

```
from_pretrained(...)                  # model on CPU (device_map=None on TPU multi-core)
model = wrap_model_fsdp(model, cfg)   # shard decoder layers (fsdp_utils.py)
self.model = get_peft_model(model, peft_config)   # LoRA adapters attach AFTER wrap
```

Rationale: LoRA params attached after wrapping are NOT sharded - they are
replicated full-size tensors on every core. PEFT's forward delta
(`lora_B @ lora_A @ x`) then computes identically on each core, and FSDP
handles the sharded base matmul. Crucially, `apply_abliteration` writes
`lora_A/lora_B .data` as full matrices - this only works if LoRA params are
unsharded, which this ordering guarantees.

### Changes to `model.py`

1. In `Model.__init__`, when `self._is_tpu and settings.tpu_use_fsdp and
   settings.tpu_cores > 1`:
   - Load with `device_map=None` (CPU), then `wrap_model_fsdp(...)` using
     `settings.tpu_fsdp_config` or `get_default_fsdp_config(model)`.
   - Then call `_apply_lora()` (existing call site stays).
2. `get_layer_modules()` / `get_layers()`: must traverse through the FSDP
   wrapper (`xla_fsdp.FSDP` exposes `.named_modules()`), so LoRA target
   collection in `_apply_lora` keeps working - verify by printing the layer
   count (existing line: `* Transformer model with N layers`).
3. `get_merged_model()`: on TPU, route to the CPU-reload branch (the current
   quantized-model path already does exactly this: clone adapter state to CPU,
   load base model on CPU, reapply, `merge_and_unload`). This avoids
   `merge_and_unload` on sharded XLA params entirely. Change the condition
   from `quantization == BNB_4BIT` to
   `quantization == BNB_4BIT or self._is_tpu`.
4. `save_merged_model()`: dead code - either delete it or make main.py call
   it for TPU. Prefer: keep main.py as-is (CPU-merged model -> plain
   `save_pretrained` works).

### FSDP smoke test

1. 0.5B on 8 cores: wrap succeeds, forward + get_responses work, LoRA apply
   (1 trial) works, reset_model works.
2. `DeepSeek-R1-Distill-Qwen-7B` single core (baseline) vs 8-core FSDP.
3. `DeepSeek-R1-Distill-Qwen-14B` on 8-core FSDP - the first real target.
4. Memory check per phase: HBM in use / 16GB per core.

### Known risks

- XLA FSDP + PEFT interplay is the least-tested area; expect iteration.
- If `.data` writes on LoRA params fail under FSDP (size mismatch due to
  unexpected sharding), fallback: set `min_num_params` high enough that LoRA
  modules stay out of shard policy (already 100M) - verify LoRA ranks are
  never wrapped.
- If forward hidden-state gathering is wrong under FSDP, residuals will be
  garbage - smoke test asserts shapes, also check values are finite.

## Phase 4 - Full 200-trial run, single-core (0.5B) (~1-2 h)

Ground-truth E2E before big models:

```
heretic --model=Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --dtypes=bfloat16 --quantization=none --device-map=auto \
  --batch-size=2 --n-trials=200 --n-startup-trials=60 \
  --max-response-length=100 --row-normalization=full \
  --full-normalization-lora-rank=3 --orthogonalize-direction \
  --winsorization-quantile=1.0 --study-checkpoint-dir=/root/checkpoints \
  --model-action=save --export-strategy=merge \
  --save-directory=/root/model \
  --good-prompts='{"dataset":"mlabonne/harmless_alpaca","split":"train[:100]","column":"text"}' \
  --bad-prompts='{"dataset":"mlabonne/harmful_behaviors","split":"train[:100]","column":"text"}'
```

Watch items:
- HBM/RAM growth across trials (XLA cache) - `empty_cache` should keep it flat.
- KL + KeywordRate scores actually move (abliteration working).
- Export: merged model saved, reload it in `tpu/smoke_test.py` style and
  confirm refusals dropped on the harmful prompts.
- Study checkpoint resume: kill mid-run, rerun same command, confirm resume.

## Phase 5 - 7B/14B full runs (multi-hour)

1. `DeepSeek-R1-Distill-Qwen-7B` - single core first (baseline config from
   `config.tpu.toml`), then FSDP 8-core to compare.
2. `DeepSeek-R1-Distill-Qwen-14B` with FSDP:
   `--tpu-cores=8 --tpu-use-fsdp=true --tpu-fsdp-config=configs/fsdp_tpu_v5e_8.json`
   (n_trials=100, batch_size=1 per config.tpu.toml presets).
3. Export + verify same as Phase 4.

## Don't-do list (traps)

- Never `pip install torch` from PyPI on the VM (breaks XLA pairing).
- Don't run `lm_eval` benchmarks on TPU (CPU-fallback slow, not the point).
- Don't use `--offload-outputs-to-cpu` on TPU (code already skips it).
- Don't use quantization/bitsandbytes - TPU path forces it off anyway.
- Don't run interactive mode over SSH without a tty - pass
  `--model-action` / `--export-strategy` explicitly.

## Definition of done

- [ ] `tpu/smoke_test.py` all PASS on 1 core of the v5e-8
- [ ] FSDP: 14B model loads, wraps, runs 200 trials on 8 cores
- [ ] Full run exports a merged model that reloads and shows reduced refusals
- [ ] Study checkpoint resume works (kill + rerun)
- [ ] `git status` clean, all TPU changes committed, Kaggle/Colab notebooks
      updated to reflect the final working flow
