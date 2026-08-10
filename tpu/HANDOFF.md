# Handoff - TPU port status (speed focus, session 2)

Session date: 2026-08-05. Kaggle TPU v5e-8, 12h session. Prior session: 2026-08-04 (see below).
This session's wins: 20x faster generation compile, resume-flag fix, eval-prompt clamp fix,
1.5B single-core validated, and **SPMD/FSDP segfault root-caused and fixed** (the big one).

## THE SPMD FIX (this session's critical discovery)

**Problem**: Every multi-core FSDP run segfaulted in
`PjRtComputationClient::ExecuteReplicated()::{lambda()}::operator()` (@0x1f0 null deref).
Also reproducible with a 4-line sharded matmul (no FSDP, no transformers). Classic
multi-device `xm.all_reduce` worked; JAX multi-device worked; torch_xla 2.8.0/2.8.1/2.9.0
all crashed identically. The TPU and driver were fine.

**Root cause**: `xr.use_spmd()` after the XLA client had already initialized (any prior
`global_device_count()` / `xla_device()` call) forced the "Replicating tensors already
initialized on non-virtual XLA device" path, which produces a broken SPMD client that
segfaults at the first execution.

**Fix**: set `XLA_USE_SPMD=1` in the environment BEFORE the first client/device access
so the client initializes in SPMD mode from the start:
- `setup_tpu_environment(enable_spmd=...)` - now takes a flag; main.py + Model.__init__ pass
  `tpu_use_fsdp and tpu_cores > 1`
- `_ensure_spmd_if_multichip` - sets the env var before checking/using the runtime
- `tpu/smoke_test.py` stage 1 - sets it from `_get_tpu_core_count_from_env()` before
  `get_xla_device_count()` (which would otherwise init the client without SPMD)

**Second bug on the same path**: `xm.xla_device()` (or any device access) must run BEFORE
`use_spmd()`, or `torch_xla.device(n)` dies in `aten_xla_bridge.cpp:30`
(`devices_ordinals_` map missing TPU:0). In the app this is naturally satisfied
(device init in Model.__init__ precedes the FSDP wrap); the smoke test's stage-1 device
call also covers it.

**Third**: with SPMD active, model inputs must be XLA tensors (`"Input tensor is not an
XLA tensor"` otherwise) and FSDP requires `shard_output` (CausalLMOutputWithPast is not
auto-supported). The app already had both (`_tokenize_prompts().to(_model_device())`,
`_spmd_shard_output`).

**Result**: `smoke_test.py` 10/10 PASS on 8-core FSDP (was SIGSEGV). Repro scripts in
`tpu/repro/`: `spmd_matmul.py` (minimal crash repro), `fsdp_repro.py` (full FSDP wrap +
forward), `multidevice_ar.py` (classic mode works), `spmd_plain.py`.

## Environment gotchas (session 2)

- **torch 2.9.0 + torch_xla 2.9.0: `from_pretrained` crashes with std::bad_alloc** on this
  VM. Reverted to torch 2.8.0 + torch_xla 2.8.1 (`pip install torch==2.8.0 torch_xla[tpu]==2.8.1
  -f https://storage.googleapis.com/libtpu-releases/index.html`). 2.8.1 keeps SPMD working
  (verified). Do NOT "upgrade" to 2.9 on this VM.
- `tpu_process_addresses="local"` metric-server error at startup is benign (appears on
  working single-core runs too).
- Multi-destination `scp a b c host:x host:y` silently mis-copies - always one file per scp.

## Speed state

- **Per-trial ~38-42s on 0.5B and 1.5B single-core** (was ~39s before; the generation
  compile fix in fa92419 collapsed one-time fixed cost from 16min to 2.7min).
- 1.5B (Qwen2.5-Coder-1.5B-Instruct) validated end-to-end: 3 trials, ~41s/trial,
  best keywords 0-5/100 (vs 15-30/100 on 0.5B) - bigger models abliterate much better.
  Script: `tpu/run_1p5b.sh`.
- 10-trial 0.5B E2E: 10.3 min incl. compile+baseline+export.
- 7B 8-core FSDP run: launched 21:38 UTC session-2 (script `tpu/run_7b.sh`) - status TBD.

## Baseline facts (from prior session, still valid)

- XLA execs cost HBM, never reclaimed (hard crash ~16.9GB); batched methods run all
  prompts in ONE exec; padding="max_length" (128) mandatory; mark_step(wait=True);
  torch.cat on XLA crashes (materialize to CPU); merge/export on CPU.
- Interactive menus need CLI flags on non-tty (`--trial-index --export-strategy
  --checkpoint-action --model-action --save-directory`); resume re-applies 11 flags from
  CLI (main.py ~line 407): trial_index/export_strategy/checkpoint_action/model_action/
  n_trials/n_startup_trials/seed/batch_size/max_response_length/good_prompts/bad_prompts.

## Next session priorities

1. Check the 7B FSDP run (validating multi-core big-model path - the whole point of SPMD).
2. If 7B works: HF token (if needed) for VL models, longer trials runs, real benchmark.
3. Update `TPU_PLAN.md` per-trial exec budget with SPMD numbers.
4. Validate exported models locally (chat + refusal checks).

---

# Session 3 (2026-08-10) - World's-first TPU-abliterated run: COMPLETE

## Mission outcome

- **Qwen2.5-VL-3B-Instruct 200-trial abliteration completed on TPU** with the
  auto-detection path (no --tpu-cores/--tpu-use-fsdp flags). ~1.44 min/trial
  (4h13m for 176 trials resumed). **Pace is comparable to a single GPU, NOT faster**:
  trials are sequential (TPE), generation is batch-2/tiny-XLA-graph bound (HBM
  16.9GB/core), eval clamped to 20 prompts. The TPU win is resilience (SPMD/FSDP
  + journal resume), not speed.
- **Best trial 166: 2/20 keyword refusals (baseline 7/20), KL 0.0247.**
- 2 VM deaths survived with zero study loss thanks to the off-VM journal sync.
  Full study journal: `tpu/resume/Qwen--Qwen2--5-VL-3B-Instruct.jsonl` (md5
  274f14b42c68811747f6f8d9592d6504 = last VM copy, verified identical).

## CRITICAL BLOCKER - restore/export path produces NO-OP model

- During optimization, abliteration works (trial kw scores 7/20 -> 2/20).
- BUT "Restoring model from trial N" (post-study export, `main.py:1002`) then
  merge/save produces weights **byte-identical to base**:
  - `tpu/logs/eval100.log`: base 22/100, ablated 22/100
  - `tpu/logs/diff.log`: 10/10 responses identical (base vs ablated)
  - `tpu/logs/wc.log`: sampled tensors `same=True` vs HF base
- The restore uses the trial's stored abliteration params but the merge ends up a
  no-op. Suspects: restore path parameter application, LoRA init/merge ordering,
  or direction recomputation from 2-prompt residual means (run used good/bad
  `train[:2]` = 2 prompts each for residuals). Fix BEFORE any upload.
- Repro: take saved study, run `tpu/run_vl3b.sh` on a fresh VM (it went straight
  to "Resuming existing study -> Optimization finished -> Restoring model from
  trial 167 [index of trial_id 166] -> Model saved to exported_model").

## Session-3 gotchas / fixes

- **eval100 OOM**: full 100-prompt generation in one XLA graph exhausts HBM.
  Fix: chunk at 20 prompts + `Settings(max_response_length=20, batch_size=2)` in
  the eval script (run config used `--max-response-length=20`, default is 100!
  The 100-token unrolled graph is ~5x bigger).
- **Study resume validation crash**: `Settings.model_validate_json` of stored
  snapshot fails: `n_additional_trials: 0` (headless default) violates
  PositiveInt. Fix: patch the jsonl user_attr settings to
  `"n_additional_trials": <remaining>` (e.g. 176) before resuming with
  `--checkpoint-action=continue`. Ran 24->200 after the patch.
- **VM provisioning churn**: repo is private (clone needs auth) - ship source via
  `tar --exclude=.git | ssh kaggle 'tar -xzf -'`. Env setup: `/root/setup_vm.sh`
  (torch 2.8.0+cu128, torch_xla 2.8.0/2.8.1+cu128, `pip install -e /root/heretic[dev]`).
- Raw (non-synced) study data is LOST with the VM - always keep `tpu/resume/`
  synced every few minutes during runs (crash insurance proved essential).

## Files added this session

- `tpu/run_vl3b.sh` - Qwen2.5-VL-3B 200-trial recipe (auto-detection only)
- `tpu/eval100.py` - one-shot x/100 refusal eval (base vs ablated)
- `tpu/resume/` - 200-trial Optuna journal (the durable study)
- `tpu/logs/` - run/eval/diff/weight-check logs from session 3
- `tpu/run_qwen35_4b.sh` - aborted 4B attempt (too slow, new arch)
- DOX tree: root + `configs/ notebooks/ src/heretic/ tests/ tpu/` AGENTS.md

## Next session priorities (quota resets Saturday)

1. Fix the restore/export no-op bug (BLOCKER - see above).
2. Get a real x/100 ablated score (eval100 with fixed export).
3. Private HF upload: Qwen2.5-VL-3B-Instruct-TPU-Abliterated-heretic.
4. Optionally resume study for more trials if export fix changes scoring.
