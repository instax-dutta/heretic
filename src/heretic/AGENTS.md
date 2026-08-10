# src/heretic/AGENTS.md

## Purpose

- The heretic package: fully automatic censorship removal for language models via abliteration (activation-magnitude direction removal), plus the TPU port: auto-detection, FSDP sharding, TPU-safe checkpointing

## Ownership

- `main.py` - CLI entry point, run orchestration, Optuna study loop, checkpoint resume logic
- `config.py` - Settings models (root config + plugin/scorer settings), noninteractive defaults
- `model.py` - Model wrapper: load, generate, FSDP wrap/unwrap, merged-model save
- `system.py`, `analyzer.py`, `evaluator.py`, `plugin.py`, `reproduce.py`, `utils.py`, `progress.py`, `scorer.py` - supporting machinery
- `fsdp_utils.py` - FSDP layer-class discovery and wrapping
- `scorers/` - optimization objectives: `keyword_rate.py` (refusal detection), `kl_divergence.py` (behavior preservation)
- `__init__.py` - package surface

## Local Contracts

- TPU auto-detection: `PJRT_DEVICE=TPU` (plus XLA_USE_BF16 / XLA_USE_SPMD env) drives setup; no `--tpu-cores` / `--tpu-use-fsdp` flags
- Critical sequencing: `XLA_USE_SPMD=1` must be set BEFORE any XLA client/device init, or multi-core runs segfault (see `tpu/HANDOFF.md`)
- `Model` loads the model in `__init__` (verified by eval scripts)
- Study checkpointing: Optuna JournalStorage jsonl under `checkpoints/`, safe for VM shutdowns via `--checkpoint-action=continue`
- Noninteractive defaults (merge export, continue checkpoint, save) keep VM runs headless-safe

## Work Guidance

- Keep TPU behavior behind auto-detection so local CPU runs are unaffected
- Any change to scoring (keyword_rate / KL divergence) or normalization must clear `tests/` reproducibility checks before landing
- Resuming a study: stored settings snapshot must have `n_additional_trials > 0` or resume validation fails; see `tpu/AGENTS.md`

## Verification

- `python -m pytest tests/ -x` for config/env behavior; `tests/run_tests.py` for tiny-random reproducibility (see `tests/AGENTS.md`)

## Child DOX Index

- No child AGENTS.md files