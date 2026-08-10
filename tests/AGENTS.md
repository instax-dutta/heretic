# tests/AGENTS.md

## Purpose

- Reproducibility test suite: verifies that model/config logic changes (row normalization, LoRA rank, winsorization, etc.) do not change abliteration output for tiny-random models, unless intentionally

## Ownership

- `run_tests.py` - main reproducibility runner: clones a tiny-random HF model, hashes weights + outputs before/after, compares SHA256SUMS
- `test_config.py` - pytest suite for config/env behavior
- `gemma-4e/`, `minicpm5/`, `mistral-3/`, `qwen2.5/`, `qwen3.5-moe/` - tiny-random model fixtures
- `README.md` - usage guide

## Local Contracts

- Use tiny-random HF models for testing (prefer models WITHOUT `special_tokens_map.json`, they are usually wrong)
- Hashing via `sha256sum -b` compared against `SHA256SUMS.LABEL`
- Tests run locally (dev/test only); heavy runs go on GCP VMs / Kaggle TPU (see root AGENTS.md)

## Work Guidance

- Before any change to `src/heretic/model.py` or config that can affect reproducibility, run these tests
- Add new model fixtures as test needs grow; document the model used in README

## Verification

- `python -m pytest tests/ -x` for config/env; `python tests/run_tests.py` for reproducibility

## Child DOX Index

- No child AGENTS.md files