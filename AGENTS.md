# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Purpose

- heretic: fully automatic censorship removal for language models (abliteration via activation-magnitude direction removal)
- This repo is being TPU-ported and driven to produce the world's first TPU-abliterated model: Qwen2.5-VL-3B-Instruct (multimodal), targeting a public Hugging Face upload
- TPU port adds: auto-detection (PJRT_DEVICE=TPU / XLA_USE_BF16 / XLA_USE_SPMD), FSDP sharding, per-core batch warmup, TPU-safe checkpoint resumption
- Personal preferences: no emojis; dashes not em dashes; AI dev speed beats development-cost worries; reproduce bugs end-to-end before fixing; never run long batch jobs locally - use GCP VMs / Kaggle TPU

## Ownership

- Root-owned files: `README.md`, `LICENSE`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `.gitattributes`, `.github/`, `.gemini/`, `config.default.toml`, `config.nohumor.toml`, `config.noslop.toml`, `TPU_PLAN.md`, `exported_model/` (output artifact), and root-level project documentation
- `configs/`, `notebooks/`, `src/`, `tests/`, `tpu/` have their own child AGENTS.md files

## Local Contracts

- Python 3.12, uv-based build (`uv_build` backend), package name `heretic-llm`, CLI entry point `heretic` (see pyproject.toml)
- Bayraktar-scale model support via transformers 4.5x; dtype bfloat16, no quantization on TPU runs
- Primary execution environment for heavy runs: Kaggle TPU v5e-8 VMs (12h sessions) reached over a zrok SSH tunnel alias `kaggle`; local machine is dev/test only
- All TPU smoke/validation/run sequence is scripted under `tpu/`; see `tpu/AGENTS.md`

## Work Guidance

- Before any model/config change that can affect reproducibility, run the tests under `tests/` (see `tests/AGENTS.md`)
- TPU runs: always use the auto-detection path (no `--tpu-cores`/`--tpu-use-fsdp` flags); keep flags in `tpu/run_*.sh` as the sole source of truth
- Long-running VM sessions: keep the Optuna journal checkpoint synced off-VM (`tpu/resume/`) so a VM death never loses the study

## Verification

- `python -m pytest tests/ -x` covers config and env behavior; `tests/run_tests.py` covers reproducibility for tiny-random models

## Child DOX Index

- `configs/AGENTS.md` - TPU/FSDP config files (torch-xla runtime settings, FSDP sharding specs)
- `notebooks/AGENTS.md` - Colab/Kaggle TPU notebooks (env check, debug, run)
- `src/heretic/AGENTS.md` - the heretic package: abliteration core, TPU auto-detection, FSDP, scorers
- `tests/AGENTS.md` - reproducibility test suite for tiny-random models
- `tpu/AGENTS.md` - TPU port ops: run scripts, smoke test, handoff, study checkpoints, VM sync