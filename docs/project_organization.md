# Project Organization

This repository is a research codebase for multi-species disease image classification. Keep the repository focused on source code, configuration, tests, and lightweight documentation. Datasets, checkpoints, generated reports, and local experiment outputs should stay out of git unless there is a specific reason to archive a small artifact.

## Canonical Layout

| Path | Purpose | Git policy |
| --- | --- | --- |
| `src/` | Shared Python package code: config, datasets, models, training helpers. | Track |
| `api/` | FastAPI inference application. | Track |
| `scripts/collection/` | Dataset import, cleanup, and download helpers. | Track code only |
| `scripts/training/` | Legacy and experiment training scripts. | Track code only |
| `scripts/evaluation/` | Evaluation and benchmarking scripts. | Track code only |
| `scripts/labeling/` | Labeling utilities. | Track code only |
| `config/` | YAML training configuration. | Track |
| `tests/` | Unit and smoke tests. | Track |
| `docs/` | Project notes, reports, and organization docs. | Track lightweight docs |
| `imports/` | Small imported archives or metadata that must be preserved. | Track only small, intentional files |
| `models/` | Trained checkpoints. | Ignore |
| `reports/`, `logs/`, `runs/`, `output/` | Generated outputs. | Ignore |
| `*_dataset*/`, `data/`, `dataset/`, `downloaded_disease_images/` | Local datasets and image caches. | Ignore |

## Root Directory Rules

Keep only these kinds of files at the repository root:

- Project entry points: `train_model.py`, `evaluate_model.py`, `predict_single.py`.
- Project metadata: `README.md`, `requirements.txt`, `.gitignore`, `.gitattributes`, `pytest.ini`.
- Small top-level scripts only when they are the primary public command for a workflow.

Move new helper scripts into the closest script group:

- Dataset work: `scripts/collection/`
- Training experiments: `scripts/training/`
- Model testing and reports: `scripts/evaluation/`
- Document/report one-off helpers: keep outside git or move to a dedicated tracked script only when reused.

## Local Artifact Rules

Do not commit:

- Model weights: `*.pth`, `*.pt`, `*.safetensors`, `*.bin`, `models/`.
- Dataset folders and dataset zip files.
- Generated reports, logs, confusion matrices, prediction CSV files, and training result folders.
- Local tool state such as `.gitnexus/`, `.pytest_cache/`, and `__pycache__/`.

When a generated result must be preserved, summarize it in `docs/` and keep the bulky generated files ignored.

## Current Cleanup Notes

- Independent model testing is now reproducible with `scripts/evaluation/evaluate_independent_models.py`.
- The independent test summary is documented in `docs/independent_model_test_summary.md`.
- The repository still has historical tracked paths that look like nested copies of the same project. Do not remove or restore those casually; handle them in a dedicated cleanup commit after reviewing git history.
