# Phase 0: Infrastructure Validation & Baselines

## Overview
Phase 0 validates the infrastructure, establishes baselines, and extracts base model representations before any fine-tuning.

## Purpose
1. **Sanity Checks**: Ensure all tasks run without errors
2. **Baseline Metrics**: Establish naive baselines (majority class, random)
3. **Base Representations**: Extract pre-training representations for drift analysis
4. **Infrastructure Test**: Validate memory usage and model architecture

## Prerequisites
- Datasets downloaded in `data/` (run `python scripts/download_datasets.py`)
- Config file: `shared/config.yaml`
- Conda environment activated with all dependencies

## Execution

```bash
cd scripts/phase0
bash vm1.sh  # Runs sanity checks for all classification tasks
```

## What This Phase Does

### 1. Sanity Checks
- Validates that each task can overfit on 10 samples
- Tests LoRA and Full FT implementations
- Confirms no gradient explosion or training failures

### 2. Baseline Metrics
- Majority class predictions (for classification tasks)
- Random predictions
- Zero-shot TinyLlama performance

### 3. Base Representations Extraction
**Important**: Run this **before** Phase 2 training:

```bash
# Extract base model representations (~2 hours, ~48GB)
python scripts/extract_base_representations.py
```

This creates:
```
base_representations/
├── mrpc_base_representations/
├── sst2_base_representations/
└── rte_base_representations/
```

## Outputs

- **Logs**: `logs/phase0/`
- **Base representations**: `base_representations/` (used in Phase 4 for drift analysis)

## Next Steps
- Proceed to Phase 1 (Hyperparameter Optimization)
