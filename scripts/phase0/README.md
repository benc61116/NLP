# Phase 0: Infrastructure Validation & Baselines

## Overview
Phase 0 validates the infrastructure, establishes baselines, and extracts base model representations before any fine-tuning.

## Purpose
1. **Sanity Checks**: Ensure all tasks run without errors
2. **Baseline Metrics**: Establish naive baselines (majority class, random)
3. **Base Representations**: Extract pre-training representations for drift analysis
4. **Infrastructure Test**: Validate memory usage and model architecture

## Prerequisites
- Dataset downloaded and processed in `data/`
- Config file: `shared/config.yaml`
- Conda environment activated with all dependencies

## VM Distribution

### VM1 (SQuAD v2)
- **Tasks**: squad_v2
- **Components**:
  - Majority/random baselines
  - Infrastructure validation (architecture test)
- **Estimated runtime**: ~4 hours

### VM2 (Classification)
- **Tasks**: mrpc, sst2, rte
- **Components**:
  - Majority/random baselines for all tasks
  - Base model representation extraction
  - Memory profiling
- **Estimated runtime**: ~5 hours

## Execution

### On VM1:
```bash
cd /path/to/workspace/NLP
bash scripts/phase0/vm1.sh
```

### On VM2:
```bash
cd /path/to/workspace/NLP
bash scripts/phase0/vm2.sh
```

## Key Features

1. **No Overlap**: VM1 and VM2 handle different tasks - can run in parallel
2. **Baseline Establishment**: Provides comparison points for fine-tuned models
3. **Representation Extraction**: Saves base model representations for Phase 3 drift analysis
4. **Sanity Validation**: Confirms all tasks load and run properly
5. **W&B Logging**: All experiments logged to `NLP-Phase0` project

## What Gets Created

### Baselines
- Majority class predictions (for classification tasks)
- Random predictions (for both QA and classification)
- Logged to W&B for comparison

### Representations
- Base model representations extracted from all tasks
- Saved for drift analysis in Phase 3
- Format: `.npy` files in `representations/base/`

### Logs
- Detailed logs in `logs/phase0/vm1/` and `logs/phase0/vm2/`
- Infrastructure validation reports
- Memory usage profiles

## Success Criteria
- ✅ All tasks run without errors
- ✅ Baseline metrics established (majority class accuracy, random F1)
- ✅ Base representations extracted successfully
- ✅ No memory issues or OOM errors
- ✅ Architecture properly configured for each task type

## Next Steps
After Phase 0 completion:
- Verify all baselines are in W&B
- Check that base representations are saved
- Proceed to Phase 1 (Hyperparameter Optimization)

## Outputs
- **Baselines**: Logged to W&B `NLP-Phase0` project
- **Representations**: `representations/base/*.npy`
- **Logs**: `logs/phase0/vm1/` and `logs/phase0/vm2/`
- **W&B**: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase0
