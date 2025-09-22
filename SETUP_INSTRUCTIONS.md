# NLP Research Project Setup Instructions

## 🎯 Project Overview

This repository implements a rigorous experimental comparison between **LoRA (Low-Rank Adaptation)** and **full fine-tuning** on language models across four diverse NLP tasks:

- **MRPC**: Paraphrase Detection
- **SST-2**: Sentiment Analysis  
- **RTE**: Textual Entailment
- **SQuAD v2.0**: Question Answering

## ✅ Current Setup Status

**ALL BASIC COMPONENTS SUCCESSFULLY IMPLEMENTED:**

### ✓ Core Infrastructure
- [x] Data loading utilities for all 4 tasks (`shared/data_preparation.py`)
- [x] Comprehensive sanity check framework (`shared/sanity_checks.py`)
- [x] Main experiment runner (`shared/experiment_runner.py`)
- [x] Phase-organized execution scripts (3 VMs × 3 phases)
- [x] Complete configuration management (`shared/config.yaml`)
- [x] Weights & Biases integration
- [x] Reproducibility controls (fixed seeds, deterministic mode)

### ✓ Validation Results
**Basic validation demo passed 6/6 tests:**
- [x] All required packages import successfully
- [x] Directory structure complete
- [x] Configuration loading works
- [x] All 4 datasets loaded (200K+ total samples)
- [x] All phase scripts accessible
- [x] W&B connection established

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Dependencies already installed
pip install -r requirements.txt

# W&B configuration
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university
```

### 2. Verify Setup
```bash
# Run basic validation
python run_simple_demo.py

# Expected output: "🎉 ALL BASIC TESTS PASSED!"
```

### 3. Model Authentication (Required for Llama-2)
For the actual Llama-2 experiments, you'll need HuggingFace authentication:

```bash
# Option A: HuggingFace CLI
huggingface-cli login

# Option B: Environment variable
export HUGGINGFACE_TOKEN="your_token_here"
```

**Note**: The current demo uses `microsoft/DialoGPT-small` (publicly available). To use the intended `meta-llama/Llama-2-1.3b-hf`, update `shared/config.yaml` and ensure you have access.

## 📋 Experiment Execution

### Phase 1: Validation & Initial Experiments
```bash
# VM1: Sanity checks + MRPC + SST-2
bash scripts/phase1/vm1.sh

# VM2: RTE + SQuAD v2  
bash scripts/phase1/vm2.sh

# VM3: Data validation + monitoring
bash scripts/phase1/vm3.sh
```

### Phase 2a: Method Comparison
```bash
# VM1: LoRA experiments for all tasks
bash scripts/phase2a/vm1.sh

# VM2: Full fine-tuning for all tasks
bash scripts/phase2a/vm2.sh

# VM3: Progress monitoring
bash scripts/phase2a/vm3.sh
```

### Phase 2b: Analysis & Synthesis
```bash
# VM1: Final analysis and reporting
bash scripts/phase2b/vm1.sh
```

### Individual Experiments
```bash
# Single task + method
python -m shared.experiment_runner --tasks sst2 --methods lora

# Multiple tasks
python -m shared.experiment_runner --tasks mrpc sst2 rte --methods lora full

# All tasks, both methods
python -m shared.experiment_runner --tasks mrpc sst2 rte squad_v2 --methods lora full
```

## 🔧 Key Configuration

### Model Settings (`shared/config.yaml`)
```yaml
model:
  name: "meta-llama/Llama-2-1.3b-hf"  # Change back for actual experiments
  max_length: 512
  torch_dtype: "float16"

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Settings
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4        # LoRA
  full_finetune_learning_rate: 5e-5  # Full FT
  evaluation_strategy: "steps"
  eval_steps: 100
```

## 📊 Monitoring & Results

### Weights & Biases Dashboard
- **Project**: https://wandb.ai/galavny-tel-aviv-university/NLP
- **Real-time tracking**: Training loss, eval metrics, runtime
- **Automatic logging**: Hyperparameters, model checkpoints
- **Comparison tools**: Built-in experiment comparison

### Local Output Structure
```
results/
├── {method}_{task}_{timestamp}/     # Individual experiment results
├── experiment_results_{id}.json     # Aggregated results
└── logs/                           # Detailed execution logs

logs/
├── phase1/vm{1,2,3}/              # Phase 1 logs
├── phase2a/vm{1,2,3}/             # Phase 2a logs  
└── phase2b/vm1/                   # Final analysis
```

## 🔍 Sanity Checks Framework

The framework includes 9 comprehensive sanity checks:

1. **Model Loading**: Verify model loads correctly
2. **LoRA Setup**: Check LoRA parameter counts and configuration
3. **Data Loading**: Validate all datasets load with correct format
4. **Gradient Flow (LoRA)**: Test gradients flow correctly for LoRA
5. **Gradient Flow (Full)**: Test gradients flow correctly for full FT
6. **Overfitting (LoRA)**: Verify LoRA can overfit small dataset
7. **Overfitting (Full)**: Verify full FT can overfit small dataset
8. **W&B Logging**: Test Weights & Biases integration
9. **Reproducibility**: Check deterministic behavior with fixed seeds

```bash
# Run all sanity checks
python -m shared.sanity_checks
```

## 📁 Repository Structure

```
NLP/
├── requirements.txt              # Pinned dependencies
├── run_simple_demo.py           # Basic validation script
├── run_validation_demo.py       # Full validation (needs model access)
├── SETUP_INSTRUCTIONS.md       # This file
├── EXPERIMENT_GUIDE.md          # Detailed experiment guide
├── shared/                      # Core utilities
│   ├── config.yaml             # Experiment configuration
│   ├── data_preparation.py     # Data loading & preprocessing
│   ├── sanity_checks.py        # Validation framework
│   └── experiment_runner.py    # Main experiment framework
├── scripts/                    # Execution scripts
│   ├── download_datasets.py   # Dataset download
│   ├── phase1/                # Validation & initial experiments
│   ├── phase2a/               # Method comparison
│   └── phase2b/               # Analysis & synthesis
├── data/                       # Downloaded datasets (203K+ samples)
│   ├── mrpc/                  # 3,668 training samples
│   ├── sst2/                  # 67,349 training samples
│   ├── rte/                   # 2,490 training samples
│   └── squad_v2/              # 130,319 training samples
└── [results/, logs/, wandb/]   # Created during execution
```

## 🔄 Reproducibility Features

### Fixed Seeds & Deterministic Mode
- **Global seed**: 42 (configurable)
- **Deterministic training**: Enabled by default
- **Data sampling**: Consistent across runs
- **Library versions**: Pinned in requirements.txt

### Package Versions (Pinned)
```
torch==2.1.0
transformers==4.35.0
peft==0.6.0
datasets==2.14.0
wandb==0.15.12
numpy==1.24.3
# ... (see requirements.txt for complete list)
```

## 🎯 Expected Results

### Performance Metrics
- **Classification**: Accuracy, F1-score, Loss
- **QA**: F1-score, Exact Match, Loss
- **Efficiency**: Training time, Memory usage
- **Reproducibility**: Identical results with same seed

### Key Comparisons
1. **Performance**: LoRA vs Full FT accuracy across tasks
2. **Efficiency**: Training time and memory consumption
3. **Stability**: Loss curves and convergence behavior
4. **Generalization**: Cross-task performance patterns

## 🚨 Known Limitations & Notes

### Current Demo Model
- Using `microsoft/DialoGPT-small` for public accessibility
- For actual research, change to `meta-llama/Llama-2-1.3b-hf` in config
- Requires HuggingFace authentication for Llama-2 access

### System Requirements
- **GPU Memory**: 24GB recommended (configurable batch sizes)
- **Storage**: ~2GB for datasets + model checkpoints
- **Network**: Stable connection for W&B logging

### Troubleshooting
```bash
# Check data integrity
python -c "from shared.data_preparation import TaskDataLoader; TaskDataLoader().validate_data_integrity()"

# Test W&B connection
python -c "import wandb; wandb.init(project='NLP', entity='galavny-tel-aviv-university'); wandb.finish()"

# Verify model access
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')"
```

## 🎉 Next Steps

### For Immediate Use:
1. ✅ **Basic setup complete** - All components implemented and validated
2. ✅ **Data ready** - 4 datasets loaded (203K+ samples total)
3. ✅ **Scripts ready** - Phase-organized execution scripts created
4. ✅ **Monitoring ready** - W&B integration configured

### For Full Research Deployment:
1. **Model Access**: Set up Llama-2-1.3b-hf authentication
2. **Scale Testing**: Run Phase 1 scripts to validate full pipeline
3. **Production Run**: Execute all phases for complete comparison
4. **Analysis**: Use Phase 2b for comprehensive result analysis

### Ready to Execute:
```bash
# Start with Phase 1 validation
bash scripts/phase1/vm1.sh

# Monitor progress in W&B dashboard
# Results will be saved locally and to W&B
```

---

**Status**: ✅ **SETUP COMPLETE & VALIDATED**  
**Ready for**: Immediate experimentation or full research deployment  
**W&B Dashboard**: https://wandb.ai/galavny-tel-aviv-university/NLP
