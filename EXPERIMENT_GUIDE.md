# NLP Research Project: LoRA vs Full Fine-tuning Comparison

## Overview

This repository contains a rigorous experimental setup for comparing LoRA (Low-Rank Adaptation) and full fine-tuning approaches on the Llama-2-1.3B model across four diverse NLP tasks.

## Experiment Design

### Model & Tasks
- **Model**: `meta-llama/Llama-2-1.3b-hf` 
- **Tasks**: 
  - MRPC (Paraphrase Detection)
  - SST-2 (Sentiment Analysis) 
  - RTE (Textual Entailment)
  - SQuAD v2.0 (Question Answering)

### Methods Compared
1. **LoRA Fine-tuning**: Low-rank adaptation with r=16, α=32
2. **Full Fine-tuning**: Traditional parameter updates

### Infrastructure
- 3 GPU VMs with task-based parallel allocation
- Weights & Biases tracking (project: "NLP", entity: "galavny-tel-aviv-university")

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure W&B
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university
wandb login
```

### 2. Download Data

```bash
python scripts/download_datasets.py
```

### 3. Run Validation Demo

```bash
python run_validation_demo.py
```

### 4. Execute Experiments

#### Option A: Run All Phases (Recommended)
```bash
# Phase 1: Sanity checks + initial experiments
bash scripts/phase1/vm1.sh  # Sanity checks + MRPC + SST-2
bash scripts/phase1/vm2.sh  # RTE + SQuAD v2
bash scripts/phase1/vm3.sh  # Validation

# Phase 2a: Method comparison
bash scripts/phase2a/vm1.sh  # LoRA for all tasks
bash scripts/phase2a/vm2.sh  # Full FT for all tasks  
bash scripts/phase2a/vm3.sh  # Monitoring

# Phase 2b: Analysis
bash scripts/phase2b/vm1.sh  # Final analysis
```

#### Option B: Run Individual Experiments
```bash
# Single task, single method
python -m shared.experiment_runner --tasks sst2 --methods lora

# Multiple tasks, single method
python -m shared.experiment_runner --tasks mrpc sst2 rte --methods lora

# All tasks, both methods
python -m shared.experiment_runner --tasks mrpc sst2 rte squad_v2 --methods lora full
```

## Repository Structure

```
NLP/
├── requirements.txt          # Python dependencies (pinned versions)
├── shared/                   # Shared utilities
│   ├── config.yaml          # Experiment configuration
│   ├── data_preparation.py  # Data loading utilities
│   ├── sanity_checks.py     # Validation and sanity checks
│   └── experiment_runner.py # Main experiment framework
├── scripts/                 # Execution scripts
│   ├── download_datasets.py # Dataset download
│   ├── phase1/              # Sanity checks + initial experiments
│   ├── phase2a/             # Method comparison experiments
│   └── phase2b/             # Final analysis
├── data/                    # Downloaded datasets
├── results/                 # Experiment outputs
├── logs/                    # Execution logs
└── run_validation_demo.py   # Validation script
```

## Configuration

Key configuration parameters in `shared/config.yaml`:

### Model Settings
```yaml
model:
  name: "meta-llama/Llama-2-1.3b-hf"
  max_length: 512
  torch_dtype: "float16"
```

### LoRA Settings
```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Settings
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4  # LoRA
  full_finetune_learning_rate: 5e-5  # Full FT
  evaluation_strategy: "steps"
  eval_steps: 100
```

## Sanity Checks

The framework includes comprehensive sanity checks:

1. **Model Loading**: Verify model loads correctly
2. **LoRA Setup**: Check LoRA configuration and parameter counts
3. **Data Loading**: Validate all datasets load properly
4. **Gradient Flow**: Test gradients flow correctly for both methods
5. **Overfitting**: Verify models can overfit small datasets
6. **W&B Logging**: Test Weights & Biases integration
7. **Reproducibility**: Check deterministic behavior with fixed seeds

Run sanity checks:
```bash
python -m shared.sanity_checks
```

## Reproducibility

### Fixed Seeds
- Global seed: 42
- Deterministic mode enabled
- Same data sampling across runs

### Package Versions
All dependencies pinned in `requirements.txt`:
- torch==2.1.0
- transformers==4.35.0
- peft==0.6.0
- datasets==2.14.0
- wandb==0.15.12

### Reproduction Steps
1. Use exact package versions: `pip install -r requirements.txt`
2. Download datasets: `python scripts/download_datasets.py`
3. Run experiments with provided scripts
4. Compare results with fixed seed (42)

## Monitoring & Results

### Weights & Biases Dashboard
- Project: https://wandb.ai/galavny-tel-aviv-university/NLP
- All experiments automatically logged
- Real-time monitoring of training progress
- Hyperparameter tracking and comparison

### Local Results
- `results/`: Model checkpoints and predictions
- `logs/`: Detailed execution logs
- CSV files with performance metrics
- Visualization plots comparing methods

## VM Allocation Strategy

### VM1 (Classification Focus)
- **Phase 1**: Sanity checks + MRPC + SST-2
- **Phase 2a**: LoRA experiments for all tasks
- **Phase 2b**: Final analysis and synthesis

### VM2 (Entailment & QA Focus)  
- **Phase 1**: RTE + SQuAD v2
- **Phase 2a**: Full fine-tuning for all tasks
- **Phase 2b**: Not used

### VM3 (Validation & Monitoring)
- **Phase 1**: Data validation + quick experiments
- **Phase 2a**: Progress monitoring + preliminary analysis
- **Phase 2b**: Not used

## Expected Results

### Performance Metrics
- **Classification tasks**: Accuracy, F1-score
- **QA task**: F1-score, Exact Match
- **All tasks**: Training loss, Evaluation loss, Runtime

### Key Comparisons
1. **Performance**: LoRA vs Full FT accuracy/F1
2. **Efficiency**: Training time and memory usage
3. **Stability**: Loss curves and convergence
4. **Generalization**: Evaluation metrics across tasks

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce batch size in `shared/config.yaml`
   - Enable gradient checkpointing
   - Use smaller max_length

2. **W&B Authentication**
   ```bash
   wandb login
   export WANDB_PROJECT=NLP
   export WANDB_ENTITY=galavny-tel-aviv-university
   ```

3. **Data Loading Errors**
   ```bash
   python scripts/download_datasets.py
   python -c "from shared.data_preparation import TaskDataLoader; TaskDataLoader().validate_data_integrity()"
   ```

4. **Reproducibility Issues**
   - Ensure deterministic=true in config
   - Check that all seeds are set consistently
   - Verify no randomness in data loading

### Debug Commands

```bash
# Test data loading
python -c "from shared.data_preparation import TaskDataLoader; TaskDataLoader().print_dataset_summary()"

# Test model loading
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-1.3b-hf')"

# Test LoRA setup
python -c "from shared.sanity_checks import ModelSanityChecker; ModelSanityChecker().check_lora_setup()"

# Check W&B connection
python -c "import wandb; wandb.init(project='NLP', entity='galavny-tel-aviv-university'); wandb.finish()"
```

## Citation

If you use this experimental framework, please cite:

```bibtex
@misc{nlp_lora_comparison_2024,
  title={LoRA vs Full Fine-tuning: A Rigorous Comparison on Llama-2-1.3B},
  author={Your Name},
  year={2024},
  institution={Tel Aviv University}
}
```

## Contact

For questions or issues, please contact the research team or create an issue in this repository.
