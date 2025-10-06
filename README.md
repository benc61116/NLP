# LoRA vs Full Fine-Tuning: Representational Drift and Deployment Efficiency

This repository contains a comprehensive empirical study comparing LoRA (Low-Rank Adaptation) and Full Fine-Tuning across 3 GLUE classification tasks (MRPC, SST-2, RTE) using TinyLlama-1.1B.

## Overview

**Research Questions:**
1. **RQ1: Representational Drift** - Does LoRA preserve base model representations better than full fine-tuning?
2. **RQ2: Deployment Efficiency** - What are the inference latency trade-offs between different LoRA deployment strategies?

**Key Findings:**
- Task-dependent representational preservation
- Merged LoRA adapters match full fine-tuning speed
- Multi-adapter deployment adds <1% overhead with identical predictions
- Training stability effects observed independently of drift reduction

## Project Structure

```
NLP/
├── ANALYSIS_REPORT.md                      # Comprehensive analysis report (main deliverable)
├── plan.md                                 # Detailed research plan and methodology
│
├── research_question_1_representational_drift.ipynb    # RQ1 analysis
├── research_question_2_deployment_efficiency.ipynb     # RQ2 analysis
│
├── deployment/                             # Phase 4B deployment benchmarks (tracked in git)
│   ├── deployment_benchmark_results.json   # Raw benchmark data
│   ├── deployment_benchmark_summary.csv    # Summary statistics
│   ├── multi_adapter_validation_results.json  # Correctness validation
│   └── *.png                               # Visualization plots
│
├── drift_analysis/                         # Phase 4A drift analysis results (tracked in git)
│   ├── drift_metrics.csv                   # Layer-wise drift metrics
│   ├── performance_metrics.csv             # Model performance metrics
│   └── *.json                              # Summary files
│
├── scripts/                                # Experiment execution scripts
│   ├── download_wandb_models.py            # Download trained models from WandB
│   ├── download_base_representations.sh    # Download base model representations
│   ├── phase0/                             # Environment setup
│   ├── phase1/                             # Hyperparameter optimization
│   ├── phase2/                             # Model training
│   ├── phase3/                             # Representation extraction
│   ├── phase4/                             # Drift analysis (Phase 4A)
│   └── phase4b/                            # Deployment benchmarks (Phase 4B)
│
├── shared/                                 # Shared utilities
│   └── data_preparation.py                 # Dataset loading and preprocessing
│
├── requirements.txt                        # Python dependencies
└── wandb_config.py                         # WandB configuration utilities
```

## What's Tracked in Git vs WandB

### **Tracked in Git (Available in Repository):**
- ✅ Analysis notebooks (`research_question_*.ipynb`)
- ✅ Final report (`ANALYSIS_REPORT.md`)
- ✅ Analysis results (`drift_analysis/`, `deployment/`)
- ✅ Scripts and utilities (`scripts/`, `shared/`)
- ✅ Configuration files (`requirements.txt`, `wandb_config.py`)

### **Ignored by Git (Too Large - Stored in WandB):**
- ❌ `results/` - Training checkpoints, model outputs
- ❌ `base_representations/` - Base model representations (~48GB)
- ❌ `data/` - Downloaded datasets
- ❌ `logs/` - Training logs
- ❌ `wandb/`, `wandb_cache/` - WandB local cache

### **WandB Projects & Artifacts:**

All trained models and large artifacts are stored in WandB:

| Project | Contents |
|---------|----------|
| `NLP-Phase1` | Hyperparameter optimization sweeps |
| `NLP-Phase2` | Full fine-tuned models (9 models: 3 tasks × 3 seeds) |
| `NLP-Phase2-LoRA-Rerun` | LoRA adapters (9 adapters: 3 tasks × 3 seeds) |
| `NLP-Phase3` | Extracted representations for drift analysis |

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd NLP

# Install dependencies
pip install -r requirements.txt

# Configure WandB
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university
wandb login
```

### 2. Download Trained Models (If Needed)

**Download all Phase 2 models:**
```bash
# Full fine-tuned models
python scripts/download_wandb_models.py \
    --entity galavny-tel-aviv-university \
    --project NLP-Phase2 \
    --all

# LoRA adapters
python scripts/download_wandb_models.py \
    --entity galavny-tel-aviv-university \
    --project NLP-Phase2-LoRA-Rerun \
    --all
```

**Download specific models:**
```bash
# Download single model
python scripts/download_wandb_models.py \
    --entity galavny-tel-aviv-university \
    --project NLP-Phase2 \
    --artifact full_finetune_model_sst2_seed42:latest

# Download by task
python scripts/download_wandb_models.py \
    --entity galavny-tel-aviv-university \
    --project NLP-Phase2-LoRA-Rerun \
    --task sst2
```

Models are downloaded to: `results/downloaded_models/`

### 3. Download Base Representations (If Needed for Drift Analysis)

```bash
# Download base model representations (~48GB)
bash scripts/download_base_representations.sh
```

Representations are saved to: `base_representations/`

### 4. View Analysis

**Option A: Read the Report**
```bash
# Open the comprehensive analysis report
cat ANALYSIS_REPORT.md
```

**Option B: Run Analysis Notebooks**
```bash
# Launch Jupyter
jupyter notebook

# Open:
# - research_question_1_representational_drift.ipynb
# - research_question_2_deployment_efficiency.ipynb
```

## Running Experiments (Optional)

All experiments have been completed. To reproduce:

### Phase 1: Hyperparameter Optimization
```bash
cd scripts/phase1
./vm1.sh  # Run sweep for first configuration
```

### Phase 2: Model Training
```bash
cd scripts/phase2
./vm1_full_finetune.sh  # Train full fine-tuned models
./vm1_lora.sh           # Train LoRA adapters
```

### Phase 3: Representation Extraction
```bash
cd scripts/phase3
python extract_all.py
```

### Phase 4A: Drift Analysis
```bash
cd scripts/phase4
python collect_performance_metrics.py
python compute_layer_drift.py
```

### Phase 4B: Deployment Benchmarks
```bash
cd scripts/phase4b
python deployment_benchmark.py --output-dir deployment
python validate_multi_adapter_correctness.py --output-dir deployment
```

## Key Files

| File | Purpose |
|------|---------|
| `ANALYSIS_REPORT.md` | **Main deliverable** - Comprehensive analysis report |
| `research_question_1_*.ipynb` | RQ1 analysis: Representational drift |
| `research_question_2_*.ipynb` | RQ2 analysis: Deployment efficiency |
| `plan.md` | Research plan, methodology, status |
| `deployment/` | Phase 4B deployment benchmark results |
| `drift_analysis/` | Phase 4A drift analysis results |

## Dependencies

Core packages (see `requirements.txt` for full list):
- `torch==2.1.2+cu118` - PyTorch with CUDA 11.8
- `transformers==4.38.2` - Hugging Face Transformers
- `peft==0.9.0` - Parameter-Efficient Fine-Tuning
- `vllm==0.3.2` - High-performance inference
- `wandb==0.16.3` - Experiment tracking
- `optuna==3.5.0` - Hyperparameter optimization

## Hardware

All experiments were conducted on:
- **GPU:** NVIDIA L4 (24GB VRAM)
- **Platform:** Google Cloud Platform (GCP)
- **Note:** This was a temporary VM setup - all important outputs are tracked in git/WandB

## Citation & Scope

**Study Scope:**
- 3 GLUE classification tasks (MRPC, SST-2, RTE)
- 1 model size (TinyLlama-1.1B, ~2GB)
- 1 LoRA rank (rank-8)
- 3 random seeds per configuration
- Single hardware configuration (NVIDIA L4 GPU)

**Limitations:**
Findings require validation across more diverse tasks, model sizes, and hardware before generalizing. See `ANALYSIS_REPORT.md` for detailed discussion.

## Troubleshooting

**Can't access WandB artifacts?**
- Ensure you have access to the `galavny-tel-aviv-university` organization
- Run `wandb login` and authenticate

**Out of disk space?**
- Models are large (~2GB each). Download only what you need
- Base representations are ~48GB total
- Use `--task` flag to download specific tasks only

**Notebook errors?**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that analysis results exist: `drift_analysis/`, `deployment/`
- If files are missing, they may need to be regenerated from WandB artifacts

## Contact

For questions or access to the `galavny-tel-aviv-university` WandB organization, please contact the project maintainers.
