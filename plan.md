# LoRA Research Implementation Plan

## Research Questions & Significance

This research project investigates two critical questions in parameter-efficient fine-tuning that directly impact production deployment decisions:

1. **Representational Drift Analysis**: Does LoRA truly preserve model internal representations better than full fine-tuning? We will quantify this using centered-kernel alignment (CKA) and layer-wise cosine similarity metrics across all transformer layers and multiple task types.

2. **Deployment Efficiency Trade-offs**: What is the real-world latency penalty when deploying multiple LoRA adapters side-by-side versus merging them in vLLM? This addresses a key production concern for multi-task systems.

**Scientific Significance**: These questions address fundamental gaps in our understanding of efficient fine-tuning methods. While LoRA has gained widespread adoption, rigorous empirical analysis of its representation preservation claims and deployment overhead remains limited. Our findings will inform:
- Continual learning strategies to mitigate catastrophic forgetting
- Production system architecture decisions for multi-task deployments
- Theoretical understanding of low-rank adaptation's impact on model internals

**Hypotheses**: We hypothesize that LoRA (rank 8) will achieve ≤3% accuracy drop compared to full fine-tuning AND either ≥20% less representational drift OR ≤30% inference overhead. Both confirming and refuting these hypotheses constitute valid scientific contributions.

## Task Selection & Diversity

**Four-Task Evaluation Suite**: To ensure robust conclusions across diverse NLP task types while maintaining computational feasibility:

1. **MRPC (Paraphrase Detection)**: Sentence-pair classification requiring semantic similarity understanding
2. **SQuAD v2 (Question Answering)**: Extractive QA with unanswerable questions, requiring reading comprehension
3. **SST-2 (Sentiment Analysis)**: Single-sentence binary classification, fundamental text understanding
4. **RTE (Recognizing Textual Entailment)**: Sentence-pair logical reasoning task

**Rationale for Task Selection**:
- **Diversity**: Covers single-sentence (SST-2), sentence-pair (MRPC, RTE), and span-extraction (SQuAD v2) paradigms
- **Complementary Skills**: Sentiment understanding, paraphrase detection, logical reasoning, and reading comprehension
- **Computational Efficiency**: All tasks can be efficiently fine-tuned on TinyLlama-1.1B within reasonable compute budgets
- **Memory Efficiency**: With optimizations, TinyLlama-1.1B uses only ~2GB GPU memory during training (vs ~26GB theoretical), allowing training on 22GB GPUs
- **Established Baselines**: Well-studied tasks with clear evaluation protocols and existing benchmarks


## Execution Strategy Overview

### Phase Structure and Dependencies

**Three-Phase Execution Design**:
- **Phase 1 (Training)**: All 3 VMs start simultaneously with **zero dependencies**
- **Phase 2a (Parallel Analysis)**: All 3 VMs start simultaneously after Phase 1 completes  
- **Phase 2b (Final Synthesis)**: Single VM after Phase 2a completes

**Why This Structure**:
- **Phase 1**: Training experiments can run completely independently - each VM works on different tasks/methods
- **Phase 2a**: Core analyses (drift, deployment) can run in parallel using trained models from Phase 1
- **Phase 2b**: Final statistical synthesis requires results from Phase 2a analyses
- **Within Each Phase**: No dependencies between VMs, enabling true parallel execution

### Task Definitions

**Phase 1 Tasks (Training)**:

| Task ID | Description | Model | Method |
|---------|-------------|-------|---------|
| `mrpc_full_finetune` | Full parameter fine-tuning on MRPC task | TinyLlama-1.1B | Full fine-tuning |
| `mrpc_lora` | LoRA fine-tuning on MRPC task | TinyLlama-1.1B | LoRA (rank 8) |
| `squad_full_finetune` | Full parameter fine-tuning on SQuAD v2 task | TinyLlama-1.1B | Full fine-tuning |
| `squad_lora` | LoRA fine-tuning on SQuAD v2 task | TinyLlama-1.1B | LoRA (rank 8) |
| `sst2_full_finetune` | Full parameter fine-tuning on SST-2 task | TinyLlama-1.1B | Full fine-tuning |
| `sst2_lora` | LoRA fine-tuning on SST-2 task | TinyLlama-1.1B | LoRA (rank 8) |
| `rte_full_finetune` | Full parameter fine-tuning on RTE task | TinyLlama-1.1B | Full fine-tuning |
| `rte_lora` | LoRA fine-tuning on RTE task | TinyLlama-1.1B | LoRA (rank 8) |
| `baselines_all_tasks` | Majority class, random, and SOTA literature baselines | Various | Baseline methods |

**Phase 2a Tasks (Parallel Analysis)**:

| Task ID | Description | Input Requirements | Output |
|---------|-------------|-------------------|---------|
| `drift_analysis_classification` | Representational drift analysis for classification tasks | Base model + saved representations from MRPC, SST-2, RTE | Drift metrics and visualizations |
| `drift_analysis_qa` | Representational drift analysis for QA task | Base model + saved representations from SQuAD v2 | Drift metrics and visualizations |
| `deployment_bench` | vLLM deployment overhead benchmarking | Trained models and LoRA adapters | Performance benchmarks |
| `correlation_analysis` | Performance-drift correlation studies | Training results and drift metrics | Correlation analysis |

**Phase 2b Tasks (Final Synthesis)**:

| Task ID | Description | Input Requirements | Output |
|---------|-------------|-------------------|---------|
| `statistical_analysis` | Final statistical tests and hypothesis validation | All Phase 1 and 2a results | Statistical test results |
| `visualization` | Generate publication-quality figures and tables | All analysis results | Figures and tables |
| `report_generation` | Compile final analysis report | All results and analyses | Final research report |

**Dependencies Summary**:
- **Phase 1**: No dependencies - all VMs start immediately
- **Phase 2a**: Depends on Phase 1 completion (needs trained models and saved representations)  
- **Phase 2b**: Depends on Phase 2a completion (needs analysis results for synthesis)
- **Within Each Phase**: No dependencies between VMs

## 3-VM Resource Allocation Strategy

**Optimal Distribution Philosophy**: Maximize parallel utilization by eliminating dependencies and splitting work by tasks/methods, similar to distributed pre-training then fine-tuning approach.

**Phase 1 - Training (All VMs start immediately in parallel)**:
- **VM1 (Classification Tasks)**: MRPC full fine-tuning + MRPC LoRA + SST-2 full fine-tuning + SST-2 LoRA (4 balanced classification tasks)
- **VM2 (Mixed Heavy Tasks)**: SQuAD v2 full fine-tuning + SQuAD v2 LoRA + RTE full fine-tuning + RTE LoRA (2 heavy QA + 2 light entailment tasks)
- **VM3 (Analysis & Baselines)**: All baseline experiments (MRPC, SST-2, RTE, SQuAD v2) + **base model representation extraction for all tasks** (baselines + representations)

**Phase 2a - Parallel Analysis (All VMs start immediately after Phase 1)**:
- **VM1 (Classification Analysis)**: Representational drift analysis for MRPC and SST-2 tasks + correlation analysis prep
- **VM2 (QA Analysis + Deployment)**: Representational drift analysis for SQuAD v2 + vLLM deployment benchmarking + RTE drift analysis
- **VM3 (Visualization + Stats Prep)**: Cross-task correlation analysis + visualization preparation + statistical test setup

**Phase 2b - Final Synthesis (Single VM after Phase 2a)**:  
- **VM1 (Statistical Synthesis)**: Final statistical analysis, hypothesis testing, visualization, and report generation
- **VM2 & VM3**: Idle (cost optimization - can be shut down)

**Justification**: This allocation ensures:
1. **No Task Overlap**: Each task (MRPC, SST-2, SQuAD v2, RTE) is handled by only one VM, eliminating resource conflicts
2. **Balanced Load Distribution**: 
   - VM1: 4 classification experiments (balanced medium load)
   - VM2: 4 mixed experiments (2 heavy QA + 2 light entailment = balanced load)
   - VM3: Analysis tasks (baselines + representations = lighter but critical load)
3. **No Dependencies**: All VMs start working immediately with zero coordination overhead
4. **Clear Task Separation**: VM1=Classification, VM2=Mixed, VM3=Analysis (easier monitoring and debugging)
5. **Efficient Resource Usage**: No idle time and optimal GPU utilization across all VMs

**Memory Optimizations Applied**:
- **Representation extraction disabled during training** to reduce memory usage from ~16GB to ~2GB
- **95% GPU memory utilization** instead of default 90% for better resource usage
- **Base model representations extracted separately** on VM3 to avoid memory conflicts during training
- **Batch size optimization**: batch_size=1, eval_batch_size=2, gradient_accumulation=8 for memory efficiency

## Distributed Coordination Protocol

### Code Architecture Perspective

**Repository Structure**:
```
NLP/
├── shared/
│   ├── config.yaml          # Centralized hyperparameters
│   ├── data_utils.py        # Common data loading/preprocessing
│   ├── metrics.py           # Shared evaluation metrics
│   └── wandb_utils.py       # W&B initialization and logging
├── experiments/
│   ├── baselines.py         # Naive/SOTA baseline runners
│   ├── full_finetune.py     # Full fine-tuning experiments
│   ├── lora_finetune.py     # LoRA training experiments
│   ├── drift_analysis.py    # CKA and cosine similarity
│   └── deployment_bench.py  # vLLM benchmarking
├── models/                  # Model utilities and wrappers
├── analysis/                # Analysis and visualization scripts
└── scripts/                 # Phase-organized execution scripts
    ├── phase1/              # Training scripts (parallel execution)
    │   ├── vm1.sh           # MRPC + RTE training
    │   ├── vm2.sh           # SQuAD v2 training  
    │   └── vm3.sh           # SST-2 training + baselines
    ├── phase2a/             # Analysis scripts (parallel execution)
    │   ├── vm1.sh           # Classification drift analysis
    │   ├── vm2.sh           # QA drift + deployment benchmarking
    │   └── vm3.sh           # Correlation analysis
    └── phase2b/             # Synthesis scripts (single VM)
        └── vm1.sh           # Statistical analysis + visualization
```

**Shared Configuration Management**:
```python
# shared/config.yaml
experiment:
  model_name: "meta-llama/Llama-2-1.3b-hf"
  tasks: ["mrpc", "squad_v2", "sst2", "rte"]
  seeds: [42, 1337, 2024]
  
metrics:
  mrpc: ["accuracy", "f1"]
  sst2: ["accuracy"]
  rte: ["accuracy"]
  squad_v2: ["em", "f1"]

training_configs:
  sequence_lengths:
    classification: 512  # MRPC, SST-2, RTE
    qa: 768  # SQuAD v2
  learning_rates:
    full_ft_classification: [1e-5, 2e-5]
    full_ft_qa: [5e-6, 1e-5]
    lora: [1e-4, 3e-4]
  lora_config:
    rank: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
  
wandb:
  # Phase-based project organization
  phase1_project: "NLP-Phase1-Training"      # All training experiments
  phase2_project: "NLP-Phase2-Analysis"      # All analysis experiments
  entity: "galavny-tel-aviv-university"
  
# Note: VM allocation handled by phase-organized scripts in scripts/ directory
# No complex configuration needed - each VM runs its dedicated script
```

### Technical/Operational Perspective

**Initial Setup (All VMs)**:
```bash
# Clone repository on each VM
git clone <repo_url> ~/NLP
cd ~/NLP

# Install dependencies (includes vLLM)
pip install -r requirements.txt

# Download datasets
python scripts/download_datasets.py

# No additional setup needed
```

**Coordination Protocol**:
1. **W&B Monitoring**: Monitor all experiment progress via W&B dashboard
2. **Tmux Sessions**: Run experiments in persistent tmux sessions
3. **Manual Phase Transitions**: Start next phase when previous completes (visible in W&B)
4. **Git Repository**: All code, configs, and results tracked via git commits

## Phase-Based Execution Scripts

### Phase 1: Training (All VMs in Parallel)

**No dependencies - all VMs start immediately**

```bash
# VM1: Run in tmux session
tmux new-session -d -s training "bash scripts/phase1/vm1.sh"

# VM2: Run in tmux session  
tmux new-session -d -s training "bash scripts/phase1/vm2.sh"

# VM3: Run in tmux session
tmux new-session -d -s training "bash scripts/phase1/vm3.sh"

# Monitor progress via W&B dashboard
```

### Phase 2a: Parallel Analysis (All VMs in Parallel)

**Dependency: Requires Phase 1 completion (monitor via W&B)**

```bash
# VM1: Run in tmux session
tmux new-session -d -s analysis "bash scripts/phase2a/vm1.sh"

# VM2: Run in tmux session
tmux new-session -d -s analysis "bash scripts/phase2a/vm2.sh"

# VM3: Run in tmux session
tmux new-session -d -s analysis "bash scripts/phase2a/vm3.sh"

# Monitor progress via W&B dashboard
```

### Phase 2b: Final Synthesis (Single VM)

**Dependency: Requires Phase 2a completion (monitor via W&B)**

```bash
# VM1 only: Run in tmux session
tmux new-session -d -s synthesis "bash scripts/phase2b/vm1.sh"

# VM2 & VM3: Can be shut down for cost optimization
```

### Simple Coordination

**Monitoring and Dependencies**:
- **W&B Dashboard**: Monitor all experiment progress in real-time
- **Tmux Sessions**: Keep experiments running persistently across VMs
- **Manual Phase Transitions**: Start next phase when previous completes (visible in W&B)
- **No Complex Scripts**: Just phase-organized bash scripts for each VM

### W&B Project Organization

**Separate Projects for Clean Organization**:
- **Phase 1 - Training**: `NLP-Phase1-Training`
  - All training experiments (baselines, full fine-tuning, LoRA)
  - VM1, VM2, VM3 training runs
  - Clear separation from analysis phase
  
- **Phase 2 - Analysis**: `NLP-Phase2-Analysis`
  - All analysis experiments (drift analysis, deployment benchmarking)
  - Phase 2a and 2b runs
  - Statistical synthesis results

**Benefits**:
- **No Data Mixing**: Training and analysis results stay separate
- **Clean Dashboard**: Easy to monitor each phase independently  
- **No Overrides**: Phase 2 won't interfere with Phase 1 results
- **Better Organization**: Clear project structure for research workflow

**Phase Script Contents**:
```bash
# Example: scripts/phase1/vm1.sh
#!/bin/bash
set -e  # Exit on error

echo "Starting Phase 1 training on VM1..."
python experiments/full_finetune.py --task mrpc
python experiments/lora_finetune.py --task mrpc
python experiments/full_finetune.py --task rte
python experiments/lora_finetune.py --task rte
echo "Phase 1 VM1 complete"
```

### Complete Workflow Example

**Step-by-Step Execution**:

```bash
# 1. Initial Setup (run on all VMs)
# Install dependencies
pip install -r requirements.txt

# Download datasets (run once on each VM)
python scripts/download_datasets.py

# 2. Phase 1: Start all training in parallel (no dependencies)
# VM1:
tmux new-session -d -s training "bash scripts/phase1/vm1.sh"

# VM2:  
tmux new-session -d -s training "bash scripts/phase1/vm2.sh"

# VM3:
tmux new-session -d -s training "bash scripts/phase1/vm3.sh"

# Monitor progress in W&B dashboard

# 3. Phase 2a: Start analysis when Phase 1 completes (check W&B)
# VM1:
tmux new-session -d -s analysis "bash scripts/phase2a/vm1.sh"

# VM2:
tmux new-session -d -s analysis "bash scripts/phase2a/vm2.sh"

# VM3:
tmux new-session -d -s analysis "bash scripts/phase2a/vm3.sh"

# Monitor progress in W&B dashboard

# 4. Phase 2b: Final synthesis when Phase 2a completes (check W&B)
# VM1 only:
tmux new-session -d -s synthesis "bash scripts/phase2b/vm1.sh"

# Shut down VM2 & VM3 for cost savings
```

## Dataset Management Strategy

### Dataset Storage Decision

**Approach: Download datasets using provided script**

**Rationale**:
- **GitHub size limits**: Some dataset files exceed GitHub's 100MB limit
- **Reproducibility**: Script ensures exact same data versions across all VMs
- **Simplicity**: Single script downloads all datasets with integrity checks
- **Reliability**: Uses official HuggingFace datasets for consistent versions

**Dataset Sizes**:
- MRPC: ~1MB (3.7K train, 408 dev examples)
- SST-2: ~5MB (67K train, 872 dev examples)  
- RTE: <1MB (2.5K train, 277 dev examples)
- SQuAD v2: ~45MB (130K train, 12K dev examples)

**Data Structure** (created by download script):
```
data/
├── manifest.json        # Dataset metadata and sizes
├── mrpc/                # GLUE-MRPC (3.7K train, 408 dev, 1.7K test)
├── sst2/                # GLUE-SST2 (67K train, 872 dev, 1.8K test)  
├── rte/                 # GLUE-RTE (2.5K train, 277 dev, 3K test)
└── squad_v2/            # SQuAD v2.0 (130K train, 12K dev)
```

**Download Command**:
```bash
python scripts/download_datasets.py
```

All datasets are saved in HuggingFace datasets format for easy loading with `datasets.load_from_disk()`.

### Requirements and Setup

**Simple Setup**:
```bash
# Setup needed on each VM:
pip install -r requirements.txt

# Download datasets (run once):
python scripts/download_datasets.py
```

**No Complex Setup Needed**: 
- **Datasets**: Download script included (`scripts/download_datasets.py`) - downloads all four tasks
- **Dependencies**: All package versions (including vLLM) specified in `requirements.txt` in repository root
- **Models**: All model downloading handled automatically by HuggingFace transformers
- **W&B**: Login handled in experiment scripts
- **No environment variables or special configuration required**

## Step 1: Environment Setup & Sanity Checks

### Agent Prompt

```
You are setting up the experimental environment for a rigorous NLP research project comparing LoRA and full fine-tuning on Llama-2-1.3B. Your primary goal is to ensure complete reproducibility and validate the experimental setup.

CONTEXT:
- Model: meta-llama/Llama-2-1.3b-hf (chosen for computational feasibility)
- Tasks: MRPC, SQuAD v2, SST-2, RTE (four diverse NLP tasks for robust evaluation)
- Infrastructure: 3 GPU VMs with task-based parallel allocation
- Tracking: Weights & Biases (Phase 1: "NLP-Phase1-Training", Phase 2: "NLP-Phase2-Analysis", entity: "galavny-tel-aviv-university")

REQUIREMENTS:
1. Implement data loading utilities for all four tasks
2. Create sanity check scripts that verify:
   - Model can overfit 10 examples from each dataset
   - Gradient flow is correct for both full fine-tuning and LoRA
   - W&B logging works correctly
   - Reproducibility across multiple runs with same seed

DELIVERABLES:
1. requirements.txt with pinned versions for all packages (already created in repository root)
2. scripts/download_datasets.py for downloading all four datasets
3. shared/data_preparation.py for loading and preprocessing datasets
4. Phase-organized execution scripts (scripts/phase{1,2a,2b}/vm{1,2,3}.sh)
5. Sanity check functionality built into each experiment script
6. Basic experiment configuration in shared/config.yaml

TECHNICAL SPECIFICATIONS:
- See requirements.txt in repository root for exact package versions (includes vLLM)
- Dataset download via scripts/download_datasets.py script 
- Minimal setup: `pip install -r requirements.txt` + `python scripts/download_datasets.py`

Focus on creating the phase-organized execution scripts and data loading utilities rather than complex setup procedures.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Load a small sample from each dataset (10 examples per task)
2. Run a quick sanity check (1 epoch overfitting test)
3. Verify W&B logging works correctly
4. Check that all results are saved to W&B dashboard
5. Validate reproducibility with fixed seeds
```

### Step 1 Validation Instructions

**How to validate this step is working correctly**:

1. **Check Dataset Loading**:
   ```bash
   python -c "from datasets import load_from_disk; print(load_from_disk('data/mrpc')['train'][0])"
   ```
   Should display a sample MRPC example without errors.

2. **Verify W&B Connection**:
   - Check that W&B dashboard shows new runs under project "NLP-Phase1-Training"
   - Confirm entity "galavny-tel-aviv-university" is accessible
   - Verify metrics are being logged in real-time

3. **Test Sanity Checks**:
   - Run the 10-example overfitting test for each task
   - Should achieve 100% accuracy within 50 epochs
   - Check that loss decreases monotonically

4. **Validate Reproducibility**:
   - Run same configuration twice with same seed
   - Results should be identical (within numerical precision)
   - Check that random state is properly controlled

**Red Flags to Watch For**:
- Dataset loading errors or missing files
- W&B authentication failures
- Models not overfitting on small samples
- Non-reproducible results across identical runs

### 3-VM Distribution

**Setup Task Distribution**:
- **All VMs**: Run `pip install -r requirements.txt` (includes vLLM)
- **All VMs**: Run `python scripts/download_datasets.py` to download all four datasets

**Rationale**: Minimal setup requirements with download script for reproducible dataset versions while avoiding GitHub size limits.

### Sanity Check Protocol

1. **Small-Sample Overfitting Test**:
   - Select 10 random examples from each dataset
   - Train for 50 epochs with high learning rate
   - Verify 100% training accuracy (critical for catching bugs)

2. **Gradient Verification**:
   - Compare gradient magnitudes between full fine-tuning and LoRA
   - Ensure LoRA only updates adapter weights
   - Verify gradient clipping works correctly

3. **Reproducibility Test**:
   - Run same configuration 3 times with fixed seed
   - Verify identical loss curves and final metrics
   - Test with different batch accumulation strategies

### Validation Criteria

- Environment setup completes without errors on all VMs
- Data preprocessing produces consistent outputs
- Sanity checks pass with <1% variation across runs
- W&B dashboard shows all expected metrics
- Model checkpoints can be loaded across VMs

## Step 2: Baseline Establishment

### Agent Prompt

```
You are implementing comprehensive baseline experiments for the LoRA research project. These baselines are CRITICAL for contextualizing all experimental results and are required for rigorous methodology.

CONTEXT:
- Already completed: Environment setup and sanity checks passed
- Model: Llama-2-1.3b-hf 
- Tasks: MRPC (accuracy, F1), SQuAD v2 (EM, F1), SST-2 (accuracy), RTE (accuracy)
- Your role: Establish strong baselines for meaningful comparison across all four task types

REQUIRED BASELINES (for performance context):
1. MAJORITY CLASS CLASSIFIER:
   - For MRPC: Always predict most frequent label in training set
   - For SST-2: Always predict most frequent sentiment class
   - For RTE: Always predict most frequent entailment label
   - For SQuAD: Always predict "no answer" for unanswerable questions
   - Report accuracy, F1, and confidence intervals

2. RANDOM BASELINE:
   - For MRPC/SST-2/RTE: Random predictions with class distribution matching training set
   - For SQuAD: Random span selection for answerable, "no answer" for others
   - Run with 5 different seeds, report mean and std

3. SOTA BASELINE FROM LITERATURE:
   - For MRPC: Use published RoBERTa-base results (~90.7% F1)
   - For SQuAD v2: Use published ALBERT-base results (~89.7% F1)
   - For SST-2: Use published BERT-base results (~93.5% accuracy)
   - For RTE: Use published BERT-base results (~66.5% accuracy)
   - These provide performance ceiling context without additional implementation

DRIFT ANALYSIS BASELINE:
- Original pre-trained Llama-2-1.3B model (no task-specific training)
- Extract representations from validation examples to serve as baseline for CKA and cosine similarity analysis
- No prompting or task-specific formatting needed - just raw model representations

IMPLEMENTATION REQUIREMENTS:
- Implement majority/random baselines in experiments/baselines.py with modular functions
- Use published SOTA numbers for literature comparison (no additional training needed)
- Extract pre-trained model representations for drift analysis baseline
- Use consistent evaluation code in shared/metrics.py across all baselines
- Log all results to W&B with clear naming convention
- Implement proper cross-validation for robust estimates

STATISTICAL RIGOR:
- Run each baseline with at least 3 random seeds
- Report mean, std, and 95% confidence intervals
- Use bootstrap sampling (n=1000) for confidence intervals
- Implement McNemar's test for pairwise comparisons

Output comprehensive baseline_results.json with all metrics and statistical tests.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Run majority class baseline on 100 examples from each task
2. Run random baseline with multiple seeds
3. Verify all baseline results are logged to W&B
4. Check that statistical tests produce reasonable p-values
5. Validate that confidence intervals are computed correctly
```

### Step 2 Validation Instructions

**How to validate baseline experiments are working correctly**:

1. **Check Baseline Results**:
   ```bash
   # Should see baseline runs in W&B dashboard
   # Majority class accuracy should match expected values (~50% for MRPC, ~68% for RTE)
   ```

2. **Verify Zero-Shot Performance**:
   - SST-2: Should achieve 80-85% accuracy (strong pre-training signal)
   - MRPC: Should achieve 60-70% accuracy  
   - RTE: Should achieve 55-65% accuracy
   - SQuAD v2: Should achieve 20-30% EM (harder for pre-trained model)

3. **Statistical Test Validation**:
   - McNemar's test should produce p-values < 0.05 for meaningful differences
   - Bootstrap confidence intervals should be reasonable (not too wide)
   - Multiple seeds should show consistent patterns

4. **W&B Dashboard Check**:
   - All baseline experiments appear with clear naming
   - Metrics are logged correctly per task
   - Run groups link related experiments

**Red Flags to Watch For**:
- Baseline performance wildly different from expected ranges
- Statistical tests failing or producing NaN values
- Missing W&B logs or incomplete metric tracking
- Confidence intervals that are unreasonably wide or narrow

### Statistical Design

**Multiple Seeds Protocol**:
- Seeds: [42, 1337, 2024] for primary runs
- Additional seeds [3407, 5489] for critical comparisons
- Separate data shuffling seed from model initialization seed

**Significance Testing**:
- McNemar's test for accuracy comparisons
- Permutation tests for F1 scores
- Bonferroni correction for multiple comparisons

**Effect Size Calculation**:
- Cohen's d for continuous metrics
- Odds ratios for binary classifications
- Practical significance thresholds (>3% accuracy difference)

### Baseline Comparisons

1. **Naive Classifiers** (establish floor):
   - Majority class: ~50% for MRPC, ~50% for SST-2, ~68% for RTE
   - Random: Expected 50% for classification tasks, ~0% EM for SQuAD

2. **Zero-shot LLaMA-2** (test pre-training knowledge):
   - Expected: 60-70% MRPC, 80-85% SST-2, 55-65% RTE, 20-30% SQuAD EM

3. **SOTA Fine-tuned** (establish ceiling):
   - RoBERTa-base: ~90% MRPC accuracy, ~94% SST-2 accuracy
   - BERT-base on RTE: ~70% accuracy
   - ALBERT-base on SQuAD v2: ~80% F1

## Step 3: Full Fine-tuning Experiments

### Agent Prompt

```
You are implementing full fine-tuning experiments for Llama-2-1.3B on MRPC, SQuAD v2, SST-2, and RTE. This serves as the primary comparison point for LoRA experiments.

CONTEXT:
- Running in parallel with other training (balanced load: VM1 SQuAD+MRPC, VM2 SQuAD+SST-2, VM3 RTE+baselines)
- Model: meta-llama/Llama-2-1.3b-hf
- Hardware: Load-balanced task splitting for optimal parallel execution
- Objective: Establish full fine-tuning performance and representation changes across all four tasks

EXPERIMENTAL DESIGN:
1. HYPERPARAMETER SEARCH:
   - Learning rates: [1e-5, 2e-5] for classification tasks (MRPC, SST-2, RTE)
   - Learning rates: [5e-6, 1e-5] for QA task (SQuAD v2)
   - Batch sizes: [8, 16] with gradient accumulation as needed
   - Sequence lengths: 512 for classification, 768 for SQuAD v2
   - Epochs: Early stopping with patience=3 on validation loss
   - Warmup: 10% of training steps
   - Use W&B sweeps for systematic search

2. TRAINING PROTOCOL:
   - Mixed precision training (bf16) for efficiency
   - Gradient checkpointing if OOM
   - Save checkpoints every 500 steps
   - Log gradients and activation statistics

3. REPRESENTATION TRACKING:
   - Extract and save hidden states every 100 steps for drift analysis
   - Save representations from 1000 validation examples
   - Store activations from all transformer layers
   - Use memory-mapped files for efficient storage

4. EVALUATION PROTOCOL:
   - Evaluate on validation set every 100 steps
   - Final evaluation on test set (only once!)
   - Generate prediction files for error analysis
   - Save all model outputs for statistical testing

IMPLEMENTATION DETAILS:
- Implement training in experiments/full_finetune.py with full configurability
- Use HuggingFace Trainer with custom callbacks in models/ directory
- Implement custom representation extraction callback
- Create gradient statistics monitoring
- Ensure deterministic training with fixed seeds

CRITICAL REQUIREMENTS:
- Run each configuration with seeds [42, 1337, 2024]
- Save model checkpoints for best validation performance
- Log training dynamics (loss, gradients, learning rate)
- Extract representations for drift analysis (every 100 steps)
- **Extract base model representations**: Run original pre-trained Llama-2-1.3B on validation sets (no fine-tuning)
- Profile memory usage and training time

Output all results to W&B and save checkpoints locally for later analysis phases.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Run full fine-tuning on one task (SST-2) for 1 epoch with 100 examples
2. Monitor training metrics in W&B (loss, accuracy, learning rate)
3. Extract and save representations for a few validation examples
4. Verify checkpoints are saved and can be loaded correctly
5. Check that gradient statistics and memory usage are logged
```

### Step 3 Validation Instructions

**How to validate full fine-tuning experiments are working correctly**:

1. **Training Progress Monitoring**:
   ```bash
   # Check W&B dashboard for:
   # - Training/validation loss curves
   # - Accuracy metrics per task
   # - Learning rate schedules
   # - Gradient norms and statistics
   ```

2. **Performance Validation**:
   - **MRPC**: Should reach 85-90% accuracy (close to SOTA)
   - **SST-2**: Should reach 90-93% accuracy 
   - **RTE**: Should reach 65-75% accuracy
   - **SQuAD v2**: Should reach 75-85% F1 score

3. **Representation Extraction Check**:
   - Verify representations are saved every 100 steps
   - Check file sizes are reasonable (not empty or corrupted)
   - Test loading saved representations works correctly

4. **Checkpoint Validation**:
   - Saved models can be loaded without errors
   - Model outputs are consistent after loading
   - Training can resume from checkpoints correctly

**Red Flags to Watch For**:
- Training loss not decreasing or unstable
- Performance much lower than expected ranges
- Missing or corrupted representation files
- Checkpoint loading failures
- Memory errors or GPU OOM issues

### 3-VM Distribution (Updated)

**Execution Commands**:
```bash
# VM1: SQuAD v2 Full FT + MRPC Mix
tmux new-session -d -s phase1 './scripts/phase1/vm1.sh'

# VM2: SQuAD v2 LoRA + SST-2 Mix  
tmux new-session -d -s phase1 './scripts/phase1/vm2.sh'

# VM3: RTE + Baselines + Base Representations
tmux new-session -d -s phase1 './scripts/phase1/vm3.sh'
```

**Load Balancing**:
- All VMs start immediately (no dependencies)
- Balanced computational load across 3 VMs
- All seeds [42, 1337, 2024] for reproducibility
- Base model representations extracted on VM3 separately

### Training Monitoring

1. **Convergence Criteria**:
   - Validation loss plateaus for 3 consecutive evaluations
   - Training loss < 0.1 (sanity check)
   - Gradient norms stabilize

2. **Quality Checks**:
   - Verify no catastrophic forgetting of language modeling
   - Check attention patterns remain reasonable
   - Monitor for training instabilities

## Step 4: LoRA Experiments

### Agent Prompt

```
You are implementing LoRA (Low-Rank Adaptation) experiments with rank 8 for Llama-2-1.3B. Your goal is to match or exceed full fine-tuning performance while dramatically reducing parameter updates.

CONTEXT:
- PREVIOUS WORK COMPLETED: Steps 1-3 have been implemented and validated
  * Step 1: Environment setup, data pipeline, and sanity checks COMPLETE
  * Step 2: All baseline experiments (majority/random/SOTA) COMPLETE & VALIDATED
  * Step 3: Full fine-tuning implementation COMPLETE & VALIDATED
- CURRENT STATUS: Ready to implement LoRA experiments as Step 4
- INFRASTRUCTURE: Complete experimental framework with W&B logging, load-balanced VM allocation
- BASELINE PERFORMANCE: Established reference scores for all 4 tasks (MRPC, SQuAD v2, SST-2, RTE)
- Target: ≤3% accuracy drop compared to full fine-tuning across all four tasks
- LoRA rank: 8 (fixed as per research protocol)
- Hardware: Load-balanced allocation (VM1: SQuAD+MRPC, VM2: SQuAD+SST-2, VM3: RTE+baselines)

LORA CONFIGURATION:
1. ARCHITECTURE SETTINGS:
   - Rank (r): 8
   - Alpha: 16 (scaling factor = alpha/r = 2)
   - Target modules: ["q_proj", "v_proj"] (query and value projections)
   - Dropout: 0.05 for regularization
   - Initialize with Kaiming uniform

2. HYPERPARAMETER SEARCH:
   - Learning rates: [1e-4, 3e-4] (typically higher than full fine-tuning)
   - LoRA-specific warmup: 6% of total steps
   - Use same batch sizes and sequence lengths as full fine-tuning for fair comparison
   - Early stopping with same criteria
   - Dropout: 0.05 for regularization

3. EXPERIMENTAL VARIATIONS:
   - Standard LoRA: Target Q,V projections
   - Extended LoRA: Additionally target K,O projections
   - All-layer LoRA: Target all linear layers (maximum coverage)
   - Compare parameter efficiency vs performance trade-offs

4. REPRESENTATION TRACKING:
   - Extract representations at same intervals as full fine-tuning
   - Save both adapted and base model representations
   - Track adapter weight magnitudes over training
   - Monitor rank utilization (singular value analysis)

IMPLEMENTATION REQUIREMENTS:
- BUILD ON EXISTING CODEBASE: 
  * Use established shared/data_preparation.py for data loading (already validated)
  * Follow shared/config.yaml structure for hyperparameters (already configured)
  * Integrate with existing shared/metrics.py evaluation framework (already working)
  * Use same W&B logging patterns as experiments/baselines.py and experiments/full_finetune.py
- IMPLEMENT NEW: experiments/lora_finetune.py using PEFT library
- CREATE UTILITIES: models/lora_utils.py for parameter efficiency analysis:
  * Trainable parameters: ~0.3% of full model
  * Memory usage during training  
  * Actual vs theoretical speedup
- VALIDATION TOOLS: LoRA merge testing utilities:
  * Test merged model equivalence
  * Benchmark merged vs adapter inference
- ABLATION STUDIES: Include in same script:
  * Different rank values [4, 8, 16] for comparison
  * Impact of alpha scaling
  * Module selection impact

CRITICAL VALIDATION:
- Verify LoRA updates don't affect base model weights
- Ensure reproducibility across seeds [42, 1337, 2024]
- Compare convergence speed vs full fine-tuning
- Validate that merged model produces identical outputs

Save all LoRA adapters locally and log all results to W&B for later analysis phases.

AVAILABLE RESOURCES FROM PREVIOUS STEPS:
- experiments/baselines.py: Working baseline implementations for comparison
- experiments/full_finetune.py: Complete full fine-tuning pipeline to compare against
- shared/data_preparation.py: Validated data loading for all 4 tasks
- shared/metrics.py: Established evaluation framework  
- shared/config.yaml: Hyperparameter configuration with VM allocation
- Phase 1 scripts: Ready for parallel execution once LoRA is implemented

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Run LoRA training on one task (SST-2) for 1 epoch with 100 examples
2. Verify only LoRA parameters update (base model frozen)
3. Test adapter merging and equivalence with separate loading
4. Monitor LoRA-specific metrics in W&B (adapter weights, rank utilization)
5. Validate parameter efficiency (should be ~0.3% of full model)
6. Confirm integration with existing codebase (data loading, metrics, W&B patterns)
```

### Step 4 Validation Instructions

**How to validate LoRA experiments are working correctly**:

1. **LoRA Parameter Verification**:
   ```bash
   # Check that only LoRA parameters have gradients
   # Base model parameters should remain frozen
   # Trainable parameters should be ~0.3% of total
   ```

2. **Performance Validation**:
   - **Performance Gap**: LoRA should be within 3% of full fine-tuning performance
   - **Training Speed**: LoRA should train faster than full fine-tuning
   - **Memory Usage**: LoRA should use significantly less memory

3. **Adapter Functionality Check**:
   - Test adapter loading/unloading works correctly
   - Verify merged model produces identical outputs to adapter model
   - Check adapter weight magnitudes are reasonable (not zero or huge)

4. **W&B LoRA Metrics**:
   - Adapter weight distributions logged correctly
   - Rank utilization metrics tracked
   - Training efficiency metrics (speed, memory) compared to full FT

**Red Flags to Watch For**:
- Base model parameters updating (should be frozen)
- LoRA performance significantly worse than full fine-tuning
- Adapter merging producing different outputs
- Unreasonable adapter weight values (all zeros or exploding)
- Missing LoRA-specific metrics in W&B

### LoRA Execution Strategy

**Memory Efficiency**:
- LoRA uses ~6-8GB vs ~15GB for full fine-tuning
- Allows faster training and more concurrent experiments
- Gradient checkpointing for longer sequences (SQuAD v2)
- Automatic checkpoint cleanup (keep only best)

### LoRA-Specific Validation

1. **Adapter Verification**:
   - Confirm only LoRA parameters update
   - Verify parameter count matches theory
   - Test adapter loading/unloading

2. **Merge Testing**:
   - Merged model == base + adapter (numerical precision)
   - No performance degradation after merging
   - Memory footprint validation

## Step 5: Representational Drift Analysis

### Agent Prompt

```
You are conducting comprehensive representational drift analysis comparing full fine-tuning and LoRA. This analysis is critical for understanding how different training methods affect model internals.

CONTEXT:
- PREVIOUS WORK COMPLETED: Steps 1-4 have been implemented and validated
  * Step 1: Environment setup, data pipeline, and sanity checks COMPLETE
  * Step 2: All baseline experiments (majority/random/SOTA) COMPLETE & VALIDATED
  * Step 3: Full fine-tuning implementation COMPLETE & VALIDATED
  * Step 4: LoRA experiments implementation COMPLETE & VALIDATED
- CURRENT STATUS: Ready to analyze representational drift as Step 5
- INFRASTRUCTURE: Complete experimental framework with W&B logging, load-balanced VM allocation
- AVAILABLE DATA: Layer-wise representations every 100 training steps for all tasks saved from Steps 3-4
- BASELINE ESTABLISHED: Original pre-trained Llama-2-1.3B representations extracted for comparison
- Objective: Quantify if LoRA preserves representations better (≥20% less drift) across task types

ANALYSIS METHODS:
1. CENTERED KERNEL ALIGNMENT (CKA):
   - **Baseline**: Original pre-trained Llama-2-1.3B (NO task-specific training)
   - **Comparison**: Base model vs Full fine-tuned model, Base model vs LoRA fine-tuned model
   - Implementation: Use linear CKA (more stable than RBF)
   - Compute between base and fine-tuned representations at each layer
   - Create layer × training_step heatmaps
   - Statistical test: Permutation test for significance

2. LAYER-WISE COSINE SIMILARITY:
   - **Baseline**: Same original pre-trained Llama-2-1.3B model
   - Average cosine similarity between base and fine-tuned representations per layer
   - Track evolution over training steps
   - Identify which layers change most during fine-tuning
   - Compare drift patterns between full fine-tuning and LoRA methods

3. REPRESENTATION ANALYSIS PROTOCOL:
   - **Base Model Representations**: Run original pre-trained Llama-2-1.3B on validation examples (no fine-tuning)
   - **Fine-tuned Representations**: Run trained models on same validation examples
   - Use adaptive validation sampling: 750-sample limit (uses all samples for small tasks, optimized for large tasks)
   - Rationale: Adaptive approach preserves 100% coverage for MRPC/RTE while optimizing SST-2/SQuAD v2 (44% memory reduction, maintains statistical validity)
   - Extract representations from all transformer layers for all models
   - Include attention patterns and MLP activations
   - Analyze both token-level and sequence-level representations
   - Generate per-task and cross-task drift summaries

4. STATISTICAL ANALYSIS:
   - Compute drift metrics for each seed separately
   - Report mean ± std across seeds
   - Use permutation tests (n=10000) for p-values
   - Calculate effect sizes (Cohen's d)

IMPLEMENTATION DETAILS:
- Implement analysis in experiments/drift_analysis.py
- Create visualization utilities in analysis/ directory
- Use shared utilities from shared/ for data loading

```python
# CKA Implementation in experiments/drift_analysis.py
def linear_cka(X, Y):
    # Center the matrices
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    
    # Compute CKA
    YTX = Y.T @ X
    norm = np.sqrt(np.trace(X.T @ X) * np.trace(Y.T @ Y))
    return np.trace(YTX @ YTX.T) / (norm ** 2)

# Extract base model representations (original pre-trained Llama-2-1.3B)
base_model = load_pretrained_model("meta-llama/Llama-2-1.3b-hf")  # NO fine-tuning
base_representations = extract_representations(base_model, validation_examples)

# Analysis per layer
for layer in range(num_layers):
    base_repr = base_representations[layer]  # Original pre-trained model
    ft_repr = load_finetuned_representations(layer)  # Full fine-tuned model
    lora_repr = load_lora_representations(layer)  # LoRA fine-tuned model
    
    # Compute representational drift (how much fine-tuning changed representations)
    ft_drift = 1 - linear_cka(base_repr, ft_repr)  # Base vs Full FT
    lora_drift = 1 - linear_cka(base_repr, lora_repr)  # Base vs LoRA
    
    # Statistical test: Does LoRA preserve representations better?
    drift_reduction = (ft_drift - lora_drift) / ft_drift * 100
```

VISUALIZATION REQUIREMENTS:
- Create drift evolution plots over training steps
- Generate layer-wise drift comparison heatmaps
- Plot confidence intervals for all metrics
- Create publication-quality figures

CRITICAL ANALYSES:
- Early vs late layer drift patterns across all tasks
- Task-specific drift differences (classification vs QA vs sentiment vs entailment)
- Cross-task drift pattern generalization
- Correlation between drift and performance across task types
- Identify "critical" layers with highest drift for each task type

Output comprehensive drift_analysis_results.json and all visualization figures.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Load base model and extract representations on 50 examples from one task
2. Load one fine-tuned model and extract representations on same examples
3. Compute CKA and cosine similarity for a few layers
4. Generate a simple drift visualization plot
5. Verify all drift metrics are logged to W&B with clear naming
```

### Step 5 Validation Instructions

**How to validate drift analysis is working correctly**:

1. **Representation Loading Check**:
   ```bash
   # Test loading base model representations
   # Test loading fine-tuned model representations  
   # Verify dimensions match and no NaN values
   ```

2. **CKA Computation Validation**:
   - **CKA values**: Should be between 0 and 1
   - **Drift values**: Should be between 0 and 1 (1 - CKA)
   - **Layer patterns**: Early layers should have less drift than later layers
   - **LoRA vs Full FT**: LoRA should show consistently lower drift values

3. **Statistical Analysis Check**:
   - Permutation tests should produce meaningful p-values
   - Confidence intervals should be computed correctly
   - Effect sizes (Cohen's d) should be reasonable

4. **Visualization Validation**:
   - Drift evolution plots show clear trends over training steps
   - Layer-wise heatmaps display interpretable patterns
   - Per-task drift comparisons are clearly visible

**Red Flags to Watch For**:
- CKA values outside [0,1] range or all NaN
- Drift patterns that don't make sense (e.g., drift decreasing during training)
- LoRA showing higher drift than full fine-tuning across all layers
- Missing or corrupted representation files
- Statistical tests producing unreasonable results

### Drift Analysis Strategy

**Implementation Approach**:
- Base model representations extracted once on VM3
- Layer-wise analysis across all transformer layers
- Statistical validation with permutation tests
- Memory-efficient processing of large representation files

### Analysis Pipeline

1. **Data Loading**:
   - Memory-map large representation files
   - Batch process to avoid OOM
   - Verify representation integrity

2. **Compute Pipeline**:
   - Parallelize CKA computation across layers
   - Cache intermediate results
   - Implement checkpointing for long analyses

3. **Statistical Rigor**:
   - Bootstrap confidence intervals
   - Multiple hypothesis correction
   - Sensitivity analysis on example selection

## Step 6: Deployment Benchmarking

### Agent Prompt

```
You are conducting comprehensive deployment benchmarking using vLLM to measure real-world inference performance of multiple LoRA adapters versus merged models.

CONTEXT:
- PREVIOUS WORK COMPLETED: Steps 1-5 have been implemented and validated
  * Step 1: Environment setup, data pipeline, and sanity checks COMPLETE
  * Step 2: All baseline experiments (majority/random/SOTA) COMPLETE & VALIDATED
  * Step 3: Full fine-tuning implementation COMPLETE & VALIDATED
  * Step 4: LoRA experiments implementation COMPLETE & VALIDATED
  * Step 5: Representational drift analysis COMPLETE & VALIDATED
- CURRENT STATUS: Ready to benchmark deployment performance as Step 6
- INFRASTRUCTURE: Complete experimental framework with vLLM installed and load-balanced VM allocation
- AVAILABLE MODELS: Trained full fine-tuned and LoRA models for all four tasks (MRPC, SQuAD v2, SST-2, RTE)
- BASELINE PERFORMANCE: Established training and validation metrics from previous steps
- Objective: Quantify deployment overhead (target: ≤30% for multi-adapter) across task types

BENCHMARKING SCENARIOS:
1. BASELINE MEASUREMENTS:
   - Original Llama-2-1.3B (no modifications)
   - Fully fine-tuned models for representative tasks
   - Single merged LoRA models per task

2. MULTI-ADAPTER DEPLOYMENT:
   - 2 adapters: SST-2 + SQuAD simultaneously (classification + QA)
   - 4 adapters: All four task adapters (MRPC, SQuAD v2, SST-2, RTE)
   - Dynamic adapter switching overhead between task types
   - Focus on representative classification (SST-2) and QA (SQuAD v2) for detailed analysis

3. METRICS TO COLLECT:
   - Throughput: Tokens/second at various batch sizes [1,2,4,8,16]
   - Latency: 50th, 95th, 99th percentile
   - Memory usage: Base + per-adapter overhead
   - First token latency (critical for interactive use)

4. IMPLEMENTATION PROTOCOL:
- Implement benchmarking in experiments/deployment_bench.py
- Use shared configuration from shared/config.yaml
- Save results to analysis/ directory

```python
import vllm
from vllm import LLM, SamplingParams

# Test configurations
configs = {
    "baseline": LLM(model="meta-llama/Llama-2-1.3b-hf"),
    "merged_sst2": LLM(model="./checkpoints/merged_models/llama2_sst2"),
    "merged_squad": LLM(model="./checkpoints/merged_models/llama2_squad"),
    "multi_adapter_2": LLM(
        model="meta-llama/Llama-2-1.3b-hf",
        enable_lora=True,
        max_lora_rank=8,
        lora_modules=["./checkpoints/adapters/sst2", "./checkpoints/adapters/squad"]
    ),
    "multi_adapter_4": LLM(
        model="meta-llama/Llama-2-1.3b-hf",
        enable_lora=True,
        max_lora_rank=8,
        lora_modules=["./checkpoints/adapters/mrpc", "./checkpoints/adapters/squad", 
                     "./checkpoints/adapters/sst2", "./checkpoints/adapters/rte"]
    )
}

# Benchmark protocol
def benchmark_config(llm, prompts, batch_sizes=[1,2,4,8,16]):
    results = {}
    for batch_size in batch_sizes:
        # Warm up
        _ = llm.generate(prompts[:10], sampling_params)
        
        # Measure
        start_time = time.time()
        outputs = llm.generate(
            prompts[:batch_size * 100],
            sampling_params,
            batch_size=batch_size
        )
        
        # Calculate metrics
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        elapsed = time.time() - start_time
        
        results[batch_size] = {
            "throughput": total_tokens / elapsed,
            "latency_p95": calculate_p95_latency(outputs),
            "memory_gb": get_gpu_memory_usage()
        }
    return results
```

STRESS TESTING:
- Adapter switching frequency impact
- Memory pressure with many adapters
- Performance degradation curves
- Cold start vs warm inference

STATISTICAL REQUIREMENTS:
- Run each benchmark 10 times
- Report median and MAD (median absolute deviation)
- Test statistical significance of overhead
- Create performance regression models

Generate deployment_benchmark_results.json with all metrics and recommendations.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Load baseline Llama-2 model in vLLM and run inference on 10 examples
2. Load one merged LoRA model and test inference equivalence
3. Load 2-adapter configuration and test adapter switching
4. Measure basic throughput and latency metrics
5. Verify all benchmarking results are logged to W&B
```

### Step 6 Validation Instructions

**How to validate deployment benchmarking is working correctly**:

1. **vLLM Setup Check**:
   ```bash
   # Test basic vLLM functionality
   python -c "from vllm import LLM; model = LLM('meta-llama/Llama-2-1.3b-hf'); print('vLLM working')"
   ```

2. **Model Loading Validation**:
   - **Baseline Model**: Original Llama-2 loads without errors
   - **Merged Models**: LoRA-merged models load and produce outputs
   - **Multi-Adapter**: 2-adapter and 4-adapter configurations work
   - **Equivalence**: Merged and adapter models produce same outputs

3. **Performance Metrics Check**:
   - **Throughput**: Measured in tokens/second across batch sizes
   - **Latency**: P95 latency values are reasonable (not extreme outliers)
   - **Memory**: GPU memory usage tracked correctly
   - **Overhead**: Multi-adapter overhead quantified vs merged models

4. **Benchmarking Results Validation**:
   - Results show expected patterns (overhead increases with more adapters)
   - Statistical significance tests work correctly
   - Performance regression models fit the data well

**Red Flags to Watch For**:
- vLLM setup failures or model loading errors
- Merged and adapter models producing different outputs
- Unreasonable performance metrics (negative latency, impossible throughput)
- Multi-adapter setup not working correctly
- Missing benchmarking data in W&B

### Deployment Benchmarking Strategy

**vLLM Performance Analysis**:
- Baseline vs merged vs multi-adapter configurations
- Throughput and latency measurements across batch sizes  
- Memory usage and overhead quantification
- Statistical significance testing of performance differences

### Benchmarking Protocol

1. **Warm-up Phase**:
   - 100 inference calls before measurement
   - Ensure GPU memory is settled
   - Verify vLLM cache is populated

2. **Measurement Phase**:
   - 1000 prompts per configuration
   - Vary sequence lengths [128, 256, 512]
   - Test both greedy and sampling decoding

3. **Overhead Analysis**:
   - Base model memory as reference
   - Per-adapter memory increment
   - Switching latency between adapters
   - CPU-GPU transfer overhead

## Step 7: Statistical Analysis Pipeline

### Agent Prompt

```
You are implementing the final statistical analysis pipeline to synthesize all experimental results and test the research hypotheses with rigorous statistical methods.

CONTEXT:
- PREVIOUS WORK COMPLETED: Steps 1-6 have been implemented and validated
  * Step 1: Environment setup, data pipeline, and sanity checks COMPLETE
  * Step 2: All baseline experiments (majority/random/SOTA) COMPLETE & VALIDATED
  * Step 3: Full fine-tuning implementation COMPLETE & VALIDATED
  * Step 4: LoRA experiments implementation COMPLETE & VALIDATED
  * Step 5: Representational drift analysis COMPLETE & VALIDATED
  * Step 6: Deployment benchmarking COMPLETE & VALIDATED
- CURRENT STATUS: Ready to perform final statistical analysis as Step 7
- INFRASTRUCTURE: Complete experimental framework with all results collected in W&B and local storage
- AVAILABLE DATA: Complete experimental results for all four tasks (MRPC, SQuAD v2, SST-2, RTE)
- HYPOTHESIS TO TEST: LoRA achieves ≤3% accuracy drop AND (≥20% less drift OR ≤30% inference overhead) across task types
- DELIVERABLE: Publication-quality statistical analysis in analysis/statistical_analysis.py

ANALYSIS COMPONENTS:
1. PERFORMANCE COMPARISON:
   - Aggregate results across all seeds and tasks
   - Calculate mean performance gaps with confidence intervals
   - Test hypothesis: LoRA accuracy ≥ 97% of full fine-tuning
   ```python
   # Performance gap analysis
   performance_gaps = []
   for task in ['mrpc', 'squad_v2', 'sst2', 'rte']:
       for seed in [42, 1337, 2024]:
           # Use appropriate metric per task
           metric = 'accuracy' if task != 'squad_v2' else 'f1'
           full_acc = results[task][seed]['full_finetune'][metric]
           lora_acc = results[task][seed]['lora'][metric]
           gap = (full_acc - lora_acc) / full_acc * 100
           performance_gaps.append({'task': task, 'gap': gap})
   
   # Statistical test
   mean_gap = np.mean(performance_gaps)
   ci_lower, ci_upper = bootstrap_ci(performance_gaps, n_bootstrap=10000)
   p_value = permutation_test(performance_gaps, null_hypothesis=3.0)
   ```

2. DRIFT ANALYSIS SYNTHESIS:
   - Compare average drift reduction across layers
   - Test hypothesis: ≥20% less representational drift
   - Analyze layer-wise patterns and their significance
   - Correlation analysis: drift vs performance

3. DEPLOYMENT OVERHEAD ANALYSIS:
   - Calculate overhead percentages for multi-adapter scenarios
   - Test hypothesis: ≤30% overhead for 2-adapter deployment
   - Create predictive model for N-adapter overhead
   - Break-even analysis: when to merge vs multi-adapter

4. COMPREHENSIVE STATISTICAL TESTS:
   - Two-way ANOVA: method (full vs LoRA) × task (MRPC vs SQuAD)
   - Post-hoc tests with Bonferroni correction
   - Effect size calculations (Cohen's d, partial eta-squared)
   - Power analysis for future studies

5. RESULTS AGGREGATION:
   ```python
   # Master results table
   results_table = pd.DataFrame({
       'Method': ['Full FT', 'LoRA-8', 'Baseline'],
       'MRPC_Acc': [mean_ci(mrpc_full), mean_ci(mrpc_lora), mean_ci(mrpc_base)],
       'SST2_Acc': [mean_ci(sst2_full), mean_ci(sst2_lora), mean_ci(sst2_base)],
       'RTE_Acc': [mean_ci(rte_full), mean_ci(rte_lora), mean_ci(rte_base)],
       'SQuAD_F1': [mean_ci(squad_full), mean_ci(squad_lora), mean_ci(squad_base)],
       'Avg_Drift': [mean_ci(drift_full), mean_ci(drift_lora), 0],
       'Deploy_Overhead': ['0%', mean_ci(lora_overhead), 'N/A'],
       'Parameters': ['100%', '0.3%', '0%']
   })
   ```

6. HYPOTHESIS TESTING SUMMARY:
   - Primary hypothesis: Accepted/Rejected with p-values
   - Secondary analyses: Unexpected findings
   - Practical significance: Real-world implications
   - Limitations and future work

VISUALIZATION REQUIREMENTS:
- Performance comparison with error bars
- Drift evolution plots with confidence bands  
- Deployment overhead scaling curves
- Correlation matrices for all metrics
- Publication-ready LaTeX tables

Generate analysis/final_analysis_report.json with all statistical tests, p-values, effect sizes, and conclusions.

VALIDATION REQUIREMENT:
Before completing this step, run a short demo to ensure everything works:
1. Load results from W&B for one task and compute performance gap
2. Test hypothesis testing functions with dummy data
3. Generate a sample results table and visualization
4. Verify statistical test p-values are computed correctly
5. Check that final report contains all required sections and metrics
```

### Step 7 Validation Instructions

**How to validate statistical analysis is working correctly**:

1. **Data Aggregation Check**:
   ```bash
   # Verify all experimental results can be loaded from W&B
   # Check that data aggregation across seeds works correctly
   # Validate that missing data is handled appropriately
   ```

2. **Hypothesis Testing Validation**:
   - **Performance Gap**: Mean gap should be ≤3% if hypothesis holds
   - **Statistical Significance**: P-values should be meaningful and interpretable
   - **Effect Sizes**: Cohen's d values should indicate practical significance
   - **Multiple Comparisons**: Bonferroni correction applied correctly

3. **Results Table Validation**:
   - All tasks and methods included in master results table
   - Confidence intervals computed correctly for all metrics
   - Drift reduction percentages calculated properly
   - Deployment overhead values match benchmarking results

4. **Final Report Check**:
   - **Hypothesis Conclusion**: Clear accept/reject decision with supporting evidence
   - **Cross-Task Analysis**: Patterns consistent across different task types
   - **Practical Implications**: Real-world deployment recommendations included
   - **Limitations**: Honest assessment of study limitations

**Red Flags to Watch For**:
- Statistical tests producing unreasonable p-values (all 0 or all 1)
- Confidence intervals that are impossibly narrow or wide
- Inconsistent results across different tasks without explanation
- Missing data that breaks the analysis pipeline
- Conclusions that don't match the actual statistical results

### Statistical Software Stack

```python
# Required packages for analysis
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestPower
import matplotlib.pyplot as plt
import seaborn as sns

# Custom statistical functions
def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence intervals"""
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped.append(np.mean(sample))
    return np.percentile(bootstrapped, [alpha/2*100, (1-alpha/2)*100])

def permutation_test(data1, data2, n_permutations=10000):
    """Two-sample permutation test"""
    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    
    permuted_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_1 = combined[:len(data1)]
        perm_2 = combined[len(data1):]
        permuted_diffs.append(np.mean(perm_1) - np.mean(perm_2))
    
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    return p_value
```

## Reproducibility Checklist

### Environment Specification

```yaml
# environment.yaml
system:
  os: "Ubuntu 20.04"
  cuda: "11.8"
  python: "3.9.16"

dependencies:
  pytorch: "2.1.0"
  transformers: "4.35.0"
  peft: "0.6.0"
  datasets: "2.14.0"
  wandb: "0.15.12"
  vllm: "0.2.0"
  numpy: "1.24.3"
  scikit-learn: "1.3.0"
  
hardware:
  gpu: "NVIDIA A100 40GB" # or specify actual GPU
  cpu: "AMD EPYC 7742"
  ram: "128GB"
```

### Validation Procedures

1. **Code Verification**:
   - All scripts pass linting (flake8, black)
   - Unit tests for critical functions
   - Integration tests for full pipeline

2. **Data Integrity**:
   - MD5 checksums for all datasets
   - Verify train/val/test splits identical across runs
   - Check for data leakage

3. **Result Reproducibility**:
   - Fixed seeds produce <1% variation
   - Checkpoint loading verified
   - Results match across different machines

4. **Documentation**:
   - README with step-by-step execution
   - Config files for all experiments
   - Troubleshooting guide

### Final Validation Steps

```bash
# Run full reproducibility test
python validate_reproducibility.py --full-test

# Expected output:
# Environment setup identical
# Data checksums match
# Baseline results within 1% tolerance  
# Training curves overlap (KS test p>0.95)
# Final metrics match (within numerical precision)
# All statistical tests reproducible
```

## Summary

This implementation plan provides a comprehensive roadmap for rigorously comparing LoRA and full fine-tuning across multiple dimensions:

1. **Scientific Rigor**: Every experiment includes proper baselines, multiple seeds, and statistical validation
2. **Memory Optimization**: Efficient resource usage with 2GB training memory vs 26GB theoretical
3. **Parallel Execution**: True 3-VM parallelization with no dependencies within phases
4. **Developer-Friendly**: Simple script-based execution with tmux coordination
5. **Reproducibility**: Detailed specifications ensure experiments can be replicated
6. **Comprehensive Analysis**: Beyond accuracy, we analyze representations and deployment characteristics

### Key Execution Strategy

**True Parallel Execution**:
- **Phase 1 (Training)**: All 3 VMs start immediately with no dependencies
  - VM1: SQuAD v2 full fine-tuning + MRPC full fine-tuning + MRPC LoRA
  - VM2: SQuAD v2 LoRA + SST-2 full fine-tuning + SST-2 LoRA
  - VM3: RTE full fine-tuning + RTE LoRA + baseline experiments + **base model representation extraction**
- **Phase 2a (Parallel Analysis)**: All 3 VMs start immediately after Phase 1
  - VM1: MRPC + RTE drift analysis + correlation analysis prep
  - VM2: SQuAD v2 drift analysis + deployment benchmarking  
  - VM3: SST-2 drift analysis + visualization preparation
- **Phase 2b (Final Synthesis)**: Single VM after Phase 2a (cost optimization)
  - VM1: Statistical analysis, hypothesis testing, and report generation
  - VM2 & VM3: Idle (can be shut down)

**Efficiency Benefits**:
- **No Idle Time in Each Phase**: All VMs working simultaneously within each phase
- **True Parallelization**: No dependencies between VMs within Phase 1 and Phase 2a
- **Task-Based Splitting**: Similar to distributed pre-training approach from GNN course  
- **Cost Optimization**: Phase 2b uses only 1 VM, allowing 2 VMs to be shut down
- **Resource Optimization**: Balanced computational load across active VMs
- **Clear Dependencies**: Simple three-phase approach with minimal coordination overhead

The plan balances ambitious research goals with practical implementation constraints, ensuring high-quality scientific output suitable for publication while maximizing compute utilization.
