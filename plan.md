# LoRA Research Implementation Plan

## Research Questions & Significance

This research project investigates two critical questions in parameter-efficient fine-tuning that directly impact production deployment decisions:

1. **Representational Drift Analysis**: Does LoRA truly preserve model internal representations better than full fine-tuning? We will quantify this using centered-kernel alignment (CKA) and layer-wise cosine similarity metrics across all transformer layers.

2. **Deployment Efficiency Trade-offs**: What is the real-world latency penalty when deploying multiple LoRA adapters side-by-side versus merging them in vLLM? This addresses a key production concern for multi-task systems.

**Scientific Significance**: These questions address fundamental gaps in our understanding of efficient fine-tuning methods. While LoRA has gained widespread adoption, rigorous empirical analysis of its representation preservation claims and deployment overhead remains limited. Our findings will inform:
- Continual learning strategies to mitigate catastrophic forgetting
- Production system architecture decisions for multi-task deployments
- Theoretical understanding of low-rank adaptation's impact on model internals

**Hypotheses**: We hypothesize that LoRA (rank 8) will achieve ≤3% accuracy drop compared to full fine-tuning AND either ≥20% less representational drift OR ≤30% inference overhead. Both confirming and refuting these hypotheses constitute valid scientific contributions.

## 3-VM Resource Allocation Strategy

**Optimal Distribution Philosophy**: Maximize parallel utilization while respecting experimental dependencies and ensuring reproducibility.

**VM Allocation**:
- **VM1 (Baseline & Full Fine-tuning)**: Handles all baseline experiments (naive classifiers, SOTA baselines) and full fine-tuning runs. This VM requires the most memory for full model updates.
- **VM2 (LoRA Experiments)**: Dedicated to all LoRA training experiments across different tasks and seeds. Memory-efficient, allowing more parallel runs.
- **VM3 (Analysis & Deployment)**: Performs representational drift analysis, deployment benchmarking, and statistical analysis. Requires different software stack (vLLM).

**Justification**: This allocation ensures:
1. No resource contention between full and LoRA fine-tuning
2. Specialized software environments (VM3 needs vLLM, others need training stack)
3. Balanced computational load across VMs
4. Clear separation of concerns for debugging and monitoring

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
├── vm1_baseline_full/
│   ├── run_baselines.py     # Naive/SOTA baseline runners
│   └── run_full_finetune.py # Full fine-tuning experiments
├── vm2_lora/
│   └── run_lora.py          # LoRA training experiments
├── vm3_analysis/
│   ├── drift_analysis.py    # CKA and cosine similarity
│   └── deployment_bench.py  # vLLM benchmarking
└── orchestrator.py          # Main coordination script
```

**Shared Configuration Management**:
```python
# shared/config.yaml
experiment:
  model_name: "meta-llama/Llama-2-1.3b-hf"
  tasks: ["mrpc", "squad_v2"]
  seeds: [42, 1337, 2024]
  
wandb:
  project: "NLP"
  entity: "galavny-tel-aviv-university"
  
vm_roles:
  vm1: ["baselines", "full_finetune"]
  vm2: ["lora_rank8"]
  vm3: ["drift_analysis", "deployment_bench"]
```

### Technical/Operational Perspective

**Initial Setup (All VMs)**:
```bash
# Clone repository on each VM
git clone <repo_url> ~/NLP
cd ~/NLP

# Create shared results directory
mkdir -p /shared/nlp_results  # NFS mount or similar

# Set environment variables
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university
export HF_HOME=/shared/hf_cache  # Shared model cache
```

**Synchronization Protocol**:
1. **Git-based Coordination**: All VMs pull latest code before experiments
2. **W&B Run Groups**: Use `group` parameter to link related experiments
3. **File-based Checkpoints**: Save to shared NFS for cross-VM access
4. **Slack/Discord Webhooks**: Real-time notifications on experiment completion

**Execution Commands**:
```bash
# VM1: Launch baseline experiments
python orchestrator.py --vm-role vm1 --mode baselines

# VM2: Launch LoRA experiments (waits for baseline completion)
python orchestrator.py --vm-role vm2 --mode lora --wait-for baseline_complete

# VM3: Launch analysis (waits for training completion)
python orchestrator.py --vm-role vm3 --mode analysis --wait-for training_complete
```

## Step 1: Environment Setup & Sanity Checks

### Agent Prompt

```
You are setting up the experimental environment for a rigorous NLP research project comparing LoRA and full fine-tuning on Llama-2-1.3B. Your primary goal is to ensure complete reproducibility and validate the experimental setup.

CONTEXT:
- Model: meta-llama/Llama-2-1.3b-hf (chosen for computational feasibility)
- Tasks: GLUE-MRPC (sentence-pair classification) and SQuAD v2 (question-answering)
- Infrastructure: 3 GPU VMs with shared NFS storage
- Tracking: Weights & Biases (project: "NLP", entity: "galavny-tel-aviv-university")

REQUIREMENTS:
1. Create a comprehensive environment setup script that installs all dependencies with exact versions
2. Implement data downloading and preprocessing pipelines for MRPC and SQuAD v2
3. Create sanity check scripts that verify:
   - Model can overfit 10 examples from each dataset
   - Gradient flow is correct for both full fine-tuning and LoRA
   - W&B logging works correctly
   - Reproducibility across multiple runs with same seed

DELIVERABLES:
1. requirements.txt with pinned versions for all packages
2. setup_environment.sh script for reproducible environment creation
3. data_preparation.py for downloading and preprocessing datasets
4. sanity_checks.py that runs all validation tests
5. Basic experiment configuration in shared/config.yaml

TECHNICAL SPECIFICATIONS:
- PyTorch 2.1.0 with CUDA 11.8
- transformers 4.35.0
- peft 0.6.0 for LoRA implementation
- datasets library for data handling
- wandb for experiment tracking
- vLLM 0.2.0 (only on VM3)

Include detailed logging and error handling. The setup must be idempotent and handle partial failures gracefully.
```

### 3-VM Distribution

**Setup Task Distribution**:
- **All VMs**: Run base environment setup (Python, CUDA, core packages)
- **VM1 & VM2**: Install training stack (transformers, peft, datasets)
- **VM3**: Additionally install vLLM and analysis packages (scikit-learn for CKA)

**Rationale**: Common base ensures consistency while specialized packages reduce overhead. Shared model cache prevents redundant downloads.

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
- Tasks: MRPC (accuracy, F1) and SQuAD v2 (EM, F1)
- Your role: Establish strong baselines for meaningful comparison

REQUIRED BASELINES (all are mandatory):
1. MAJORITY CLASS CLASSIFIER:
   - For MRPC: Always predict most frequent label in training set
   - For SQuAD: Always predict "no answer" for unanswerable questions
   - Report accuracy, F1, and confidence intervals

2. RANDOM BASELINE:
   - For MRPC: Random predictions with class distribution matching training set
   - For SQuAD: Random span selection for answerable, "no answer" for others
   - Run with 5 different seeds, report mean and std

3. ZERO-SHOT LLAMA-2:
   - Test Llama-2-1.3b without any fine-tuning
   - Use carefully crafted prompts for each task
   - Try 3 different prompt templates, report best

4. SOTA BASELINE FROM LITERATURE:
   - For MRPC: Implement RoBERTa-base fine-tuning (Liu et al., 2019)
   - For SQuAD v2: Implement ALBERT-base fine-tuning (Lan et al., 2019)
   - Use published hyperparameters, verify you match reported scores (±2%)

IMPLEMENTATION REQUIREMENTS:
- Create separate scripts for each baseline type
- Use consistent evaluation code across all baselines
- Log all results to W&B with clear naming convention
- Implement proper cross-validation for robust estimates
- Calculate statistical significance between baselines

STATISTICAL RIGOR:
- Run each baseline with at least 3 random seeds
- Report mean, std, and 95% confidence intervals
- Use bootstrap sampling (n=1000) for confidence intervals
- Implement McNemar's test for pairwise comparisons

Output comprehensive baseline_results.json with all metrics and statistical tests.
```

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
   - Majority class: ~50% for balanced MRPC
   - Random: Expected 50% for MRPC, ~0% EM for SQuAD

2. **Zero-shot LLaMA-2** (test pre-training knowledge):
   - Expected: 60-70% MRPC, 20-30% SQuAD EM

3. **SOTA Fine-tuned** (establish ceiling):
   - RoBERTa-base on MRPC: ~90% accuracy
   - ALBERT-base on SQuAD v2: ~80% F1

## Step 3: Full Fine-tuning Experiments

### Agent Prompt

```
You are implementing full fine-tuning experiments for Llama-2-1.3B on MRPC and SQuAD v2. This serves as the primary comparison point for LoRA experiments.

CONTEXT:
- Baselines established: Majority class, random, zero-shot, and SOTA
- Model: meta-llama/Llama-2-1.3b-hf
- Hardware: Full access to VM1 with gradient checkpointing if needed
- Objective: Establish full fine-tuning performance and representation changes

EXPERIMENTAL DESIGN:
1. HYPERPARAMETER SEARCH:
   - Learning rates: [1e-5, 2e-5, 5e-5] 
   - Batch sizes: [8, 16] with gradient accumulation as needed
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
- Implement training script with full configurability
- Use HuggingFace Trainer with custom callbacks
- Implement custom representation extraction callback
- Create gradient statistics monitoring
- Ensure deterministic training with fixed seeds

CRITICAL REQUIREMENTS:
- Run each configuration with seeds [42, 1337, 2024]
- Save model checkpoints for best validation performance
- Log training dynamics (loss, gradients, learning rate)
- Extract representations for drift analysis (every 100 steps)
- Profile memory usage and training time

Output all results to W&B and save checkpoints to shared storage for VM3 analysis.
```

### 3-VM Distribution

**VM1 Exclusive Tasks**:
- All full fine-tuning runs (high memory requirement)
- Hyperparameter sweeps using W&B agents
- Checkpoint saving to shared NFS

**Parallel Execution Strategy**:
- Run MRPC and SQuAD experiments in parallel (different GPUs if multi-GPU)
- Sequential seeds to ensure reproducibility
- Automatic job queuing for failed runs

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
- Full fine-tuning complete with performance benchmarks established
- Target: ≤3% accuracy drop compared to full fine-tuning
- LoRA rank: 8 (fixed as per research protocol)
- Hardware: VM2 dedicated to LoRA experiments

LORA CONFIGURATION:
1. ARCHITECTURE SETTINGS:
   - Rank (r): 8
   - Alpha: 16 (scaling factor = alpha/r = 2)
   - Target modules: ["q_proj", "v_proj"] (query and value projections)
   - Dropout: 0.1 for regularization
   - Initialize with Kaiming uniform

2. HYPERPARAMETER SEARCH:
   - Learning rates: [1e-4, 3e-4, 5e-4] (typically higher than full fine-tuning)
   - LoRA-specific warmup: 6% of total steps
   - Use same batch sizes as full fine-tuning for fair comparison
   - Early stopping with same criteria

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
- Use PEFT library for standardized LoRA implementation
- Implement custom metrics for parameter efficiency:
  - Trainable parameters: ~0.3% of full model
  - Memory usage during training
  - Actual vs theoretical speedup
- Create LoRA merge testing:
  - Test merged model equivalence
  - Benchmark merged vs adapter inference
- Implement ablation studies:
  - Different rank values [4, 8, 16] for comparison
  - Impact of alpha scaling
  - Module selection impact

CRITICAL VALIDATION:
- Verify LoRA updates don't affect base model weights
- Ensure reproducibility across seeds [42, 1337, 2024]
- Compare convergence speed vs full fine-tuning
- Validate that merged model produces identical outputs

Save all LoRA adapters and base model for deployment testing on VM3.
```

### 3-VM Distribution

**VM2 Exclusive Tasks**:
- All LoRA training experiments
- Ablation studies on rank and module selection
- Adapter weight analysis

**Efficiency Optimizations**:
- Share base model across all LoRA runs (read-only)
- Parallel training of different configurations
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
- Completed: Full fine-tuning and LoRA training on both tasks
- Saved: Layer-wise representations every 100 training steps
- Objective: Quantify if LoRA preserves representations better (≥20% less drift)

ANALYSIS METHODS:
1. CENTERED KERNEL ALIGNMENT (CKA):
   - Implementation: Use linear CKA (more stable than RBF)
   - Compute between base model and fine-tuned at each layer
   - Create layer × training_step heatmaps
   - Statistical test: Permutation test for significance

2. LAYER-WISE COSINE SIMILARITY:
   - Average cosine similarity per layer
   - Track evolution over training steps
   - Identify which layers change most
   - Compare drift patterns between methods

3. REPRESENTATION ANALYSIS PROTOCOL:
   - Use 1000 fixed validation examples (same across all runs)
   - Extract representations from all transformer layers
   - Include attention patterns and MLP activations
   - Analyze both token-level and sequence-level representations

4. STATISTICAL ANALYSIS:
   - Compute drift metrics for each seed separately
   - Report mean ± std across seeds
   - Use permutation tests (n=10000) for p-values
   - Calculate effect sizes (Cohen's d)

IMPLEMENTATION DETAILS:
```python
# CKA Implementation
def linear_cka(X, Y):
    # Center the matrices
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    
    # Compute CKA
    YTX = Y.T @ X
    norm = np.sqrt(np.trace(X.T @ X) * np.trace(Y.T @ Y))
    return np.trace(YTX @ YTX.T) / (norm ** 2)

# Analysis per layer
for layer in range(num_layers):
    base_repr = load_base_representations(layer)
    ft_repr = load_finetuned_representations(layer)
    lora_repr = load_lora_representations(layer)
    
    # Compute drift
    ft_drift = 1 - linear_cka(base_repr, ft_repr)
    lora_drift = 1 - linear_cka(base_repr, lora_repr)
    
    # Statistical test
    drift_reduction = (ft_drift - lora_drift) / ft_drift * 100
```

VISUALIZATION REQUIREMENTS:
- Create drift evolution plots over training steps
- Generate layer-wise drift comparison heatmaps
- Plot confidence intervals for all metrics
- Create publication-quality figures

CRITICAL ANALYSES:
- Early vs late layer drift patterns
- Task-specific drift differences (MRPC vs SQuAD)
- Correlation between drift and performance
- Identify "critical" layers with highest drift

Output comprehensive drift_analysis_results.json and all visualization figures.
```

### 3-VM Distribution

**VM3 Tasks**:
- Load saved representations from shared storage
- Compute all drift metrics
- Generate statistical analyses and visualizations
- No GPU needed (CPU-only analysis)

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
- Completed: All training experiments with saved models
- Infrastructure: VM3 with vLLM 0.2.0 installed
- Objective: Quantify deployment overhead (target: ≤30% for multi-adapter)

BENCHMARKING SCENARIOS:
1. BASELINE MEASUREMENTS:
   - Original Llama-2-1.3B (no modifications)
   - Fully fine-tuned models (MRPC and SQuAD)
   - Single merged LoRA models

2. MULTI-ADAPTER DEPLOYMENT:
   - 2 adapters: MRPC + SQuAD simultaneously
   - 4 adapters: Add synthetic task adapters
   - 8 adapters: Stress test scenario
   - Dynamic adapter switching overhead

3. METRICS TO COLLECT:
   - Throughput: Tokens/second at various batch sizes [1,2,4,8,16]
   - Latency: 50th, 95th, 99th percentile
   - Memory usage: Base + per-adapter overhead
   - First token latency (critical for interactive use)

4. IMPLEMENTATION PROTOCOL:
```python
import vllm
from vllm import LLM, SamplingParams

# Test configurations
configs = {
    "baseline": LLM(model="meta-llama/Llama-2-1.3b-hf"),
    "merged_mrpc": LLM(model="./merged_models/llama2_mrpc"),
    "multi_adapter": LLM(
        model="meta-llama/Llama-2-1.3b-hf",
        enable_lora=True,
        max_lora_rank=8,
        lora_modules=["./adapters/mrpc", "./adapters/squad"]
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
```

### 3-VM Distribution

**VM3 Exclusive**:
- All vLLM benchmarking
- Requires different CUDA setup than training
- Full GPU dedication for accurate measurements

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
- All experiments complete with raw results in W&B
- Hypothesis: LoRA achieves ≤3% accuracy drop AND (≥20% less drift OR ≤30% inference overhead)
- Need publication-quality statistical analysis

ANALYSIS COMPONENTS:
1. PERFORMANCE COMPARISON:
   - Aggregate results across all seeds and tasks
   - Calculate mean performance gaps with confidence intervals
   - Test hypothesis: LoRA accuracy ≥ 97% of full fine-tuning
   ```python
   # Performance gap analysis
   performance_gaps = []
   for task in ['mrpc', 'squad']:
       for seed in [42, 1337, 2024]:
           full_acc = results[task][seed]['full_finetune']['accuracy']
           lora_acc = results[task][seed]['lora']['accuracy']
           gap = (full_acc - lora_acc) / full_acc * 100
           performance_gaps.append(gap)
   
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

Generate final_analysis_report.json with all statistical tests, p-values, effect sizes, and conclusions.
```

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
# ✓ Environment setup identical
# ✓ Data checksums match
# ✓ Baseline results within 1% tolerance  
# ✓ Training curves overlap (KS test p>0.95)
# ✓ Final metrics match (within numerical precision)
# ✓ All statistical tests reproducible
```

## Summary

This implementation plan provides a comprehensive roadmap for rigorously comparing LoRA and full fine-tuning across multiple dimensions:

1. **Scientific Rigor**: Every experiment includes proper baselines, multiple seeds, and statistical validation
2. **3-VM Optimization**: Tasks distributed based on computational requirements and dependencies
3. **Reproducibility**: Detailed specifications ensure experiments can be replicated
4. **Comprehensive Analysis**: Beyond accuracy, we analyze representations and deployment characteristics

The plan balances ambitious research goals with practical implementation constraints, ensuring high-quality scientific output suitable for publication.
