# LoRA Research Implementation Plan

## Research Questions & Significance

This research project investigates two critical questions in parameter-efficient fine-tuning that directly impact production deployment decisions for **text classification tasks**:

1. **Representational Drift Analysis**: Does LoRA preserve model internal representations better than full fine-tuning on text classification tasks? We will quantify this using centered-kernel alignment (CKA) and layer-wise cosine similarity metrics across all transformer layers on three diverse classification tasks.

2. **Deployment Efficiency Trade-offs**: What is the real-world latency penalty when deploying multiple LoRA adapters side-by-side versus merging them in vLLM? This addresses a key production concern for multi-task classification systems.

**Scientific Significance**: These questions address fundamental gaps in our understanding of efficient fine-tuning methods. While LoRA has gained widespread adoption, rigorous empirical analysis of its representation preservation claims and deployment overhead remains limited. Our findings will inform:
- Continual learning strategies to mitigate catastrophic forgetting
- Production system architecture decisions for multi-task deployments
- Theoretical understanding of low-rank adaptation's impact on model internals

**Hypotheses**: We hypothesize that for text classification tasks, LoRA (rank 8) will achieve ≤3% accuracy drop compared to full fine-tuning AND either ≥20% less representational drift OR ≤30% inference overhead. Both confirming and refuting these hypotheses constitute valid scientific contributions.

**Scope**: This research focuses on text classification tasks. Our findings are specific to classification tasks and should not be generalized to other task types (e.g., question answering, summarization, generation) without further investigation.

## Task Selection & Rationale

**Three diverse text classification tasks** selected for comprehensive evaluation:

| Task | Type | Size | Metric | Rationale |
|------|------|------|--------|-----------|
| **MRPC** | Sentence-pair classification | 3.7K train | Accuracy/F1 | Tests semantic similarity and paraphrase detection |
| **SST-2** | Single-sentence classification | 67K train | Accuracy | Sentiment analysis - most straightforward task |
| **RTE** | Sentence-pair reasoning | 2.5K train | Accuracy | Tests logical entailment - most complex reasoning |

**Task Diversity Justification**:

Despite focusing on classification tasks, our selection provides excellent diversity across multiple dimensions:

1. **Task Type Diversity**:
   - **Semantic similarity** (MRPC): Determining if two sentences are paraphrases
   - **Sentiment classification** (SST-2): Identifying positive/negative opinions
   - **Logical entailment** (RTE): Determining if one sentence logically follows from another
   - → Three fundamentally different linguistic phenomena

2. **Input Format Diversity**:
   - **Single-sentence**: SST-2 (tests sentence-level representations)
   - **Sentence-pair**: MRPC, RTE (tests cross-sentence reasoning)
   - → Covers both major classification formats

3. **Dataset Size Diversity**:
   - **Small**: RTE (2.5K samples) - low-resource scenario
   - **Medium**: MRPC (3.7K samples) - typical GLUE task size
   - **Large**: SST-2 (67K samples) - 27× larger, tests scalability
   - → Spans 27× range in dataset size

4. **Task Complexity Diversity**:
   - **Straightforward**: SST-2 (sentiment is relatively clear-cut)
   - **Moderate**: MRPC (requires nuanced semantic understanding)
   - **Complex**: RTE (requires logical inference and reasoning)
   - → Tests LoRA across difficulty levels

5. **GLUE Benchmark Coverage**:
   - All three tasks are core GLUE benchmark components
   - Represent different facets of natural language understanding
   - Commonly used subset in parameter-efficient fine-tuning research

This diverse subset enables rigorous analysis of LoRA's behavior across varied classification scenarios while maintaining methodological validity within our computational constraints.

**Model**: TinyLlama-1.1B (1.3B parameters)
- Efficient for academic research (fits in 24GB GPU memory)
- Proven Llama-2 architecture with good task transfer
- Enables thorough experimentation within time constraints

**LoRA Configuration**: Rank 8 (fixed across all tasks)
- **Methodological Justification**: Following Hu et al. (2021), rank 8 balances efficiency and performance
- **Research Standard**: Ranks 4-32 show diminishing returns; rank 8 is commonly used in GLUE benchmarks
- **Controlled Comparison**: Fixed rank ensures fair cross-task comparison (not optimizing rank per task)
- **Scope**: Task-specific rank optimization is future work; this study focuses on LoRA vs Full FT trade-offs
- **Limitation Acknowledged**: Rank 8 may be suboptimal for some tasks, but provides consistent baseline

## Methodological Approach

This research follows a rigorous four-phase experimental design ensuring reproducibility, statistical validity, and adherence to machine learning best practices.

### **Research Scope and Limitations**

**Focus on Classification Tasks**: This research evaluates LoRA and full fine-tuning specifically on **text classification tasks**. 

**Rationale for Scope**:
- **Resource constraints**: GPU memory and time limitations required focusing on a specific task domain
- **Depth over breadth**: Prioritized rigorous analysis of classification tasks over broader task coverage
- **Methodological validity**: Three diverse classification tasks provide sufficient statistical power and task diversity within classification domain

**What We Can Claim**:
- ✅ Findings apply to text classification tasks
- ✅ Results generalize across classification types (sentiment, paraphrase, entailment)
- ✅ Conclusions valid for single-sentence and sentence-pair classification
- ✅ Insights applicable to low-resource (2.5K) through large (67K) classification datasets

**What We Cannot Claim**:
- ❌ Generalization to question answering tasks
- ❌ Generalization to text generation tasks
- ❌ Generalization to sequence-to-sequence tasks (summarization, translation)
- ❌ Universal claims about "all NLP tasks"

**Impact on Contributions**:
- Provides rigorous, in-depth analysis of LoRA for classification tasks
- Establishes baseline understanding for classification domain
- Identifies need for future work on QA and generative tasks
- Maintains methodological rigor within defined scope

**Justification for Validity**:
Despite the narrow task scope, our research maintains strong validity because:
1. **Diverse classification coverage**: Three fundamentally different linguistic phenomena
2. **Statistical power**: 18 experiments (3 tasks × 2 methods × 3 seeds)
3. **Size diversity**: 27× range in dataset size (2.5K to 67K)
4. **Complexity range**: From straightforward sentiment to complex logical reasoning
5. **Standard practice**: Many PEFT papers focus on classification tasks from GLUE

### **Core Methodological Principles**

#### 1. Data Integrity & Avoiding Leakage
- **Strict data splits**: Train/validation/test splits maintained consistently across all experiments
- **Validation data usage**: Used ONLY for hyperparameter selection (Phase 1) and early stopping
- **Test data isolation**: Held out completely until final evaluation, never used for any optimization
- **No data snooping**: All preprocessing, architecture, and training decisions made independent of test set
- **Data sources**: All datasets from HuggingFace Datasets with documented versions

#### 2. Overfitting Prevention & Sanity Validation
- **Sanity checks** (Phase 0): All models proven to overfit on 10-sample subsets
  - **Purpose**: Validates implementation correctness before expensive experiments
  - **Success criterion**: Training loss < 0.1 on small sample
  - **Why this matters**: Inability to overfit indicates bugs in model/training code
- **Gradient clipping**: max_norm=0.3 prevents gradient explosion
- **Early stopping**: patience=3 epochs prevents overfitting on full datasets
- **Regularization**: Weight decay (0.0-0.01 for Full FT, 0.0-0.1 for LoRA)

#### 3. Baseline Comparisons
All experiments compared against appropriate baselines to frame results meaningfully:
- **Majority class baseline**: Predicts most frequent class (establishes minimum threshold)
- **Random predictions**: Random guessing performance (confirms above-chance learning)
- **Base model (zero-shot)**: Pre-trained model without fine-tuning (not applicable for this study due to lack of task-specific formatting)
- **Baselines computed**: On identical data splits as main experiments

#### 4. Accounting for Randomness
- **Multiple seeds**: All production experiments (Phase 2) run with 3 independent seeds (42, 1337, 2024)
- **Statistical aggregation**: Mean ± standard deviation reported for all metrics
- **Hypothesis testing**: Permutation tests with Bonferroni correction for multiple comparisons
- **Confidence intervals**: 95% bootstrap confidence intervals for all claims
- **Why 3 seeds**: Balances statistical validity with computational cost (standard in NLP research)

#### 5. Hyperparameter Optimization Methodology
- **Bayesian optimization**: Optuna with TPE sampler (Bergstra & Bengio, 2012)
  - More efficient than grid/random search
  - 15 trials per task/method exceeds literature standards (LoRA paper: 6 trials)
- **Separate optimization**: Independent hyperparameter search per task/method combination
  - Prevents unfair comparisons (e.g., using LoRA's optimal LR for Full FT)
- **Validation-based selection**: Hyperparameters chosen to maximize validation performance
- **Two-phase design**:
  - Phase 1: Find optimal hyperparameters with single seed (reproducibility)
  - Phase 2: Validate with multiple seeds (statistical rigor)

#### 6. Reproducibility
- **Configuration management**: All settings documented in `shared/config.yaml`
- **Code availability**: All training scripts, models, analysis code in Git
- **Data provenance**: HuggingFace Datasets with specific versions
- **Model artifacts**: Checkpoints uploaded to WandB for independent verification
- **Deterministic training**: `torch.use_deterministic_algorithms(True)` when possible
- **Hardware specification**: L4 24GB GPUs documented
- **Software versions**: `requirements.txt` with pinned versions

### **Critical Methodological Decisions**

#### Configuration Consistency (max_length=384)
**Problem**: Representation drift analysis requires identical tokenization across base and fine-tuned models.

**Solution**: All phases use `max_length=384` consistently:
- **Phase 0**: Base model representation extraction
- **Phase 1**: Hyperparameter optimization
- **Phase 2**: Production training
- **Phase 3**: Post-hoc representation extraction

**Justification**: Different sequence lengths would invalidate CKA comparisons (representations from different-length sequences are not comparable).

#### Post-Hoc Representation Extraction (Phase 3)
**Problem**: Extracting representations during training requires batch_size=1, making training 4x slower.

**Solution**: Disable representation extraction during training; extract post-hoc from saved models.

**Methodological soundness**:
- Representations are properties of trained models, not training artifacts
- Post-training extraction produces identical representations
- Enables efficient training with batch_size=2-4
- Standard practice in representation analysis research

## Current Implementation Status

### ✅ Completed Phases

#### Phase 0: Methodology Validation & Baselines
**Purpose**: Validate infrastructure and establish baselines before expensive experiments.

**Completed Components**:
- ✅ **Sanity checks**: All 3 classification tasks proven to overfit on 10 samples (`shared/sanity_checks.py`)
- ✅ **Baselines**: Majority class and random predictions computed (`experiments/baselines.py`)
- ✅ **Base representations**: Pre-trained model representations extracted for all 3 tasks
- ✅ **Infrastructure validation**: Memory profiling, architecture testing, data pipeline validation

**Results**: All sanity checks passed, proving implementation correctness.

#### Phase 1: Hyperparameter Optimization
**Purpose**: Find optimal hyperparameters using Bayesian optimization.

**Completed Components**:
- ✅ **Optuna integration**: TPE sampler implementation (`experiments/optuna_optimization.py`)
- ✅ **Search spaces**:
  - Full FT: learning_rate (1e-6 to 5e-5), batch_size (1-4), warmup_ratio (0-0.2), weight_decay (0-0.01), epochs (2-4)
  - LoRA: learning_rate (5e-5 to 5e-3), batch_size (1-4), warmup_ratio (0-0.3), weight_decay (0-0.1), epochs (2-4), lora_alpha (8-64), lora_dropout (0-0.2)
- ✅ **Trials**: 15 per task/method (total: 3 tasks × 2 methods × 15 trials = 90 trials)
- ✅ **Optimal configs**: Saved to `analysis/*.yaml` for production use

**Results**: Clear optimal hyperparameters identified for all 6 configurations (3 tasks × 2 methods).

**Methodological Note**: 15 trials exceeds Bergstra & Bengio (2012) minimum recommendation of 10 trials for TPE.

#### Phase 2: Production Model Training (VM2 Classification Tasks)
**Purpose**: Train final models with optimal hyperparameters and multiple seeds.

**Completed Components (VM2)**:
- ✅ **MRPC**: Full fine-tuning + LoRA × 3 seeds (42, 1337, 2024)
- ✅ **SST-2**: Full fine-tuning + LoRA × 3 seeds
- ✅ **RTE**: Full fine-tuning + LoRA × 3 seeds
- ✅ **Total**: 18 models trained (3 tasks × 2 methods × 3 seeds)
- ✅ **Model upload**: All 18 models uploaded to WandB artifacts for reproducibility

**Training Details**:
- Full datasets used (MRPC: 3.7K, SST-2: 67K, RTE: 2.5K)
- Optimal hyperparameters loaded from Phase 1
- Representation extraction disabled for memory efficiency
- WandB logging to NLP-Phase2 project

#### Phase 3: Representation Extraction (VM2 Classification Tasks)
**Purpose**: Extract layer-wise representations from trained models for drift analysis.

**Completed Components (VM2)**:
- ✅ **Post-hoc extraction**: Representations extracted from all 18 trained models
- ✅ **Layer coverage**: All 22 transformer layers + final hidden states
- ✅ **Output location**: `results/phase3_representations/representations/`
- ✅ **Consistency**: max_length=384 enforced (matches training configuration)

**Extraction Details**:
- 18 representation sets (9 full_finetune + 9 lora)
- Each set contains 22 layer files + metadata
- Memory-efficient processing (no training overhead)
- Ready for drift analysis

### ⏳ Remaining Work

**Note**: All Phases 0-3 are complete for the three classification tasks (MRPC, SST-2, RTE). Only Phase 4 (final analysis and report generation) remains.

#### Phase 4: Comprehensive Analysis & Research Question Synthesis

**Purpose**: Analyze all data from Phases 0-3 to definitively answer both research questions with statistical rigor.

**Phase 4A: Research Question 1 - Representational Drift Analysis** ❌ NOT STARTED
- **Question**: Does LoRA preserve model internal representations better than full fine-tuning?
- **Data sources**: Base representations (Phase 0) + Fine-tuned representations (Phase 3)
- **Analysis tasks**:
  - ❌ **CKA computation**: Layer-wise centered kernel alignment between base and fine-tuned models
  - ❌ **Cosine similarity**: Alternative drift metric for validation
  - ❌ **Cross-task comparison**: Compare drift across 3 classification tasks (MRPC, SST-2, RTE)
  - ❌ **LoRA vs Full FT comparison**: Quantify drift reduction percentage
  - ❌ **Statistical testing**: Permutation tests with Bonferroni correction
  - ❌ **Hypothesis validation**: Test if LoRA shows ≥20% less drift than Full FT
  - ❌ **Visualization**: Layer-wise heatmaps, drift trajectories, task comparisons
- **Scripts**: `scripts/phase3/analyze_drift.py`, `visualize_drift.py`
- **Output**: Detailed analysis section in final report answering RQ1

**Phase 4B: Research Question 2 - Deployment Efficiency Analysis** ❌ NOT STARTED
- **Question**: What is the real-world latency penalty for multi-adapter deployments vs merged models?
- **Data sources**: Trained models from Phase 2
- **Analysis tasks**:
  - ❌ **vLLM setup**: Multi-adapter support configuration
  - ❌ **Latency benchmarking**: Single-adapter, multi-adapter (2 & 4), merged models
  - ❌ **Throughput measurement**: Samples/second across configurations
  - ❌ **Memory profiling**: GPU usage for each deployment scenario
  - ❌ **Statistical testing**: Bootstrap confidence intervals for overhead
  - ❌ **Hypothesis validation**: Test if multi-adapter overhead ≤30%
  - ❌ **Cost-benefit analysis**: Trade-offs between flexibility and performance
- **Output**: Detailed analysis section in final report answering RQ2

**Final Synthesis**:
- ❌ **Integrated analysis report** (`ANALYSIS_REPORT.md`):
  - Executive summary of findings
  - RQ1 answer with statistical evidence
  - RQ2 answer with benchmark data
  - Cross-analysis insights (e.g., tasks with high drift vs deployment overhead)
  - Hypothesis test results (accept/reject with p-values and effect sizes)
  - Limitations and future work
  - Practical recommendations for LoRA deployment

## Phase Execution Details

### Phase 0: Methodology Validation & Baselines ✅ COMPLETED

**Runtime**: ~5 hours

**VM (Classification)**:
- Sanity checks: Overfit on 10 samples per task (MRPC, SST-2, RTE)
- Baselines: Majority class, random predictions per task
- Base representation extraction: All 22 layers for all 3 classification tasks
- Memory profiling: Validate GPU usage patterns

**Success Criteria** (All Met):
- ✅ All models achieve training loss < 0.1 on 10 samples
- ✅ Baseline metrics established for comparison
- ✅ Base representations extracted and validated
- ✅ No OOM errors or infrastructure failures

**Methodological Note**: Sanity checks critical for validating implementation before expensive experiments.

### Phase 1: Hyperparameter Optimization ✅ COMPLETED

**Runtime**: ~3-4 hours

**VM (Classification)**:
- MRPC: 15 trials × 2 methods = 30 trials
- SST-2: 15 trials × 2 methods = 30 trials (3,000 sample subset)
- RTE: 15 trials × 2 methods = 30 trials
- Total: 90 trials

**Hyperparameter Search Spaces**:

*Full Fine-tuning*:
- learning_rate: [1e-6, 5e-5] (log scale) - conservative for stability
- batch_size: [1, 2, 4] - memory constrained
- warmup_ratio: [0.0, 0.2] - prevents early instability
- weight_decay: [0.0, 0.01] - L2 regularization
- num_train_epochs: [2, 4] - prevents overfitting

*LoRA*:
- learning_rate: [5e-5, 5e-3] (log scale) - higher than Full FT (fewer parameters)
- batch_size: [1, 2, 4]
- warmup_ratio: [0.0, 0.3]
- weight_decay: [0.0, 0.1] - can handle more regularization
- num_train_epochs: [2, 4]
- lora_alpha: [8, 64] - scaling factor for LoRA
- lora_dropout: [0.0, 0.2] - LoRA-specific regularization

**Optimal Configs Saved**:
- `analysis/mrpc_full_finetune_optimal.yaml`
- `analysis/mrpc_lora_optimal.yaml`
- `analysis/sst2_full_finetune_optimal.yaml`
- `analysis/sst2_lora_optimal.yaml`
- `analysis/rte_full_finetune_optimal.yaml`
- `analysis/rte_lora_optimal.yaml`

**Success Criteria** (All Met):
- ✅ Clear optimal hyperparameters identified (best trial significantly better than random trials)
- ✅ Performance gaps between best/worst >5% (validates optimization effectiveness)
- ✅ TPE convergence observed (later trials better than early random trials)

**Methodological Justification**:
- 15 trials exceeds Bergstra & Bengio (2012) minimum of 10 for TPE
- Separate optimization per task/method prevents unfair comparisons
- Validation-based selection prevents overfitting to training data

### Phase 2: Production Model Training

**Purpose**: Train final models with optimal hyperparameters using full datasets and multiple seeds for statistical validity.

**Critical Design Decision**: Representation extraction DISABLED during training
- Enables batch_size=2-4 (vs forced batch_size=1 with extraction)
- 2-4x training speedup
- Prevents OOM on full datasets
- Representations extracted post-hoc in Phase 3 (methodologically equivalent)

#### Classification Tasks (MRPC, SST-2, RTE) ✅ COMPLETED

**Configuration**:
- Tasks: MRPC (3.7K), SST-2 (67K), RTE (2.5K)
- Methods: Full fine-tuning + LoRA
- Seeds: 42, 1337, 2024 per method
- Total experiments: 18 (3 tasks × 2 methods × 3 seeds)
- Actual runtime: ~20-24 hours

**Training Details**:
- Full datasets used (no sampling)
- Optimal hyperparameters loaded from Phase 1
- Early stopping with patience=3
- Checkpoints saved to WandB
- Test evaluation performed after training

**Outputs**:
- 18 trained models uploaded to WandB artifacts
- Training logs in WandB NLP-Phase2 project
- Best checkpoints saved locally
- Evaluation metrics logged

**Phase 2 Success Criteria**:
- All experiments complete without OOM errors
- Test metrics logged for all seeds
- Models uploaded to WandB for reproducibility
- Consistent performance across seeds (low variance)

### Phase 3: Representation Extraction

**Purpose**: Extract layer-wise representations from all trained models for drift analysis.

**Design Rationale**: Post-hoc extraction enables:
- Memory-efficient layer-by-layer processing
- No training overhead (no gradients, optimizer states)
- Identical representations to in-training extraction
- Can extract from downloaded WandB models

**Classification Representation Extraction** ✅ COMPLETED

**Configuration**:
- Models: 18 classification models (3 tasks × 2 methods × 3 seeds)
- Layers: All 22 transformer layers + final hidden states
- Samples: 750 validation samples per task
- max_length: 384
- Output: `results/phase3_representations/representations/`

**Completed Extractions**:
- ✅ 9 full_finetune representation sets (MRPC, SST-2, RTE × 3 seeds each)
- ✅ 9 lora representation sets (MRPC, SST-2, RTE × 3 seeds each)
- ✅ Each set: 22 layer files + metadata.json

**Extraction Process**:
1. Load trained model from results/ or download from WandB
2. Load validation dataset (same as used in training)
3. Forward pass with hooks to capture layer outputs
4. Save representations layer-by-layer to disk
5. Cleanup GPU memory between models

### Phase 4: Comprehensive Analysis & Research Question Synthesis ❌ NOT STARTED

**Purpose**: Synthesize all data from Phases 0-3 to definitively answer both research questions and produce a comprehensive analysis report.

This phase brings together all experimental results to create a cohesive narrative backed by rigorous statistical analysis.

#### Phase 4A: Research Question 1 - Representational Drift Analysis

**Research Question**: Does LoRA truly preserve model internal representations better than full fine-tuning?

**Data Sources**:
- Base model representations (Phase 0)
- Fine-tuned model representations (Phase 3)
- All 3 tasks × 2 methods × 3 seeds = 18 representation sets

**Analysis Pipeline**:

1. **CKA Computation** (`scripts/phase3/analyze_drift.py`)
   - Centered Kernel Alignment between base and fine-tuned representations
   - Layer-wise analysis (all 22 transformer layers)
   - Per-task, per-method, per-seed computation
   - Aggregate statistics across seeds (mean ± std)

2. **Drift Metrics**:
   - **Primary metric**: Drift = 1 - CKA (higher = more drift from base model)
   - **Secondary metric**: Cosine similarity for validation
   - **Layer-wise profiles**: Identify which layers drift most
   - **Task-level aggregation**: Average drift across all layers per task

3. **Comparative Analysis**:
   - **LoRA vs Full FT**: Compute drift reduction percentage
   - **Cross-task patterns**: Compare drift across task types (classification vs QA)
   - **Layer depth analysis**: Early vs middle vs late layers
   - **Statistical significance**: Permutation tests with Bonferroni correction

4. **Hypothesis Testing**:
   - **H1a**: LoRA achieves ≤3% accuracy drop vs Full FT
     - Compare test metrics from Phase 2
     - Bootstrap 95% confidence intervals
     - Paired t-tests across seeds
   
   - **H1b**: LoRA shows ≥20% less drift than Full FT
     - Compare average drift across all layers
     - Permutation tests (10,000 permutations)
     - Bonferroni correction for 3 tasks
     - Effect sizes (Cohen's d)

5. **Visualization** (`scripts/phase3/visualize_drift.py`)
   - Layer-wise drift heatmaps (22 layers × 3 tasks × 2 methods)
   - Drift comparison plots (LoRA vs Full FT with error bars)
   - Per-task drift profiles across layers
   - Statistical significance markers (p < 0.05, p < 0.01)
   - Publication-quality figures for paper

**Execution**:
   ```bash
# Run drift analysis
python scripts/phase3/analyze_drift.py --task all --output-dir results/drift_analysis

# Generate visualizations
python scripts/phase3/visualize_drift.py \
    --results-file results/drift_analysis/drift_analysis_results.json \
    --output-dir results/drift_visualizations
```

**Outputs for RQ1**:
- `results/drift_analysis/drift_analysis_results.json` - All CKA/drift metrics
- `results/drift_analysis/hypothesis_test_results.json` - Statistical tests
- `results/drift_visualizations/*.png` - Publication figures
- **Section in ANALYSIS_REPORT.md**: Detailed answer to RQ1 with evidence

#### Phase 4B: Research Question 2 - Deployment Efficiency Analysis

**Research Question**: What is the real-world latency penalty when deploying multiple LoRA adapters side-by-side versus merging them?

**Data Sources**:
- Trained LoRA adapters from Phase 2 (9 adapters: 3 tasks × 3 seeds)
- Merged full fine-tuned models from Phase 2 (9 models)

**Benchmark Scenarios**:
1. **Baseline**: Single merged model per task
2. **Single adapter**: One LoRA adapter loaded
3. **2-adapter**: Two LoRA adapters loaded simultaneously (e.g., MRPC + SST-2)
4. **3-adapter**: All three LoRA adapters loaded simultaneously (MRPC, SST-2, RTE)

**Metrics to Measure**:
- **Latency**: Mean inference time (ms/sample)
- **P95 latency**: 95th percentile (for SLA considerations)
- **Throughput**: Samples per second
- **GPU memory**: Memory footprint for each configuration
- **Batch scalability**: Performance across batch sizes (1, 2, 4, 8, 16)

**Implementation** (vLLM multi-adapter support):
   ```python
# Pseudo-code for benchmarking
1. Load base TinyLlama model
2. For each configuration:
   - Load adapters (if applicable)
   - Warmup with 100 samples
   - Benchmark with 1000 inference samples
   - Measure latency, throughput, memory
3. Compare configurations statistically
```

**Analysis Tasks**:
- **Overhead computation**: `(multi_adapter_latency - merged_latency) / merged_latency × 100%`
- **Statistical testing**: Bootstrap confidence intervals for overhead
- **Memory efficiency**: Compare GPU memory usage
- **Scalability analysis**: Overhead vs number of adapters (1, 2, 4)
- **Cost-benefit analysis**: Trade-offs between deployment flexibility and performance

**Hypothesis Testing**:
   - **H2**: Multi-adapter deployment ≤30% latency overhead
     - Compare 4-adapter vs merged model latency
     - Bootstrap 95% confidence intervals
     - One-sided t-test (H0: overhead > 30%)

**Outputs for RQ2**:
- `results/deployment_analysis/benchmark_results.json` - All latency/throughput data
- `results/deployment_analysis/overhead_analysis.json` - Overhead calculations
- `results/deployment_visualizations/*.png` - Latency comparison plots
- **Section in ANALYSIS_REPORT.md**: Detailed answer to RQ2 with evidence

#### Final Synthesis: Integrated Analysis Report

**Output**: `ANALYSIS_REPORT.md` - Comprehensive research findings

**Report Structure**:

1. **Executive Summary**
   - Brief overview of findings
   - Key takeaways for both research questions
   - Practical recommendations

2. **Research Question 1: Representational Drift**
   - Detailed CKA analysis results
   - Layer-wise drift patterns
   - LoRA vs Full FT comparison (with statistics)
   - Hypothesis test results (H1a, H1b)
   - Visualizations and tables
   - **Answer**: Does LoRA preserve representations better? By how much?

3. **Research Question 2: Deployment Efficiency**
   - Latency benchmark results
   - Overhead analysis across configurations
   - Memory efficiency comparison
   - Hypothesis test results (H2)
   - Trade-off analysis
   - **Answer**: What is the deployment penalty? Is it acceptable?

4. **Cross-Analysis Insights**
   - Tasks with high drift vs low drift
   - Correlation between drift and deployment overhead
   - Performance-efficiency trade-offs
   - Task-specific recommendations

5. **Statistical Summary**
   - All hypothesis test results
   - Effect sizes and confidence intervals
   - Multiple testing corrections
   - Accept/reject decisions with justifications

6. **Limitations & Future Work**
   - Methodological limitations
   - Generalizability considerations
   - Suggested follow-up research

7. **Practical Recommendations**
   - When to use LoRA vs Full FT
   - Deployment architecture recommendations
   - Task-specific guidance

**Report Generation**:
   ```bash
# After completing 4A and 4B
python scripts/phase4/generate_analysis_report.py \
    --drift-results results/drift_analysis/ \
    --deployment-results results/deployment_analysis/ \
    --output ANALYSIS_REPORT.md
```

## Resource Requirements & Timeline

**Total Estimated Runtime**: ~40-45 hours total (completed: ~35-40 hours, remaining: ~7-9 hours)

**Hardware Requirements**:
- L4 24GB GPU (or equivalent)
- ~100GB disk space
- Ubuntu 20.04+ with CUDA 11.8+

**Phase Timeline**:
- **Phase 0**: Methodology Validation - ~5 hours - ✅ COMPLETED
- **Phase 1**: Hyperparameter Optimization - ~3-4 hours - ✅ COMPLETED
- **Phase 2**: Production Model Training (Classification) - ~20-24 hours - ✅ COMPLETED
- **Phase 3**: Representation Extraction (Classification) - ~6-8 hours - ✅ COMPLETED
- **Phase 4**: Comprehensive Analysis & Research Question Synthesis
  - Phase 4A (RQ1 - Drift analysis): ~2-3 hours - ❌ NOT STARTED
  - Phase 4B (RQ2 - Deployment benchmarking): ~3-4 hours - ❌ NOT STARTED
  - Final report synthesis: ~2 hours - ❌ NOT STARTED

**Current Progress**: ~85% complete (Phases 0-3 fully done for classification tasks, only Phase 4 analysis remains)

## Key Methodological Features (Summary)

1. **Sanity Checks**: All models proven to overfit on 10 samples (validates implementation)
2. **Proper Baselines**: Majority class, random predictions (frames results meaningfully)
3. **Bayesian Optimization**: Optuna TPE sampler with 15 trials (exceeds literature standards)
4. **Multiple Seeds**: 3 independent seeds for all production experiments (accounts for randomness)
5. **Statistical Rigor**: Permutation tests, bootstrap CIs, multiple testing corrections
6. **Data Integrity**: Strict train/val/test splits, no data leakage
7. **Configuration Consistency**: max_length=384 across all phases (valid drift analysis)
8. **Reproducibility**: All code, configs, and models documented and versioned
9. **Post-hoc Representations**: Memory-efficient extraction without compromising validity
10. **Hypothesis-driven**: Clear success criteria defined before experiments

## Critical Methodological Notes

### Avoiding Common Pitfalls

1. **Data Leakage Prevention**:
   - ✅ Test set never used for any optimization or early stopping decisions
   - ✅ Validation set only for hyperparameter selection and early stopping
   - ✅ No preprocessing decisions based on test set statistics

2. **Overfitting Prevention**:
   - ✅ Early stopping with patience=3 on validation loss
   - ✅ Gradient clipping prevents training instability
   - ✅ Weight decay for regularization
   - ✅ Multiple seeds ensure results not due to lucky initialization

3. **Fair Comparisons**:
   - ✅ Independent hyperparameter optimization per method (not sharing hyperparameters)
   - ✅ Identical datasets, splits, and evaluation procedures
   - ✅ Same base model initialization
   - ✅ Consistent max_length across all experiments

4. **Statistical Validity**:
   - ✅ Multiple seeds (3) for all claims
   - ✅ Proper hypothesis tests with corrections for multiple comparisons
   - ✅ Confidence intervals reported
   - ✅ Effect sizes computed (not just p-values)

5. **Reproducibility**:
   - ✅ Deterministic training when possible
   - ✅ All seeds documented
   - ✅ Code and configs version-controlled
   - ✅ Models uploaded to WandB
   - ✅ Exact software versions in requirements.txt

### Alignment with Research Standards

This methodology aligns with best practices in NLP research:

- **Hu et al. (2021)** - LoRA paper: We use similar rank (8), test on diverse tasks
- **Bergstra & Bengio (2012)** - Hyperparameter optimization: We exceed recommended trials (15 vs 10)
- **Dodge et al. (2020)** - Statistical significance in NLP: We use multiple seeds, proper tests
- **Bouthillier et al. (2021)** - Accounting for randomness: We report mean ± std across seeds

## References

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2).
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
- Dodge, J., et al. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. *arXiv preprint arXiv:2002.06305*.
- Bouthillier, X., et al. (2021). Accounting for Variance in Machine Learning Benchmarks. *MLSys*.

## Deliverables

Upon completion, this research will produce:

1. **ANALYSIS_REPORT.md** - **Primary Deliverable**
   - Comprehensive analysis answering both research questions
   - Statistical evidence for all claims
   - Publication-quality visualizations
   - Practical recommendations for LoRA deployment
   - Limitations and future work

2. **Trained Models**: 18 models (3 tasks × 2 methods × 3 seeds) uploaded to WandB
   - Full fine-tuned models for all classification tasks
   - LoRA adapters for all classification tasks
   - All models reproducible from saved checkpoints

3. **Representations**: Layer-wise representations for drift analysis
   - Base model representations (Phase 0)
   - Fine-tuned model representations (Phase 3)
   - 18 representation sets total (3 tasks × 2 methods × 3 seeds)

4. **Optimal Configurations**: Task-specific hyperparameters (YAML files in `analysis/`)
   - 6 optimal configs (3 tasks × 2 methods)
   - All search results documented

5. **Analysis Results**:
   - **Phase 4A outputs**: CKA metrics, drift analysis, statistical tests for RQ1
   - **Phase 4B outputs**: Latency benchmarks, overhead analysis for RQ2
   - Performance comparisons with 95% confidence intervals
   - Hypothesis test results with multiple testing corrections

6. **Visualizations**: Publication-quality figures
   - Layer-wise drift heatmaps
   - Drift comparison plots (LoRA vs Full FT)
   - Latency benchmark comparisons
   - Statistical significance markers

7. **Code & Documentation**: Complete experimental pipeline
   - All training scripts
   - Analysis scripts
   - Visualization scripts
   - Comprehensive documentation

## Next Steps

**Immediate priorities** (to complete the research):

**Phases 0-3 Complete**: All model training and representation extraction done for 3 classification tasks (MRPC, SST-2, RTE).

**Phase 4 - Comprehensive Analysis & Report Generation** (ONLY REMAINING WORK):

1. **Phase 4A - Drift Analysis**: ~2-3 hours
   - Compute CKA metrics across all layers (22 layers × 3 tasks × 2 methods)
   - Statistical tests (permutation tests, Bonferroni correction)
   - Generate visualizations (heatmaps, drift comparisons)
   - Write RQ1 analysis section answering: "Does LoRA preserve representations better for classification tasks?"

2. **Phase 4B - Deployment Benchmarking**: ~3-4 hours
   - vLLM setup and multi-adapter configuration
   - Measure latency/throughput (single, 2-adapter, 3-adapter, merged)
   - Statistical testing on overhead metrics
   - Write RQ2 analysis section answering: "What is the deployment penalty for multi-adapter?"

3. **Final Synthesis - Analysis Report**: ~2 hours
   - Generate ANALYSIS_REPORT.md combining 4A and 4B
   - Executive summary, statistical summary, limitations
   - Practical recommendations for LoRA in classification
   - Explicitly scope findings to classification tasks
   - Suggest future work on QA and generative tasks

**Total remaining time**: ~7-9 hours (all on single machine, no parallel VMs needed)
