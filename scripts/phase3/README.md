# Phase 3: Post-Training Representation Extraction & Drift Analysis

## Overview

Phase 3 extracts representations from models trained in Phase 2 and performs drift analysis. This separation is critical for:

### **Why Separate Representation Extraction from Training?**

**The Problem:**
- Representation extraction during training consumes **12-15GB GPU memory**
- This forced `batch_size=1`, making training **4x slower**
- Made Phase 2 (full dataset training) impossible within time constraints

**The Solution:**
- Train models efficiently in Phases 1-2 (batch_size=2-4, representation extraction disabled)
- Extract representations post-hoc in Phase 3 (memory-efficient, no training overhead)

**Methodological Soundness:**
- Representations are **properties of trained models**, not training artifacts
- Post-training extraction is standard practice in representation analysis research
- Produces identical results to in-training extraction
- Enables memory-efficient layer-by-layer processing

---

## Infrastructure Status

### ✅ **Already Implemented**
1. **Representation Extractors**:
   - `RepresentationExtractor` class in `experiments/full_finetune.py`
   - `LoRARepresentationExtractor` class in `experiments/lora_finetune.py`
   - Layer-wise hooks, chunked processing, disk streaming

2. **Model Saving**:
   - All Phase 2 models automatically saved to `results/{method}_{timestamp}/final_model`
   - Includes model weights, configuration, and metadata

3. **Analysis Tools**:
   - CKA (Centered Kernel Alignment) in `shared/metrics.py`
   - Cosine similarity metrics in `shared/metrics.py`

### ✅ **Fully Implemented**
1. **Phase 3 Extraction Scripts** ✅:
   - `scripts/phase3/extract_representations.py` - Complete model loading and extraction
   - `scripts/phase3/vm1.sh` - SQuAD v2 extraction pipeline
   - `scripts/phase3/vm2.sh` - Classification tasks pipeline (CREATED)

2. **Drift Analysis Scripts** ✅:
   - `scripts/phase3/analyze_drift.py` - Complete CKA/cosine similarity analysis (CREATED)
   - Compare base model vs fine-tuned representations
   - Compare LoRA vs Full FT drift across all seeds
   - Statistical significance testing with permutation tests
   - Hypothesis testing for 20% drift reduction target

3. **Visualization Scripts** ✅:
   - `scripts/phase3/visualize_drift.py` - Publication-quality plots (CREATED)
   - Layer-wise drift heatmaps
   - Drift reduction comparison plots
   - Statistical significance visualizations
   - Comprehensive research dashboard

---

## Execution Flow

### **Step 1: Run Extraction Scripts**

```bash
# VM1: SQuAD v2
cd /path/to/NLP
bash scripts/phase3/vm1.sh

# VM2: Classification tasks (MRPC, SST-2, RTE)
bash scripts/phase3/vm2.sh
```

**Runtime**: ~8-12 hours total (both VMs in parallel)

### **Step 2: Drift Analysis** ✅

```bash
# Analyze drift for all tasks (recommended)
python scripts/phase3/analyze_drift.py --task all --output-dir results/drift_analysis

# Or analyze individual tasks
python scripts/phase3/analyze_drift.py --task squad_v2 --output-dir results/drift_analysis
python scripts/phase3/analyze_drift.py --task mrpc --output-dir results/drift_analysis
python scripts/phase3/analyze_drift.py --task sst2 --output-dir results/drift_analysis
python scripts/phase3/analyze_drift.py --task rte --output-dir results/drift_analysis
```

### **Step 3: Visualization** ✅

```bash
# Generate comprehensive visualizations
python scripts/phase3/visualize_drift.py \
    --results-file results/drift_analysis/drift_analysis_results.json \
    --output-dir results/drift_visualizations

# Output: Publication-quality plots for research paper
```

### **Step 4: Complete Analysis Pipeline** ✅

```bash
# Run the complete Phase 3 pipeline
bash scripts/phase3/vm1.sh     # SQuAD v2 extraction
bash scripts/phase3/vm2.sh     # Classification extraction  
python scripts/phase3/analyze_drift.py --task all  # Comprehensive analysis
python scripts/phase3/visualize_drift.py           # Generate all plots
```

---

## Output Structure

```
results/phase3_representations/
├── squad_v2_full_finetune_seed42/
│   └── step_000000/
│       ├── layer_0.pt
│       ├── layer_1.pt
│       ├── ...
│       ├── layer_23.pt
│       └── metadata.json
├── squad_v2_lora_seed42/
│   └── step_000000/
│       ├── layer_0.pt
│       ├── ...
│       └── adapter_weights/
│           ├── lora_A_layer_0.pt
│           └── lora_B_layer_0.pt
└── ...
```

---

## Memory Requirements

**During Training (Phases 1-2):**
- Representation extraction: **DISABLED**
- Batch size: 2-4 (instead of forced 1)
- Memory usage: ~15-18GB

**During Extraction (Phase 3):**
- No training overhead (no gradients, no optimizer states)
- Layer-by-layer processing with immediate disk save
- Memory usage: ~8-12GB (much lower!)

---

## Research Questions Addressed

1. **Representational Drift**: Compare CKA similarity between base and fine-tuned models
2. **LoRA vs Full FT**: Quantify difference in representation preservation
3. **Layer-wise Analysis**: Identify which layers drift most during fine-tuning
4. **Task-specific Patterns**: Compare drift across SQuAD v2 vs classification tasks

---

## Next Steps

1. **Complete VM2 script**: Create `scripts/phase3/vm2.sh` for classification tasks
2. **Test extraction**: Run Phase 3 on one model to validate pipeline
3. **Drift analysis**: Implement comparison scripts using existing CKA/cosine tools
4. **Visualization**: Create publication-quality plots for paper

---

## Technical Notes

- Representations extracted at **step 0** (final trained model)
- Can optionally extract at multiple checkpoints if Phase 2 saved them
- Extraction uses same 750 validation samples as Phase 0-2 for consistency
- All 24 transformer layers extracted for comprehensive analysis


## 🔄 Base Representations Management

**Storage**: Base model representations (48GB) are stored in WandB artifacts, NOT in git.

**Download when needed** (e.g., for Phase 3 analysis on a new VM):
```bash
bash scripts/download_base_representations.sh
```

This downloads the base representations from WandB to `base_representations/` directory.

**Why not in git?**
- 48GB is too large for git/GitHub
- Would consume limited VM disk space unnecessarily  
- Already safely stored in WandB artifacts
- Can be re-downloaded anytime for analysis
