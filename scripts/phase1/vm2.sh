#!/bin/bash
# Phase 1 - VM2: BALANCED SWEEP-FIRST (Classification Tasks)
# MRPC + SST-2 + RTE - perfectly balanced workload
set -e  # Exit on error

echo "🚀 PHASE 1 - VM2: BALANCED SWEEP-FIRST (Classification Tasks)"
echo "============================================================"
echo "Independent workflow - All classification tasks:"
echo "1. Hyperparameter sweeps for MRPC + SST-2 + RTE"
echo "2. Analysis of each task separately (3 YAML files)"
echo "3. Production experiments using optimal hyperparameters"
echo "4. Base representations for MRPC + SST-2 + RTE"
echo "============================================================"

# Setup environment
export WANDB_PROJECT=NLP-Phase1-Training
export WANDB_ENTITY=galavny-tel-aviv-university

# Clear GPU memory cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('No CUDA available')
"

# Create logs directory
mkdir -p logs/phase1_balanced/vm2

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# PHASE 1A: HYPERPARAMETER SWEEPS (All Classification Tasks)
# ============================================================================
echo "🔬 PHASE 1A: HYPERPARAMETER SWEEPS"
echo "Find optimal hyperparameters for all classification tasks"
echo "------------------------------------------------------------"

# MRPC Sweeps
echo "⚡ [1/6] MRPC Full Fine-tuning Hyperparameter Sweep"
if python experiments/full_finetune.py --task mrpc --mode sweep --no-base-representations > logs/phase1_balanced/vm2/mrpc_full_sweep.log 2>&1; then
    echo "✅ MRPC full fine-tuning sweep completed"
else
    echo "❌ MRPC full fine-tuning sweep FAILED"
    exit 1
fi

echo "⚡ [2/6] MRPC LoRA Hyperparameter Sweep"
if python experiments/lora_finetune.py --task mrpc --mode sweep > logs/phase1_balanced/vm2/mrpc_lora_sweep.log 2>&1; then
    echo "✅ MRPC LoRA sweep completed"
else
    echo "❌ MRPC LoRA sweep FAILED"
    exit 1
fi

# SST-2 Sweeps
echo "⚡ [3/6] SST-2 Full Fine-tuning Hyperparameter Sweep"
if python experiments/full_finetune.py --task sst2 --mode sweep --no-base-representations > logs/phase1_balanced/vm2/sst2_full_sweep.log 2>&1; then
    echo "✅ SST-2 full fine-tuning sweep completed"
else
    echo "❌ SST-2 full fine-tuning sweep FAILED"
    exit 1
fi

echo "⚡ [4/6] SST-2 LoRA Hyperparameter Sweep"
if python experiments/lora_finetune.py --task sst2 --mode sweep > logs/phase1_balanced/vm2/sst2_lora_sweep.log 2>&1; then
    echo "✅ SST-2 LoRA sweep completed"
else
    echo "❌ SST-2 LoRA sweep FAILED"
    exit 1
fi

# RTE Sweeps
echo "⚡ [5/6] RTE Full Fine-tuning Hyperparameter Sweep"
if python experiments/full_finetune.py --task rte --mode sweep --no-base-representations > logs/phase1_balanced/vm2/rte_full_sweep.log 2>&1; then
    echo "✅ RTE full fine-tuning sweep completed"
else
    echo "❌ RTE full fine-tuning sweep FAILED"
    exit 1
fi

echo "⚡ [6/6] RTE LoRA Hyperparameter Sweep"
if python experiments/lora_finetune.py --task rte --mode sweep > logs/phase1_balanced/vm2/rte_lora_sweep.log 2>&1; then
    echo "✅ RTE LoRA sweep completed"
else
    echo "❌ RTE LoRA sweep FAILED"
    exit 1
fi

echo "🎯 PHASE 1A COMPLETE: All classification task sweeps finished!"
echo ""

# ============================================================================
# PHASE 1B: TASK-SPECIFIC SWEEP ANALYSIS (One YAML per task)
# ============================================================================
echo "📊 PHASE 1B: TASK-SPECIFIC SWEEP ANALYSIS"
echo "Analyzing each task separately for complete independence"
echo "------------------------------------------------------------"

mkdir -p analysis

# Analyze MRPC
echo "⚡ [1/3] Analyzing MRPC sweep results..."
if python scripts/analyze_sweeps.py --tasks mrpc --export-optimal-configs --output-dir analysis > logs/phase1_balanced/vm2/mrpc_sweep_analysis.log 2>&1; then
    echo "✅ MRPC sweep analysis completed"
    mv analysis/optimal_hyperparameters.yaml analysis/mrpc_optimal_hyperparameters.yaml
    echo "📄 MRPC optimal hyperparameters saved to: analysis/mrpc_optimal_hyperparameters.yaml"
else
    echo "❌ MRPC sweep analysis FAILED"
    exit 1
fi

# Analyze SST-2
echo "⚡ [2/3] Analyzing SST-2 sweep results..."
if python scripts/analyze_sweeps.py --tasks sst2 --export-optimal-configs --output-dir analysis > logs/phase1_balanced/vm2/sst2_sweep_analysis.log 2>&1; then
    echo "✅ SST-2 sweep analysis completed"
    mv analysis/optimal_hyperparameters.yaml analysis/sst2_optimal_hyperparameters.yaml
    echo "📄 SST-2 optimal hyperparameters saved to: analysis/sst2_optimal_hyperparameters.yaml"
else
    echo "❌ SST-2 sweep analysis FAILED"
    exit 1
fi

# Analyze RTE
echo "⚡ [3/3] Analyzing RTE sweep results..."
if python scripts/analyze_sweeps.py --tasks rte --export-optimal-configs --output-dir analysis > logs/phase1_balanced/vm2/rte_sweep_analysis.log 2>&1; then
    echo "✅ RTE sweep analysis completed"
    mv analysis/optimal_hyperparameters.yaml analysis/rte_optimal_hyperparameters.yaml
    echo "📄 RTE optimal hyperparameters saved to: analysis/rte_optimal_hyperparameters.yaml"
else
    echo "❌ RTE sweep analysis FAILED"
    exit 1
fi

echo "📋 ALL TASK-SPECIFIC OPTIMAL HYPERPARAMETERS IDENTIFIED:"
echo "------------------------------------------------------------"
for task in mrpc sst2 rte; do
    echo ""
    echo "${task^^} OPTIMAL HYPERPARAMETERS:"
    python -c "
import yaml
with open('analysis/${task}_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for task_name, methods in optimal_hp.items():
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
        print(f'  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Acc={acc:.3f}')
"
done

echo "🎯 PHASE 1B COMPLETE: All task-specific optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: PRODUCTION EXPERIMENTS (Using Task-Specific Optimal Hyperparameters)
# ============================================================================
echo "🚀 PHASE 1C: PRODUCTION EXPERIMENTS"
echo "Running classification experiments with task-specific optimal hyperparameters"
echo "------------------------------------------------------------"

# MRPC Production Experiments
echo "🎯 [1/9] MRPC Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/mrpc_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'mrpc' in optimal_hp and method in optimal_hp['mrpc']:
        hp = optimal_hp['mrpc'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export MRPC_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export MRPC_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export MRPC_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export MRPC_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export MRPC_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export MRPC_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export MRPC_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > mrpc_hyperparams.sh
source mrpc_hyperparams.sh

# MRPC Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_FULLFINETUNE_LR \
        --batch-size $MRPC_FULLFINETUNE_BS \
        --warmup-ratio $MRPC_FULLFINETUNE_WU \
        --epochs $MRPC_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/mrpc_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# MRPC LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_LORA_LR \
        --batch-size $MRPC_LORA_BS \
        --warmup-ratio $MRPC_LORA_WU \
        --epochs $MRPC_LORA_EP \
        --lora-r $MRPC_LORA_R \
        --lora-alpha $MRPC_LORA_A \
        --lora-dropout $MRPC_LORA_D \
        > logs/phase1_balanced/vm2/mrpc_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 Production Experiments
echo "🎯 [2/9] SST-2 Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/sst2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'sst2' in optimal_hp and method in optimal_hp['sst2']:
        hp = optimal_hp['sst2'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export SST2_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export SST2_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export SST2_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export SST2_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export SST2_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export SST2_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export SST2_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > sst2_hyperparams.sh
source sst2_hyperparams.sh

# SST-2 Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_FULLFINETUNE_LR \
        --batch-size $SST2_FULLFINETUNE_BS \
        --warmup-ratio $SST2_FULLFINETUNE_WU \
        --epochs $SST2_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/sst2_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_LORA_LR \
        --batch-size $SST2_LORA_BS \
        --warmup-ratio $SST2_LORA_WU \
        --epochs $SST2_LORA_EP \
        --lora-r $SST2_LORA_R \
        --lora-alpha $SST2_LORA_A \
        --lora-dropout $SST2_LORA_D \
        > logs/phase1_balanced/vm2/sst2_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# RTE Production Experiments
echo "🎯 [3/9] RTE Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'rte' in optimal_hp and method in optimal_hp['rte']:
        hp = optimal_hp['rte'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export RTE_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export RTE_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export RTE_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export RTE_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export RTE_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export RTE_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export RTE_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > rte_hyperparams.sh
source rte_hyperparams.sh

# RTE Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_FULLFINETUNE_LR \
        --batch-size $RTE_FULLFINETUNE_BS \
        --warmup-ratio $RTE_FULLFINETUNE_WU \
        --epochs $RTE_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/rte_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) FAILED"
    exit 1
fi
done

# RTE LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_LORA_LR \
        --batch-size $RTE_LORA_BS \
        --warmup-ratio $RTE_LORA_WU \
        --epochs $RTE_LORA_EP \
        --lora-r $RTE_LORA_R \
        --lora-alpha $RTE_LORA_A \
        --lora-dropout $RTE_LORA_D \
        > logs/phase1_balanced/vm2/rte_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 PHASE 1C COMPLETE: All classification production experiments finished!"
echo ""

# ============================================================================
# PHASE 1D: BASE REPRESENTATIONS (Classification Tasks)
# ============================================================================
echo "🧠 PHASE 1D: BASE REPRESENTATIONS EXTRACTION"
echo "Extracting base model representations for classification tasks"
echo "------------------------------------------------------------"

echo "⚡ [1/3] Extracting base model representations for MRPC..."
if python scripts/extract_base_representations.py --task mrpc --output-dir representations/base_model > logs/phase1_balanced/vm2/base_mrpc.log 2>&1; then
    echo "✅ MRPC base representations extracted"
else
    echo "❌ MRPC base representation extraction FAILED"
    exit 1
fi

echo "⚡ [2/3] Extracting base model representations for SST-2..."
if python scripts/extract_base_representations.py --task sst2 --output-dir representations/base_model > logs/phase1_balanced/vm2/base_sst2.log 2>&1; then
    echo "✅ SST-2 base representations extracted"
else
    echo "❌ SST-2 base representation extraction FAILED"
    exit 1
fi

echo "⚡ [3/3] Extracting base model representations for RTE..."
if python scripts/extract_base_representations.py --task rte --output-dir representations/base_model > logs/phase1_balanced/vm2/base_rte.log 2>&1; then
    echo "✅ RTE base representations extracted"
else
    echo "❌ RTE base representation extraction FAILED"
    exit 1
fi

echo "🎯 PHASE 1D COMPLETE: All classification base representations extracted!"
echo ""

# ============================================================================
# PHASE 1E: INFRASTRUCTURE PREPARATION
# ============================================================================
echo "📋 PHASE 1E: INFRASTRUCTURE PREPARATION"
echo "Preparing shared drift analysis infrastructure"
echo "------------------------------------------------------------"

# Create directory structure for drift analysis
mkdir -p analysis/drift_analysis/{base_model,full_finetune,lora}/{representations,metadata}

# Generate metadata for drift analysis
echo "⚡ Generating drift analysis metadata..."
python -c "
import json
import os
from datetime import datetime

# Create metadata for drift analysis
metadata = {
    'base_model': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'extraction_date': datetime.now().isoformat(),
        'tasks': ['mrpc', 'sst2', 'rte', 'squad_v2'],
        'representation_layers': list(range(22)),  # TinyLlama has 22 layers
        'representation_dir': 'representations/base_model/'
    },
    'analysis_config': {
        'drift_metrics': ['linear_cka', 'cosine_similarity'],
        'comparison_pairs': [
            ['base_model', 'full_finetune'],
            ['base_model', 'lora']
        ],
        'statistical_tests': ['permutation_test', 'bootstrap_ci'],
        'visualization': ['heatmaps', 'layer_evolution', 'correlation_plots']
    },
    'tasks_metadata': {
        'mrpc': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'sst2': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'}, 
        'rte': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'squad_v2': {'type': 'qa', 'metric': 'f1', 'vm': 'vm1'}
    },
    'vm_distribution': {
        'vm1': ['squad_v2'],
        'vm2': ['mrpc', 'sst2', 'rte']
    },
    'file_structure': {
        'optimal_hyperparameters': {
            'mrpc': 'analysis/mrpc_optimal_hyperparameters.yaml',
            'sst2': 'analysis/sst2_optimal_hyperparameters.yaml',
            'rte': 'analysis/rte_optimal_hyperparameters.yaml',
            'squad_v2': 'analysis/squad_v2_optimal_hyperparameters.yaml'
        }
    }
}

# Save metadata
os.makedirs('analysis/drift_analysis', exist_ok=True)
with open('analysis/drift_analysis/drift_analysis_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✅ Drift analysis metadata created')
" > logs/phase1_balanced/vm2/drift_prep.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Drift analysis metadata generated"
else
    echo "❌ Drift analysis metadata generation FAILED"
    exit 1
fi

# Cleanup and optimization
echo "⚡ Cleaning up temporary files..."
python scripts/auto_cleanup.py --target-dir logs/phase1_balanced/vm2 --keep-latest 5 > logs/phase1_balanced/vm2/cleanup.log 2>&1

echo "🎯 PHASE 1E COMPLETE: Infrastructure preparation finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM2 BALANCED WORKFLOW COMPLETE! $(date)"
echo "=========================================="
echo "✅ Phase 1A: Classification hyperparameter sweeps completed (6 sweeps)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified (3 YAML files)"
echo "✅ Phase 1C: Production experiments completed (18 experiments)"
echo "✅ Phase 1D: Base representations extracted (3 tasks)"
echo "✅ Phase 1E: Infrastructure preparation completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "📄 Task-specific optimal configs:"
echo "   • analysis/mrpc_optimal_hyperparameters.yaml"
echo "   • analysis/sst2_optimal_hyperparameters.yaml"
echo "   • analysis/rte_optimal_hyperparameters.yaml"
echo "🧠 Base representations: representations/base_model/{mrpc,sst2,rte}/"
echo "📋 Drift metadata: analysis/drift_analysis/drift_analysis_metadata.json"
echo "📋 VM2 results ready for Phase 2a analysis!"
echo ""
echo "🚀 INDEPENDENT EXECUTION: VM2 complete with NO dependencies on VM1!"
echo "⚖️ BALANCED WORKLOAD: Classification tasks (3x weight) perfectly balanced with VM1!"
for task in mrpc sst2 rte; do
    echo ""
    echo "${task^^} OPTIMAL HYPERPARAMETERS:"
    python -c "
import yaml
with open('analysis/${task}_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for task_name, methods in optimal_hp.items():
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
        print(f'  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Acc={acc:.3f}')
"
done

echo "🎯 PHASE 1B COMPLETE: All task-specific optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: PRODUCTION EXPERIMENTS (Using Task-Specific Optimal Hyperparameters)
# ============================================================================
echo "🚀 PHASE 1C: PRODUCTION EXPERIMENTS"
echo "Running classification experiments with task-specific optimal hyperparameters"
echo "------------------------------------------------------------"

# MRPC Production Experiments
echo "🎯 [1/9] MRPC Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/mrpc_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'mrpc' in optimal_hp and method in optimal_hp['mrpc']:
        hp = optimal_hp['mrpc'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export MRPC_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export MRPC_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export MRPC_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export MRPC_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export MRPC_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export MRPC_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export MRPC_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > mrpc_hyperparams.sh
source mrpc_hyperparams.sh

# MRPC Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_FULLFINETUNE_LR \
        --batch-size $MRPC_FULLFINETUNE_BS \
        --warmup-ratio $MRPC_FULLFINETUNE_WU \
        --epochs $MRPC_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/mrpc_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# MRPC LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_LORA_LR \
        --batch-size $MRPC_LORA_BS \
        --warmup-ratio $MRPC_LORA_WU \
        --epochs $MRPC_LORA_EP \
        --lora-r $MRPC_LORA_R \
        --lora-alpha $MRPC_LORA_A \
        --lora-dropout $MRPC_LORA_D \
        > logs/phase1_balanced/vm2/mrpc_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 Production Experiments
echo "🎯 [2/9] SST-2 Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/sst2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'sst2' in optimal_hp and method in optimal_hp['sst2']:
        hp = optimal_hp['sst2'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export SST2_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export SST2_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export SST2_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export SST2_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export SST2_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export SST2_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export SST2_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > sst2_hyperparams.sh
source sst2_hyperparams.sh

# SST-2 Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_FULLFINETUNE_LR \
        --batch-size $SST2_FULLFINETUNE_BS \
        --warmup-ratio $SST2_FULLFINETUNE_WU \
        --epochs $SST2_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/sst2_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_LORA_LR \
        --batch-size $SST2_LORA_BS \
        --warmup-ratio $SST2_LORA_WU \
        --epochs $SST2_LORA_EP \
        --lora-r $SST2_LORA_R \
        --lora-alpha $SST2_LORA_A \
        --lora-dropout $SST2_LORA_D \
        > logs/phase1_balanced/vm2/sst2_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# RTE Production Experiments
echo "🎯 [3/9] RTE Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'rte' in optimal_hp and method in optimal_hp['rte']:
        hp = optimal_hp['rte'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export RTE_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export RTE_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export RTE_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export RTE_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export RTE_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export RTE_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export RTE_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > rte_hyperparams.sh
source rte_hyperparams.sh

# RTE Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_FULLFINETUNE_LR \
        --batch-size $RTE_FULLFINETUNE_BS \
        --warmup-ratio $RTE_FULLFINETUNE_WU \
        --epochs $RTE_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/rte_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# RTE LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_LORA_LR \
        --batch-size $RTE_LORA_BS \
        --warmup-ratio $RTE_LORA_WU \
        --epochs $RTE_LORA_EP \
        --lora-r $RTE_LORA_R \
        --lora-alpha $RTE_LORA_A \
        --lora-dropout $RTE_LORA_D \
        > logs/phase1_balanced/vm2/rte_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 PHASE 1C COMPLETE: All classification production experiments finished!"
echo ""

# ============================================================================
# PHASE 1D: BASE REPRESENTATIONS (Classification Tasks)
# ============================================================================
echo "🧠 PHASE 1D: BASE REPRESENTATIONS EXTRACTION"
echo "Extracting base model representations for classification tasks"
echo "------------------------------------------------------------"

echo "⚡ [1/3] Extracting base model representations for MRPC..."
if python scripts/extract_base_representations.py --task mrpc --output-dir representations/base_model > logs/phase1_balanced/vm2/base_mrpc.log 2>&1; then
    echo "✅ MRPC base representations extracted"
else
    echo "❌ MRPC base representation extraction FAILED"
    exit 1
fi

echo "⚡ [2/3] Extracting base model representations for SST-2..."
if python scripts/extract_base_representations.py --task sst2 --output-dir representations/base_model > logs/phase1_balanced/vm2/base_sst2.log 2>&1; then
    echo "✅ SST-2 base representations extracted"
else
    echo "❌ SST-2 base representation extraction FAILED"
    exit 1
fi

echo "⚡ [3/3] Extracting base model representations for RTE..."
if python scripts/extract_base_representations.py --task rte --output-dir representations/base_model > logs/phase1_balanced/vm2/base_rte.log 2>&1; then
    echo "✅ RTE base representations extracted"
else
    echo "❌ RTE base representation extraction FAILED"
    exit 1
fi

echo "🎯 PHASE 1D COMPLETE: All classification base representations extracted!"
echo ""

# ============================================================================
# PHASE 1E: INFRASTRUCTURE PREPARATION
# ============================================================================
echo "📋 PHASE 1E: INFRASTRUCTURE PREPARATION"
echo "Preparing shared drift analysis infrastructure"
echo "------------------------------------------------------------"

# Create directory structure for drift analysis
mkdir -p analysis/drift_analysis/{base_model,full_finetune,lora}/{representations,metadata}

# Generate metadata for drift analysis
echo "⚡ Generating drift analysis metadata..."
python -c "
import json
import os
from datetime import datetime

# Create metadata for drift analysis
metadata = {
    'base_model': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'extraction_date': datetime.now().isoformat(),
        'tasks': ['mrpc', 'sst2', 'rte', 'squad_v2'],
        'representation_layers': list(range(22)),  # TinyLlama has 22 layers
        'representation_dir': 'representations/base_model/'
    },
    'analysis_config': {
        'drift_metrics': ['linear_cka', 'cosine_similarity'],
        'comparison_pairs': [
            ['base_model', 'full_finetune'],
            ['base_model', 'lora']
        ],
        'statistical_tests': ['permutation_test', 'bootstrap_ci'],
        'visualization': ['heatmaps', 'layer_evolution', 'correlation_plots']
    },
    'tasks_metadata': {
        'mrpc': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'sst2': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'}, 
        'rte': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'squad_v2': {'type': 'qa', 'metric': 'f1', 'vm': 'vm1'}
    },
    'vm_distribution': {
        'vm1': ['squad_v2'],
        'vm2': ['mrpc', 'sst2', 'rte']
    },
    'file_structure': {
        'optimal_hyperparameters': {
            'mrpc': 'analysis/mrpc_optimal_hyperparameters.yaml',
            'sst2': 'analysis/sst2_optimal_hyperparameters.yaml',
            'rte': 'analysis/rte_optimal_hyperparameters.yaml',
            'squad_v2': 'analysis/squad_v2_optimal_hyperparameters.yaml'
        }
    }
}

# Save metadata
os.makedirs('analysis/drift_analysis', exist_ok=True)
with open('analysis/drift_analysis/drift_analysis_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✅ Drift analysis metadata created')
" > logs/phase1_balanced/vm2/drift_prep.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Drift analysis metadata generated"
else
    echo "❌ Drift analysis metadata generation FAILED"
    exit 1
fi

# Cleanup and optimization
echo "⚡ Cleaning up temporary files..."
python scripts/auto_cleanup.py --target-dir logs/phase1_balanced/vm2 --keep-latest 5 > logs/phase1_balanced/vm2/cleanup.log 2>&1

echo "🎯 PHASE 1E COMPLETE: Infrastructure preparation finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM2 BALANCED WORKFLOW COMPLETE! $(date)"
echo "=========================================="
echo "✅ Phase 1A: Classification hyperparameter sweeps completed (6 sweeps)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified (3 YAML files)"
echo "✅ Phase 1C: Production experiments completed (18 experiments)"
echo "✅ Phase 1D: Base representations extracted (3 tasks)"
echo "✅ Phase 1E: Infrastructure preparation completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "📄 Task-specific optimal configs:"
echo "   • analysis/mrpc_optimal_hyperparameters.yaml"
echo "   • analysis/sst2_optimal_hyperparameters.yaml"
echo "   • analysis/rte_optimal_hyperparameters.yaml"
echo "🧠 Base representations: representations/base_model/{mrpc,sst2,rte}/"
echo "📋 Drift metadata: analysis/drift_analysis/drift_analysis_metadata.json"
echo "📋 VM2 results ready for Phase 2a analysis!"
echo ""
echo "🚀 INDEPENDENT EXECUTION: VM2 complete with NO dependencies on VM1!"
echo "⚖️ BALANCED WORKLOAD: Classification tasks (3x weight) perfectly balanced with VM1!"
# RTE LoRA fine-tuning with multiple seeds (LIGHT LOAD)
echo "🔬 [4/4] RTE LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA fine-tuning (seed $seed)..."
    if python experiments/lora_finetune.py --task rte --mode single --seed $seed > logs/phase1/vm2/rte_lora_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA fine-tuning (seed $seed) complete"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA fine-tuning (seed $seed) FAILED"
        echo "Check logs/phase1/vm2/rte_lora_seed${seed}.log for details"
        exit 1
    fi
done

echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA hyperparameter sweep..."
if python experiments/lora_finetune.py --task rte --mode sweep > logs/phase1/vm2/rte_lora_sweep.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - RTE LoRA hyperparameter sweep complete"
else
    echo "  ❌ $(date +'%H:%M') - RTE LoRA hyperparameter sweep FAILED"
    echo "Check logs/phase1/vm2/rte_lora_sweep.log for details"
    exit 1
fi
echo "🎯 [4/4] RTE LoRA Fine-tuning COMPLETE"

echo ""
echo "🎉 VM2 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"
for task in mrpc sst2 rte; do
    echo ""
    echo "${task^^} OPTIMAL HYPERPARAMETERS:"
    python -c "
import yaml
with open('analysis/${task}_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for task_name, methods in optimal_hp.items():
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
        print(f'  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Acc={acc:.3f}')
"
done

echo "🎯 PHASE 1B COMPLETE: All task-specific optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: PRODUCTION EXPERIMENTS (Using Task-Specific Optimal Hyperparameters)
# ============================================================================
echo "🚀 PHASE 1C: PRODUCTION EXPERIMENTS"
echo "Running classification experiments with task-specific optimal hyperparameters"
echo "------------------------------------------------------------"

# MRPC Production Experiments
echo "🎯 [1/9] MRPC Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/mrpc_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'mrpc' in optimal_hp and method in optimal_hp['mrpc']:
        hp = optimal_hp['mrpc'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export MRPC_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export MRPC_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export MRPC_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export MRPC_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export MRPC_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export MRPC_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export MRPC_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > mrpc_hyperparams.sh
source mrpc_hyperparams.sh

# MRPC Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_FULLFINETUNE_LR \
        --batch-size $MRPC_FULLFINETUNE_BS \
        --warmup-ratio $MRPC_FULLFINETUNE_WU \
        --epochs $MRPC_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/mrpc_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# MRPC LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_LORA_LR \
        --batch-size $MRPC_LORA_BS \
        --warmup-ratio $MRPC_LORA_WU \
        --epochs $MRPC_LORA_EP \
        --lora-r $MRPC_LORA_R \
        --lora-alpha $MRPC_LORA_A \
        --lora-dropout $MRPC_LORA_D \
        > logs/phase1_balanced/vm2/mrpc_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 Production Experiments
echo "🎯 [2/9] SST-2 Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/sst2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'sst2' in optimal_hp and method in optimal_hp['sst2']:
        hp = optimal_hp['sst2'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export SST2_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export SST2_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export SST2_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export SST2_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export SST2_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export SST2_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export SST2_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > sst2_hyperparams.sh
source sst2_hyperparams.sh

# SST-2 Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_FULLFINETUNE_LR \
        --batch-size $SST2_FULLFINETUNE_BS \
        --warmup-ratio $SST2_FULLFINETUNE_WU \
        --epochs $SST2_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/sst2_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_LORA_LR \
        --batch-size $SST2_LORA_BS \
        --warmup-ratio $SST2_LORA_WU \
        --epochs $SST2_LORA_EP \
        --lora-r $SST2_LORA_R \
        --lora-alpha $SST2_LORA_A \
        --lora-dropout $SST2_LORA_D \
        > logs/phase1_balanced/vm2/sst2_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# RTE Production Experiments
echo "🎯 [3/9] RTE Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'rte' in optimal_hp and method in optimal_hp['rte']:
        hp = optimal_hp['rte'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export RTE_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export RTE_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export RTE_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export RTE_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export RTE_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export RTE_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export RTE_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > rte_hyperparams.sh
source rte_hyperparams.sh

# RTE Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_FULLFINETUNE_LR \
        --batch-size $RTE_FULLFINETUNE_BS \
        --warmup-ratio $RTE_FULLFINETUNE_WU \
        --epochs $RTE_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/rte_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) FAILED"
    exit 1
fi
done

# RTE LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_LORA_LR \
        --batch-size $RTE_LORA_BS \
        --warmup-ratio $RTE_LORA_WU \
        --epochs $RTE_LORA_EP \
        --lora-r $RTE_LORA_R \
        --lora-alpha $RTE_LORA_A \
        --lora-dropout $RTE_LORA_D \
        > logs/phase1_balanced/vm2/rte_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 PHASE 1C COMPLETE: All classification production experiments finished!"
echo ""

# ============================================================================
# PHASE 1D: BASE REPRESENTATIONS (Classification Tasks)
# ============================================================================
echo "🧠 PHASE 1D: BASE REPRESENTATIONS EXTRACTION"
echo "Extracting base model representations for classification tasks"
echo "------------------------------------------------------------"

echo "⚡ [1/3] Extracting base model representations for MRPC..."
if python scripts/extract_base_representations.py --task mrpc --output-dir representations/base_model > logs/phase1_balanced/vm2/base_mrpc.log 2>&1; then
    echo "✅ MRPC base representations extracted"
else
    echo "❌ MRPC base representation extraction FAILED"
    exit 1
fi

echo "⚡ [2/3] Extracting base model representations for SST-2..."
if python scripts/extract_base_representations.py --task sst2 --output-dir representations/base_model > logs/phase1_balanced/vm2/base_sst2.log 2>&1; then
    echo "✅ SST-2 base representations extracted"
else
    echo "❌ SST-2 base representation extraction FAILED"
    exit 1
fi

echo "⚡ [3/3] Extracting base model representations for RTE..."
if python scripts/extract_base_representations.py --task rte --output-dir representations/base_model > logs/phase1_balanced/vm2/base_rte.log 2>&1; then
    echo "✅ RTE base representations extracted"
else
    echo "❌ RTE base representation extraction FAILED"
    exit 1
fi

echo "🎯 PHASE 1D COMPLETE: All classification base representations extracted!"
echo ""

# ============================================================================
# PHASE 1E: INFRASTRUCTURE PREPARATION
# ============================================================================
echo "📋 PHASE 1E: INFRASTRUCTURE PREPARATION"
echo "Preparing shared drift analysis infrastructure"
echo "------------------------------------------------------------"

# Create directory structure for drift analysis
mkdir -p analysis/drift_analysis/{base_model,full_finetune,lora}/{representations,metadata}

# Generate metadata for drift analysis
echo "⚡ Generating drift analysis metadata..."
python -c "
import json
import os
from datetime import datetime

# Create metadata for drift analysis
metadata = {
    'base_model': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'extraction_date': datetime.now().isoformat(),
        'tasks': ['mrpc', 'sst2', 'rte', 'squad_v2'],
        'representation_layers': list(range(22)),  # TinyLlama has 22 layers
        'representation_dir': 'representations/base_model/'
    },
    'analysis_config': {
        'drift_metrics': ['linear_cka', 'cosine_similarity'],
        'comparison_pairs': [
            ['base_model', 'full_finetune'],
            ['base_model', 'lora']
        ],
        'statistical_tests': ['permutation_test', 'bootstrap_ci'],
        'visualization': ['heatmaps', 'layer_evolution', 'correlation_plots']
    },
    'tasks_metadata': {
        'mrpc': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'sst2': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'}, 
        'rte': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'squad_v2': {'type': 'qa', 'metric': 'f1', 'vm': 'vm1'}
    },
    'vm_distribution': {
        'vm1': ['squad_v2'],
        'vm2': ['mrpc', 'sst2', 'rte']
    },
    'file_structure': {
        'optimal_hyperparameters': {
            'mrpc': 'analysis/mrpc_optimal_hyperparameters.yaml',
            'sst2': 'analysis/sst2_optimal_hyperparameters.yaml',
            'rte': 'analysis/rte_optimal_hyperparameters.yaml',
            'squad_v2': 'analysis/squad_v2_optimal_hyperparameters.yaml'
        }
    }
}

# Save metadata
os.makedirs('analysis/drift_analysis', exist_ok=True)
with open('analysis/drift_analysis/drift_analysis_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✅ Drift analysis metadata created')
" > logs/phase1_balanced/vm2/drift_prep.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Drift analysis metadata generated"
else
    echo "❌ Drift analysis metadata generation FAILED"
    exit 1
fi

# Cleanup and optimization
echo "⚡ Cleaning up temporary files..."
python scripts/auto_cleanup.py --target-dir logs/phase1_balanced/vm2 --keep-latest 5 > logs/phase1_balanced/vm2/cleanup.log 2>&1

echo "🎯 PHASE 1E COMPLETE: Infrastructure preparation finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM2 BALANCED WORKFLOW COMPLETE! $(date)"
echo "=========================================="
echo "✅ Phase 1A: Classification hyperparameter sweeps completed (6 sweeps)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified (3 YAML files)"
echo "✅ Phase 1C: Production experiments completed (18 experiments)"
echo "✅ Phase 1D: Base representations extracted (3 tasks)"
echo "✅ Phase 1E: Infrastructure preparation completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "📄 Task-specific optimal configs:"
echo "   • analysis/mrpc_optimal_hyperparameters.yaml"
echo "   • analysis/sst2_optimal_hyperparameters.yaml"
echo "   • analysis/rte_optimal_hyperparameters.yaml"
echo "🧠 Base representations: representations/base_model/{mrpc,sst2,rte}/"
echo "📋 Drift metadata: analysis/drift_analysis/drift_analysis_metadata.json"
echo "📋 VM2 results ready for Phase 2a analysis!"
echo ""
echo "🚀 INDEPENDENT EXECUTION: VM2 complete with NO dependencies on VM1!"
echo "⚖️ BALANCED WORKLOAD: Classification tasks (3x weight) perfectly balanced with VM1!"
for task in mrpc sst2 rte; do
    echo ""
    echo "${task^^} OPTIMAL HYPERPARAMETERS:"
    python -c "
import yaml
with open('analysis/${task}_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for task_name, methods in optimal_hp.items():
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
        print(f'  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Acc={acc:.3f}')
"
done

echo "🎯 PHASE 1B COMPLETE: All task-specific optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: PRODUCTION EXPERIMENTS (Using Task-Specific Optimal Hyperparameters)
# ============================================================================
echo "🚀 PHASE 1C: PRODUCTION EXPERIMENTS"
echo "Running classification experiments with task-specific optimal hyperparameters"
echo "------------------------------------------------------------"

# MRPC Production Experiments
echo "🎯 [1/9] MRPC Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/mrpc_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'mrpc' in optimal_hp and method in optimal_hp['mrpc']:
        hp = optimal_hp['mrpc'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export MRPC_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export MRPC_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export MRPC_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export MRPC_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export MRPC_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export MRPC_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export MRPC_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > mrpc_hyperparams.sh
source mrpc_hyperparams.sh

# MRPC Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_FULLFINETUNE_LR \
        --batch-size $MRPC_FULLFINETUNE_BS \
        --warmup-ratio $MRPC_FULLFINETUNE_WU \
        --epochs $MRPC_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/mrpc_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# MRPC LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_LORA_LR \
        --batch-size $MRPC_LORA_BS \
        --warmup-ratio $MRPC_LORA_WU \
        --epochs $MRPC_LORA_EP \
        --lora-r $MRPC_LORA_R \
        --lora-alpha $MRPC_LORA_A \
        --lora-dropout $MRPC_LORA_D \
        > logs/phase1_balanced/vm2/mrpc_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 Production Experiments
echo "🎯 [2/9] SST-2 Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/sst2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'sst2' in optimal_hp and method in optimal_hp['sst2']:
        hp = optimal_hp['sst2'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export SST2_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export SST2_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export SST2_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export SST2_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export SST2_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export SST2_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export SST2_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > sst2_hyperparams.sh
source sst2_hyperparams.sh

# SST-2 Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_FULLFINETUNE_LR \
        --batch-size $SST2_FULLFINETUNE_BS \
        --warmup-ratio $SST2_FULLFINETUNE_WU \
        --epochs $SST2_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/sst2_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_LORA_LR \
        --batch-size $SST2_LORA_BS \
        --warmup-ratio $SST2_LORA_WU \
        --epochs $SST2_LORA_EP \
        --lora-r $SST2_LORA_R \
        --lora-alpha $SST2_LORA_A \
        --lora-dropout $SST2_LORA_D \
        > logs/phase1_balanced/vm2/sst2_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# RTE Production Experiments
echo "🎯 [3/9] RTE Production Experiments (3 seeds)"
python -c "
import yaml
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if 'rte' in optimal_hp and method in optimal_hp['rte']:
        hp = optimal_hp['rte'][method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        print(f'export RTE_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export RTE_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export RTE_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export RTE_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        if method == 'lora':
            print(f'export RTE_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export RTE_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export RTE_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > rte_hyperparams.sh
source rte_hyperparams.sh

# RTE Full Fine-tuning
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE full fine-tuning (seed $seed)..."
    if python experiments/full_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_FULLFINETUNE_LR \
        --batch-size $RTE_FULLFINETUNE_BS \
        --warmup-ratio $RTE_FULLFINETUNE_WU \
        --epochs $RTE_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_balanced/vm2/rte_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

# RTE LoRA
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - RTE LoRA (seed $seed)..."
    if python experiments/lora_finetune.py \
        --task rte --mode single --seed $seed \
        --learning-rate $RTE_LORA_LR \
        --batch-size $RTE_LORA_BS \
        --warmup-ratio $RTE_LORA_WU \
        --epochs $RTE_LORA_EP \
        --lora-r $RTE_LORA_R \
        --lora-alpha $RTE_LORA_A \
        --lora-dropout $RTE_LORA_D \
        > logs/phase1_balanced/vm2/rte_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 PHASE 1C COMPLETE: All classification production experiments finished!"
echo ""

# ============================================================================
# PHASE 1D: BASE REPRESENTATIONS (Classification Tasks)
# ============================================================================
echo "🧠 PHASE 1D: BASE REPRESENTATIONS EXTRACTION"
echo "Extracting base model representations for classification tasks"
echo "------------------------------------------------------------"

echo "⚡ [1/3] Extracting base model representations for MRPC..."
if python scripts/extract_base_representations.py --task mrpc --output-dir representations/base_model > logs/phase1_balanced/vm2/base_mrpc.log 2>&1; then
    echo "✅ MRPC base representations extracted"
else
    echo "❌ MRPC base representation extraction FAILED"
    exit 1
fi

echo "⚡ [2/3] Extracting base model representations for SST-2..."
if python scripts/extract_base_representations.py --task sst2 --output-dir representations/base_model > logs/phase1_balanced/vm2/base_sst2.log 2>&1; then
    echo "✅ SST-2 base representations extracted"
else
    echo "❌ SST-2 base representation extraction FAILED"
    exit 1
fi

echo "⚡ [3/3] Extracting base model representations for RTE..."
if python scripts/extract_base_representations.py --task rte --output-dir representations/base_model > logs/phase1_balanced/vm2/base_rte.log 2>&1; then
    echo "✅ RTE base representations extracted"
else
    echo "❌ RTE base representation extraction FAILED"
    exit 1
fi

echo "🎯 PHASE 1D COMPLETE: All classification base representations extracted!"
echo ""

# ============================================================================
# PHASE 1E: INFRASTRUCTURE PREPARATION
# ============================================================================
echo "📋 PHASE 1E: INFRASTRUCTURE PREPARATION"
echo "Preparing shared drift analysis infrastructure"
echo "------------------------------------------------------------"

# Create directory structure for drift analysis
mkdir -p analysis/drift_analysis/{base_model,full_finetune,lora}/{representations,metadata}

# Generate metadata for drift analysis
echo "⚡ Generating drift analysis metadata..."
python -c "
import json
import os
from datetime import datetime

# Create metadata for drift analysis
metadata = {
    'base_model': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'extraction_date': datetime.now().isoformat(),
        'tasks': ['mrpc', 'sst2', 'rte', 'squad_v2'],
        'representation_layers': list(range(22)),  # TinyLlama has 22 layers
        'representation_dir': 'representations/base_model/'
    },
    'analysis_config': {
        'drift_metrics': ['linear_cka', 'cosine_similarity'],
        'comparison_pairs': [
            ['base_model', 'full_finetune'],
            ['base_model', 'lora']
        ],
        'statistical_tests': ['permutation_test', 'bootstrap_ci'],
        'visualization': ['heatmaps', 'layer_evolution', 'correlation_plots']
    },
    'tasks_metadata': {
        'mrpc': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'sst2': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'}, 
        'rte': {'type': 'classification', 'metric': 'accuracy', 'vm': 'vm2'},
        'squad_v2': {'type': 'qa', 'metric': 'f1', 'vm': 'vm1'}
    },
    'vm_distribution': {
        'vm1': ['squad_v2'],
        'vm2': ['mrpc', 'sst2', 'rte']
    },
    'file_structure': {
        'optimal_hyperparameters': {
            'mrpc': 'analysis/mrpc_optimal_hyperparameters.yaml',
            'sst2': 'analysis/sst2_optimal_hyperparameters.yaml',
            'rte': 'analysis/rte_optimal_hyperparameters.yaml',
            'squad_v2': 'analysis/squad_v2_optimal_hyperparameters.yaml'
        }
    }
}

# Save metadata
os.makedirs('analysis/drift_analysis', exist_ok=True)
with open('analysis/drift_analysis/drift_analysis_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✅ Drift analysis metadata created')
" > logs/phase1_balanced/vm2/drift_prep.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Drift analysis metadata generated"
else
    echo "❌ Drift analysis metadata generation FAILED"
    exit 1
fi

# Cleanup and optimization
echo "⚡ Cleaning up temporary files..."
python scripts/auto_cleanup.py --target-dir logs/phase1_balanced/vm2 --keep-latest 5 > logs/phase1_balanced/vm2/cleanup.log 2>&1

echo "🎯 PHASE 1E COMPLETE: Infrastructure preparation finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM2 BALANCED WORKFLOW COMPLETE! $(date)"
echo "=========================================="
echo "✅ Phase 1A: Classification hyperparameter sweeps completed (6 sweeps)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified (3 YAML files)"
echo "✅ Phase 1C: Production experiments completed (18 experiments)"
echo "✅ Phase 1D: Base representations extracted (3 tasks)"
echo "✅ Phase 1E: Infrastructure preparation completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "📄 Task-specific optimal configs:"
echo "   • analysis/mrpc_optimal_hyperparameters.yaml"
echo "   • analysis/sst2_optimal_hyperparameters.yaml"
echo "   • analysis/rte_optimal_hyperparameters.yaml"
echo "🧠 Base representations: representations/base_model/{mrpc,sst2,rte}/"
echo "📋 Drift metadata: analysis/drift_analysis/drift_analysis_metadata.json"
echo "📋 VM2 results ready for Phase 2a analysis!"
echo ""
echo "🚀 INDEPENDENT EXECUTION: VM2 complete with NO dependencies on VM1!"
echo "⚖️ BALANCED WORKLOAD: Classification tasks (3x weight) perfectly balanced with VM1!"
# RTE LoRA fine-tuning with multiple seeds (LIGHT LOAD)
echo "🔬 [4/4] RTE LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA fine-tuning (seed $seed)..."
    if python experiments/lora_finetune.py --task rte --mode single --seed $seed > logs/phase1/vm2/rte_lora_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - RTE LoRA fine-tuning (seed $seed) complete"
    else
        echo "  ❌ $(date +'%H:%M') - RTE LoRA fine-tuning (seed $seed) FAILED"
        echo "Check logs/phase1/vm2/rte_lora_seed${seed}.log for details"
        exit 1
    fi
done

echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA hyperparameter sweep..."
if python experiments/lora_finetune.py --task rte --mode sweep > logs/phase1/vm2/rte_lora_sweep.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - RTE LoRA hyperparameter sweep complete"
else
    echo "  ❌ $(date +'%H:%M') - RTE LoRA hyperparameter sweep FAILED"
    echo "Check logs/phase1/vm2/rte_lora_sweep.log for details"
    exit 1
fi
echo "🎯 [4/4] RTE LoRA Fine-tuning COMPLETE"

echo ""
echo "🎉 VM2 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"