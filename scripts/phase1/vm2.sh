#!/bin/bash
# Phase 1 - VM2: CLASSIFICATION TASKS OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# Academic-grade Bayesian optimization for MRPC + SST-2 + RTE (plan.md compliance)
set -e  # Exit on error

echo "🚀 PHASE 1 - VM2: CLASSIFICATION TASKS OPTUNA BAYESIAN OPTIMIZATION"
echo "=================================================================="
echo "Academic-grade hyperparameter optimization workflow:"
echo "1. Bayesian optimization (TPE) for MRPC + SST-2 + RTE (30 trials × 6 configs = 180 trials)"
echo "2. Optimal hyperparameter extraction"
echo "3. Production experiments using optimal hyperparameters"
echo "4. Shared infrastructure setup for all VMs"
echo "=================================================================="

# Setup environment
# Auto-detect workspace directory (works on any VM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

export WANDB_PROJECT=NLP-Phase1-Optuna
export WANDB_ENTITY=galavny-tel-aviv-university

echo "🔧 Running on workspace: $WORKSPACE_DIR"

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
mkdir -p logs/phase1_optuna/vm2
mkdir -p analysis

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# PHASE 1A: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
echo "🔬 PHASE 1A: OPTUNA BAYESIAN OPTIMIZATION"
echo "Find optimal hyperparameters using Tree-structured Parzen Estimator (TPE)"
echo "30 trials per task/method combination (research-efficient vs 100+ for grid search)"
echo "Classification tasks: MRPC, SST-2, RTE (lighter computational load vs SQuAD v2)"
echo "------------------------------------------------------------"

# MRPC Optimization
echo "⚡ [1/6] MRPC Full Fine-tuning Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task mrpc \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/mrpc_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm2/mrpc_full_optuna.log 2>&1; then
    echo "✅ MRPC full fine-tuning optimization completed (30 trials)"
else
    echo "❌ MRPC full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [2/6] MRPC LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task mrpc \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/mrpc_lora_optimal.yaml \
    > logs/phase1_optuna/vm2/mrpc_lora_optuna.log 2>&1; then
    echo "✅ MRPC LoRA optimization completed (30 trials)"
else
    echo "❌ MRPC LoRA optimization FAILED"
    exit 1
fi

# SST-2 Optimization
echo "⚡ [3/6] SST-2 Full Fine-tuning Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task sst2 \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/sst2_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm2/sst2_full_optuna.log 2>&1; then
    echo "✅ SST-2 full fine-tuning optimization completed (30 trials)"
else
    echo "❌ SST-2 full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [4/6] SST-2 LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task sst2 \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/sst2_lora_optimal.yaml \
    > logs/phase1_optuna/vm2/sst2_lora_optuna.log 2>&1; then
    echo "✅ SST-2 LoRA optimization completed (30 trials)"
else
    echo "❌ SST-2 LoRA optimization FAILED"
    exit 1
fi

# RTE Optimization  
echo "⚡ [5/6] RTE Full Fine-tuning Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task rte \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/rte_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm2/rte_full_optuna.log 2>&1; then
    echo "✅ RTE full fine-tuning optimization completed (30 trials)"
else
    echo "❌ RTE full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [6/6] RTE LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task rte \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/rte_lora_optimal.yaml \
    > logs/phase1_optuna/vm2/rte_lora_optuna.log 2>&1; then
    echo "✅ RTE LoRA optimization completed (30 trials)"
else
    echo "❌ RTE LoRA optimization FAILED"
    exit 1
fi

echo "🎯 PHASE 1A COMPLETE: All VM2 classification Optuna optimizations finished!"
echo "Total trials: 180 (6 × 30 trials with TPE sampler + median pruning)"
echo ""

# ============================================================================
# PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION
# ============================================================================
echo "📊 PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION"
echo "Consolidating VM2 classification Optuna results into task-specific optimal configurations"
echo "------------------------------------------------------------"

# Create task-specific optimal hyperparameters files
echo "⚡ Creating task-specific optimal hyperparameter files..."

# Process MRPC results
python -c "
import yaml
from pathlib import Path

# Process MRPC results
mrpc_files = [
    'analysis/mrpc_full_finetune_optimal.yaml',
    'analysis/mrpc_lora_optimal.yaml'
]

mrpc_config = {
    'task': 'mrpc',
    'optimization_method': 'optuna_tpe',
    'total_trials': 60,
    'trials_per_method': 30,
    'optimal_hyperparameters': {}
}

for file_path in mrpc_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        method = config['method']
        mrpc_config['optimal_hyperparameters'][method] = {
            'hyperparameters': config['best_hyperparameters'],
            'expected_performance': config['expected_performance'],
            'optimization_summary': config['optimization_summary']
        }
        print(f'✅ Loaded optimal hyperparameters for MRPC {method}')

# Save MRPC-specific config
with open('analysis/mrpc_optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(mrpc_config, f, default_flow_style=False)
print('📄 MRPC optimal hyperparameters saved')
"

# Process SST-2 results
python -c "
import yaml
from pathlib import Path

# Process SST-2 results
sst2_files = [
    'analysis/sst2_full_finetune_optimal.yaml',
    'analysis/sst2_lora_optimal.yaml'
]

sst2_config = {
    'task': 'sst2',
    'optimization_method': 'optuna_tpe',
    'total_trials': 60,
    'trials_per_method': 30,
    'optimal_hyperparameters': {}
}

for file_path in sst2_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        method = config['method']
        sst2_config['optimal_hyperparameters'][method] = {
            'hyperparameters': config['best_hyperparameters'],
            'expected_performance': config['expected_performance'],
            'optimization_summary': config['optimization_summary']
        }
        print(f'✅ Loaded optimal hyperparameters for SST-2 {method}')

# Save SST-2-specific config
with open('analysis/sst2_optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(sst2_config, f, default_flow_style=False)
print('📄 SST-2 optimal hyperparameters saved')
"

# Process RTE results
python -c "
import yaml
from pathlib import Path

# Process RTE results
rte_files = [
    'analysis/rte_full_finetune_optimal.yaml',
    'analysis/rte_lora_optimal.yaml'
]

rte_config = {
    'task': 'rte',
    'optimization_method': 'optuna_tpe',
    'total_trials': 60,
    'trials_per_method': 30,
    'optimal_hyperparameters': {}
}

for file_path in rte_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        method = config['method']
        rte_config['optimal_hyperparameters'][method] = {
            'hyperparameters': config['best_hyperparameters'],
            'expected_performance': config['expected_performance'],
            'optimization_summary': config['optimization_summary']
        }
        print(f'✅ Loaded optimal hyperparameters for RTE {method}')

# Save RTE-specific config
with open('analysis/rte_optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(rte_config, f, default_flow_style=False)
print('📄 RTE optimal hyperparameters saved')
"

# Verify optimal configurations were generated
if [ ! -f "analysis/mrpc_optimal_hyperparameters.yaml" ] || [ ! -f "analysis/sst2_optimal_hyperparameters.yaml" ] || [ ! -f "analysis/rte_optimal_hyperparameters.yaml" ]; then
    echo "❌ Task-specific optimal hyperparameters files not found!"
    exit 1
fi

echo "📋 OPTIMAL HYPERPARAMETERS IDENTIFIED (from Optuna TPE):"
echo "------------------------------------------------------------"

echo "🎯 MRPC (Paraphrase Detection):"
python -c "
import yaml
with open('analysis/mrpc_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

print(f'Optimization: {config[\"optimization_method\"]} ({config[\"total_trials\"]} total trials)')
optimal_hp = config['optimal_hyperparameters']
for method, info in optimal_hp.items():
    hp = info['hyperparameters']
    perf = info['expected_performance']
    summary = info['optimization_summary']
    
    lr = hp['learning_rate']
    bs = hp['per_device_train_batch_size']
    wu = hp['warmup_ratio']
    ep = hp['num_train_epochs']
    
    print(f'  {method:15}: LR={lr:.2e} BS={bs:2} WU={wu:.2f} EP={ep} → F1={perf:.3f}')
    print(f'                   Trials: {summary[\"n_completed\"]}/{summary[\"n_trials\"]} completed, {summary[\"n_pruned\"]} pruned')
    
    if method == 'lora':
        r = hp.get('lora_r', 'N/A')
        alpha = hp.get('lora_alpha', 'N/A')
        dropout = hp.get('lora_dropout', 'N/A')
        print(f'                   LoRA: r={r}, α={alpha}, dropout={dropout:.3f}')
"

echo ""
echo "🎯 SST-2 (Sentiment Analysis):"
python -c "
import yaml
with open('analysis/sst2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

optimal_hp = config['optimal_hyperparameters']
for method, info in optimal_hp.items():
    hp = info['hyperparameters']
    perf = info['expected_performance']
    summary = info['optimization_summary']
    
    lr = hp['learning_rate']
    bs = hp['per_device_train_batch_size']
    wu = hp['warmup_ratio']
    ep = hp['num_train_epochs']
    
    print(f'  {method:15}: LR={lr:.2e} BS={bs:2} WU={wu:.2f} EP={ep} → Acc={perf:.3f}')
    print(f'                   Trials: {summary[\"n_completed\"]}/{summary[\"n_trials\"]} completed, {summary[\"n_pruned\"]} pruned')
    
    if method == 'lora':
        r = hp.get('lora_r', 'N/A')
        alpha = hp.get('lora_alpha', 'N/A')
        dropout = hp.get('lora_dropout', 'N/A')
        print(f'                   LoRA: r={r}, α={alpha}, dropout={dropout:.3f}')
"

echo ""
echo "🎯 RTE (Textual Entailment):"
python -c "
import yaml
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

optimal_hp = config['optimal_hyperparameters']
for method, info in optimal_hp.items():
    hp = info['hyperparameters']
    perf = info['expected_performance']
    summary = info['optimization_summary']
    
    lr = hp['learning_rate']
    bs = hp['per_device_train_batch_size']
    wu = hp['warmup_ratio']
    ep = hp['num_train_epochs']
    
    print(f'  {method:15}: LR={lr:.2e} BS={bs:2} WU={wu:.2f} EP={ep} → Acc={perf:.3f}')
    print(f'                   Trials: {summary[\"n_completed\"]}/{summary[\"n_trials\"]} completed, {summary[\"n_pruned\"]} pruned')
    
    if method == 'lora':
        r = hp.get('lora_r', 'N/A')
        alpha = hp.get('lora_alpha', 'N/A')
        dropout = hp.get('lora_dropout', 'N/A')
        print(f'                   LoRA: r={r}, α={alpha}, dropout={dropout:.3f}')
"

echo ""
echo "🎯 PHASE 1B COMPLETE: Task-specific optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: HYPERPARAMETER VALIDATION (Quick Test)
# ============================================================================
echo "🧪 PHASE 1C: HYPERPARAMETER VALIDATION"
echo "Quick validation test with optimal hyperparameters (single seed)"
echo "------------------------------------------------------------"

# Extract optimal hyperparameters for all tasks
python -c "
import yaml

# Extract hyperparameters for all tasks
tasks = ['mrpc', 'sst2', 'rte']
for task in tasks:
    with open(f'analysis/{task}_optimal_hyperparameters.yaml') as f:
        config = yaml.safe_load(f)
    
    optimal_hp = config['optimal_hyperparameters']
    for method in ['full_finetune', 'lora']:
        if method in optimal_hp:
            hp = optimal_hp[method]['hyperparameters']
            task_upper = task.upper()
            method_upper = method.upper().replace('_', '')
            
            print(f'export {task_upper}_{method_upper}_LR={hp[\"learning_rate\"]}')
            print(f'export {task_upper}_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
            print(f'export {task_upper}_{method_upper}_WU={hp[\"warmup_ratio\"]}')
            print(f'export {task_upper}_{method_upper}_EP={hp[\"num_train_epochs\"]}')
            print(f'export {task_upper}_{method_upper}_WD={hp.get(\"weight_decay\", 0.01)}')
            
            if method == 'lora':
                print(f'export {task_upper}_{method_upper}_R={hp.get(\"lora_r\", 8)}')
                print(f'export {task_upper}_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
                print(f'export {task_upper}_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > classification_hyperparams.sh

source classification_hyperparams.sh

# Change wandb project for production runs
export WANDB_PROJECT=NLP-Phase1-Production

# MRPC Validation Tests
echo "🎯 [1/6] MRPC Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$MRPC_FULLFINETUNE_LR, BS=$MRPC_FULLFINETUNE_BS, WU=$MRPC_FULLFINETUNE_WU, EP=$MRPC_FULLFINETUNE_EP"

if python experiments/full_finetune.py \
    --task mrpc --mode single --seed 42 \
    --learning-rate $MRPC_FULLFINETUNE_LR \
    --batch-size $MRPC_FULLFINETUNE_BS \
    --warmup-ratio $MRPC_FULLFINETUNE_WU \
    --epochs $MRPC_FULLFINETUNE_EP \
    --weight-decay $MRPC_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm2/mrpc_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [2/6] MRPC LoRA Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - MRPC LoRA validation with Optuna hyperparameters..."
echo "    LR=$MRPC_LORA_LR, BS=$MRPC_LORA_BS, WU=$MRPC_LORA_WU, R=$MRPC_LORA_R, α=$MRPC_LORA_A"

if python experiments/lora_finetune.py \
    --task mrpc --mode single --seed 42 \
    --learning-rate $MRPC_LORA_LR \
    --batch-size $MRPC_LORA_BS \
    --warmup-ratio $MRPC_LORA_WU \
    --epochs $MRPC_LORA_EP \
    --weight-decay $MRPC_LORA_WD \
    --lora-r $MRPC_LORA_R \
    --lora-alpha $MRPC_LORA_A \
    --lora-dropout $MRPC_LORA_D \
    > logs/phase1_optuna/vm2/mrpc_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - MRPC LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - MRPC LoRA validation FAILED"
    exit 1
fi

# SST-2 Validation Tests
echo "🎯 [3/6] SST-2 Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$SST2_FULLFINETUNE_LR, BS=$SST2_FULLFINETUNE_BS, WU=$SST2_FULLFINETUNE_WU, EP=$SST2_FULLFINETUNE_EP"

if python experiments/full_finetune.py \
    --task sst2 --mode single --seed 42 \
    --learning-rate $SST2_FULLFINETUNE_LR \
    --batch-size $SST2_FULLFINETUNE_BS \
    --warmup-ratio $SST2_FULLFINETUNE_WU \
    --epochs $SST2_FULLFINETUNE_EP \
    --weight-decay $SST2_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm2/sst2_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [4/6] SST-2 LoRA Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA validation with Optuna hyperparameters..."
echo "    LR=$SST2_LORA_LR, BS=$SST2_LORA_BS, WU=$SST2_LORA_WU, R=$SST2_LORA_R, α=$SST2_LORA_A"

if python experiments/lora_finetune.py \
    --task sst2 --mode single --seed 42 \
    --learning-rate $SST2_LORA_LR \
    --batch-size $SST2_LORA_BS \
    --warmup-ratio $SST2_LORA_WU \
    --epochs $SST2_LORA_EP \
    --weight-decay $SST2_LORA_WD \
    --lora-r $SST2_LORA_R \
    --lora-alpha $SST2_LORA_A \
    --lora-dropout $SST2_LORA_D \
    > logs/phase1_optuna/vm2/sst2_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SST-2 LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SST-2 LoRA validation FAILED"
    exit 1
fi

# RTE Validation Tests
echo "🎯 [5/6] RTE Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - RTE full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$RTE_FULLFINETUNE_LR, BS=$RTE_FULLFINETUNE_BS, WU=$RTE_FULLFINETUNE_WU, EP=$RTE_FULLFINETUNE_EP"

if python experiments/full_finetune.py \
    --task rte --mode single --seed 42 \
    --learning-rate $RTE_FULLFINETUNE_LR \
    --batch-size $RTE_FULLFINETUNE_BS \
    --warmup-ratio $RTE_FULLFINETUNE_WU \
    --epochs $RTE_FULLFINETUNE_EP \
    --weight-decay $RTE_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm2/rte_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - RTE full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [6/6] RTE LoRA Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - RTE LoRA validation with Optuna hyperparameters..."
echo "    LR=$RTE_LORA_LR, BS=$RTE_LORA_BS, WU=$RTE_LORA_WU, R=$RTE_LORA_R, α=$RTE_LORA_A"

if python experiments/lora_finetune.py \
    --task rte --mode single --seed 42 \
    --learning-rate $RTE_LORA_LR \
    --batch-size $RTE_LORA_BS \
    --warmup-ratio $RTE_LORA_WU \
    --epochs $RTE_LORA_EP \
    --weight-decay $RTE_LORA_WD \
    --lora-r $RTE_LORA_R \
    --lora-alpha $RTE_LORA_A \
    --lora-dropout $RTE_LORA_D \
    > logs/phase1_optuna/vm2/rte_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - RTE LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - RTE LoRA validation FAILED"
    exit 1
fi

echo "🎯 PHASE 1C COMPLETE: All VM2 classification hyperparameter validations finished!"
echo ""

# ============================================================================
# PHASE 1D: SHARED INFRASTRUCTURE SETUP
# ============================================================================
echo "🏗️ PHASE 1D: SHARED INFRASTRUCTURE SETUP"
echo "Preparing shared analysis infrastructure for all VMs"
echo "------------------------------------------------------------"

# Create shared directories
mkdir -p results/shared_analysis
mkdir -p results/phase1_summaries

# Extract base model representations for drift analysis (if not done in Phase 0)
if [ ! -d "results/base_model_representations" ]; then
    echo "⚡ Extracting base model representations for drift analysis..."
    python scripts/extract_base_representations.py --tasks all > logs/phase1_optuna/vm2/base_representations.log 2>&1
    echo "✅ Base model representations extracted"
else
    echo "✅ Base model representations already available from Phase 0"
fi

# Consolidate all optimal hyperparameters from both VMs into unified file
echo "⚡ Creating unified optimal hyperparameters summary..."
python -c "
import yaml
from pathlib import Path

# Files from both VMs
hyperparams_files = [
    'analysis/squad_v2_optimal_hyperparameters.yaml',  # VM1 SQuAD v2
    'analysis/mrpc_optimal_hyperparameters.yaml',     # VM2 MRPC
    'analysis/sst2_optimal_hyperparameters.yaml',     # VM2 SST-2
    'analysis/rte_optimal_hyperparameters.yaml'       # VM2 RTE
]

unified_summary = {
    'phase1_summary': {
        'optimization_method': 'optuna_tpe_bayesian',
        'total_trials_across_all_vms': 240,  # 2×30 VM1 + 6×30 VM2
        'trials_per_task_method': 30,
        'vm_distribution': {
            'vm1': ['squad_v2'],
            'vm2': ['mrpc', 'sst2', 'rte']
        }
    },
    'task_optimal_hyperparameters': {}
}

# Process each file
for file_path in hyperparams_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        # All files are in single task format
        task = config['task']
        unified_summary['task_optimal_hyperparameters'][task] = config['optimal_hyperparameters']
        
        print(f'✅ Processed: {file_path}')

# Save unified summary
with open('results/phase1_summaries/all_optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(unified_summary, f, default_flow_style=False)

print('📄 Unified optimal hyperparameters saved to: results/phase1_summaries/all_optimal_hyperparameters.yaml')
"

echo "📊 Creating Phase 1 completion summary..."
python -c "
import yaml
from pathlib import Path
from datetime import datetime

summary = {
    'phase1_completion': {
        'completed_at': datetime.now().isoformat(),
        'vm2_tasks': ['mrpc', 'sst2', 'rte'],
        'optimization_method': 'optuna_tpe',
        'total_trials_vm2': 180,
        'infrastructure_setup': 'completed',
        'next_phase': 'phase2_production_with_representations'
    }
}

with open('results/phase1_summaries/vm2_completion.yaml', 'w') as f:
    yaml.dump(summary, f, default_flow_style=False)

print('📄 VM2 completion summary saved')
"

echo "✅ PHASE 1D COMPLETE: Shared infrastructure ready!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM2 CLASSIFICATION OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETE! $(date)"
echo "========================================================================"
echo "✅ Phase 1A: Bayesian optimization completed (180 trials VM2)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified"
echo "✅ Phase 1C: Hyperparameter validation completed (6 validation tests)"
echo "✅ Phase 1D: Shared infrastructure setup completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Optuna"
echo "📄 Task configs: analysis/mrpc_optimal_hyperparameters.yaml"
echo "📄 Task configs: analysis/sst2_optimal_hyperparameters.yaml"
echo "📄 Task configs: analysis/rte_optimal_hyperparameters.yaml"
echo "📄 Unified summary: results/phase1_summaries/all_optimal_hyperparameters.yaml"
echo "📋 Ready for Phase 2: Production experiments with optimal hyperparameters"
echo ""
echo "🧠 ACADEMIC RIGOR: Bayesian optimization (TPE) > Grid search efficiency"
echo "   • 30 trials/config vs 100+ for grid search"
echo "   • Median pruning eliminates poor trials early"
echo "   • Quick validation confirms hyperparameters work"
echo "   • Classification tasks optimized independently"
echo "   • Full production experiments moved to Phase 2"