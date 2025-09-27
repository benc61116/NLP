#!/bin/bash
# Phase 1 - VM1: OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# Academic-grade Bayesian optimization: More efficient than grid search, better than random search
set -e  # Exit on error

echo "🚀 PHASE 1 - VM1: OPTUNA BAYESIAN OPTIMIZATION"
echo "==============================================="
echo "Academic-grade hyperparameter optimization workflow:"
echo "1. Bayesian optimization (TPE) for MRPC + SST-2 (30 trials each)"
echo "2. Optimal hyperparameter extraction"  
echo "3. Production experiments using optimal hyperparameters"
echo "==============================================="

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
mkdir -p logs/phase1_optuna/vm1
mkdir -p analysis

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# PHASE 1A: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
echo "🔬 PHASE 1A: OPTUNA BAYESIAN OPTIMIZATION"
echo "Find optimal hyperparameters using Tree-structured Parzen Estimator (TPE)"
echo "30 trials per task/method combination (research-efficient vs 100+ for grid search)"
echo "------------------------------------------------------------"

# MRPC Optimization
echo "⚡ [1/4] MRPC Full Fine-tuning Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task mrpc \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/mrpc_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm1/mrpc_full_optuna.log 2>&1; then
    echo "✅ MRPC full fine-tuning optimization completed (30 trials)"
else
    echo "❌ MRPC full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [2/4] MRPC LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task mrpc \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/mrpc_lora_optimal.yaml \
    > logs/phase1_optuna/vm1/mrpc_lora_optuna.log 2>&1; then
    echo "✅ MRPC LoRA optimization completed (30 trials)"
else
    echo "❌ MRPC LoRA optimization FAILED"
    exit 1
fi

# SST-2 Optimization
echo "⚡ [3/4] SST-2 Full Fine-tuning Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task sst2 \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/sst2_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm1/sst2_full_optuna.log 2>&1; then
    echo "✅ SST-2 full fine-tuning optimization completed (30 trials)"
else
    echo "❌ SST-2 full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [4/4] SST-2 LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task sst2 \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/sst2_lora_optimal.yaml \
    > logs/phase1_optuna/vm1/sst2_lora_optuna.log 2>&1; then
    echo "✅ SST-2 LoRA optimization completed (30 trials)"
else
    echo "❌ SST-2 LoRA optimization FAILED"
    exit 1
fi

echo "🎯 PHASE 1A COMPLETE: All Optuna optimizations finished!"
echo "Total trials: 120 (4 × 30 trials with TPE sampler + median pruning)"
echo ""

# ============================================================================
# PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION
# ============================================================================
echo "📊 PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION"
echo "Consolidating Optuna results into unified optimal configuration"
echo "------------------------------------------------------------"

# Create unified optimal hyperparameters file
echo "⚡ Consolidating optimal hyperparameters from Optuna results..."

python -c "
import yaml
from pathlib import Path

# Load individual Optuna results
optuna_files = [
    'analysis/mrpc_full_finetune_optimal.yaml',
    'analysis/mrpc_lora_optimal.yaml', 
    'analysis/sst2_full_finetune_optimal.yaml',
    'analysis/sst2_lora_optimal.yaml'
]

# Create unified structure
unified_config = {
    'optimization_method': 'optuna_tpe',
    'total_trials': 120,
    'trials_per_config': 30,
    'optimal_hyperparameters': {}
}

for file_path in optuna_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        task = config['task']
        method = config['method']
        
        # Initialize task if not exists
        if task not in unified_config['optimal_hyperparameters']:
            unified_config['optimal_hyperparameters'][task] = {}
        
        # Store optimal hyperparameters and performance
        unified_config['optimal_hyperparameters'][task][method] = {
            'hyperparameters': config['best_hyperparameters'],
            'expected_performance': config['expected_performance'],
            'optimization_summary': config['optimization_summary']
        }
        
        print(f'✅ Loaded optimal hyperparameters for {task} {method}')
    else:
        print(f'❌ Missing file: {file_path}')

# Save unified configuration
with open('analysis/optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(unified_config, f, default_flow_style=False)

print('📄 Unified optimal hyperparameters saved to: analysis/optimal_hyperparameters.yaml')
"

# Verify optimal configurations were generated
if [ ! -f "analysis/optimal_hyperparameters.yaml" ]; then
    echo "❌ Optimal hyperparameters file not found!"
    exit 1
fi

echo "📋 OPTIMAL HYPERPARAMETERS IDENTIFIED (from Optuna TPE):"
echo "------------------------------------------------------------"
python -c "
import yaml
with open('analysis/optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

print(f'Optimization: {config[\"optimization_method\"]} ({config[\"total_trials\"]} total trials)')
print('')

optimal_hp = config['optimal_hyperparameters']
for task, methods in optimal_hp.items():
    print(f'{task.upper()}:')
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        summary = info['optimization_summary']
        
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        
        print(f'  {method:15}: LR={lr:.2e} BS={bs:2} WU={wu:.2f} → Perf={perf:.3f}')
        print(f'                   Trials: {summary[\"n_completed\"]}/{summary[\"n_trials\"]} completed, {summary[\"n_pruned\"]} pruned')
        
        if method == 'lora':
            r = hp.get('lora_r', 'N/A')
            alpha = hp.get('lora_alpha', 'N/A')
            dropout = hp.get('lora_dropout', 'N/A')
            print(f'                   LoRA: r={r}, α={alpha}, dropout={dropout:.3f}')
        print('')
"

echo "🎯 PHASE 1B COMPLETE: Optimal hyperparameters identified via Bayesian optimization!"
echo ""

# ============================================================================
# PHASE 1C: HYPERPARAMETER VALIDATION (Quick Test)
# ============================================================================
echo "🧪 PHASE 1C: HYPERPARAMETER VALIDATION"
echo "Quick validation test with optimal hyperparameters (single seed)"
echo "------------------------------------------------------------"

# Extract optimal hyperparameters for each task/method combination
python -c "
import yaml
import json

with open('analysis/optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

optimal_hp = config['optimal_hyperparameters']

# Create shell variables for each task/method combination
for task in ['mrpc', 'sst2']:
    for method in ['full_finetune', 'lora']:
        if task in optimal_hp and method in optimal_hp[task]:
            hp = optimal_hp[task][method]['hyperparameters']
            
            # Create variable names
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
" > optimal_hyperparams.sh

source optimal_hyperparams.sh

# Change wandb project for production runs
export WANDB_PROJECT=NLP-Phase1-Production

# MRPC Validation Test with Optuna-Optimized Hyperparameters
echo "🎯 [1/4] MRPC Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$MRPC_FULLFINETUNE_LR, BS=$MRPC_FULLFINETUNE_BS, WU=$MRPC_FULLFINETUNE_WU, WD=$MRPC_FULLFINETUNE_WD"

if python experiments/full_finetune.py \
    --task mrpc --mode single --seed 42 \
    --learning-rate $MRPC_FULLFINETUNE_LR \
    --batch-size $MRPC_FULLFINETUNE_BS \
    --warmup-ratio $MRPC_FULLFINETUNE_WU \
    --epochs $MRPC_FULLFINETUNE_EP \
    --weight-decay $MRPC_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm1/mrpc_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [2/4] MRPC LoRA Validation (1 seed, quick test)"
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
    > logs/phase1_optuna/vm1/mrpc_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - MRPC LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - MRPC LoRA validation FAILED"
    exit 1
fi

# SST-2 Validation Tests with Optuna-Optimized Hyperparameters
echo "🎯 [3/4] SST-2 Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$SST2_FULLFINETUNE_LR, BS=$SST2_FULLFINETUNE_BS, WU=$SST2_FULLFINETUNE_WU, WD=$SST2_FULLFINETUNE_WD"

if python experiments/full_finetune.py \
    --task sst2 --mode single --seed 42 \
    --learning-rate $SST2_FULLFINETUNE_LR \
    --batch-size $SST2_FULLFINETUNE_BS \
    --warmup-ratio $SST2_FULLFINETUNE_WU \
    --epochs $SST2_FULLFINETUNE_EP \
    --weight-decay $SST2_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm1/sst2_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [4/4] SST-2 LoRA Validation (1 seed, quick test)"
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
    > logs/phase1_optuna/vm1/sst2_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SST-2 LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SST-2 LoRA validation FAILED"
    exit 1
fi

echo "🎯 PHASE 1C COMPLETE: All hyperparameter validations finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM1 OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETE! $(date)"
echo "========================================================="
echo "✅ Phase 1A: Bayesian optimization completed (120 trials total)"
echo "✅ Phase 1B: Optimal hyperparameters identified via TPE"
echo "✅ Phase 1C: Hyperparameter validation completed (4 validation tests)"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Optuna"
echo "📄 Optimal configs: analysis/optimal_hyperparameters.yaml"
echo "📋 Ready for Phase 2: Production experiments with optimal hyperparameters"
echo ""
echo "🧠 ACADEMIC RIGOR: Bayesian optimization (TPE) > Grid search efficiency"
echo "   • 30 trials/config vs 100+ for grid search"
echo "   • Median pruning eliminates poor trials early"
echo "   • Quick validation confirms hyperparameters work"
echo "   • Full production experiments moved to Phase 2"
