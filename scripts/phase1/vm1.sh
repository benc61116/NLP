#!/bin/bash
# Phase 1 - VM1: SQuAD v2 OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# Academic-grade Bayesian optimization for SQuAD v2 only (plan.md compliance)
set -e  # Exit on error

echo "🚀 PHASE 1 - VM1: SQuAD v2 OPTUNA BAYESIAN OPTIMIZATION"
echo "========================================================"
echo "Academic-grade hyperparameter optimization workflow:"
echo "1. Bayesian optimization (TPE) for SQuAD v2 (30 trials × 2 methods = 60 trials)"
echo "2. Optimal hyperparameter extraction"  
echo "3. Production experiments using optimal hyperparameters"
echo "========================================================"

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
echo "30 trials per method (research-efficient vs 100+ for grid search)"
echo "SQuAD v2 focus: 3x computational weight vs classification tasks"
echo "------------------------------------------------------------"

# SQuAD v2 Optimization (Heavy computational load)
echo "⚡ [1/2] SQuAD v2 Full Fine-tuning Optimization (30 trials)"
echo "   🔥 Note: SQuAD v2 has 3x computational weight vs classification"
if python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/squad_v2_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm1/squad_v2_full_optuna.log 2>&1; then
    echo "✅ SQuAD v2 full fine-tuning optimization completed (30 trials)"
else
    echo "❌ SQuAD v2 full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [2/2] SQuAD v2 LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/squad_v2_lora_optimal.yaml \
    > logs/phase1_optuna/vm1/squad_v2_lora_optuna.log 2>&1; then
    echo "✅ SQuAD v2 LoRA optimization completed (30 trials)"
else
    echo "❌ SQuAD v2 LoRA optimization FAILED"
    exit 1
fi

echo "🎯 PHASE 1A COMPLETE: All VM1 Optuna optimizations finished!"
echo "Total trials: 60 (2 × 30 trials with TPE sampler + median pruning)"
echo ""

# ============================================================================
# PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION
# ============================================================================
echo "📊 PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION"
echo "Consolidating SQuAD v2 Optuna results into task-specific optimal configuration"
echo "------------------------------------------------------------"

# Create SQuAD v2-specific optimal hyperparameters file
echo "⚡ Creating SQuAD v2-specific optimal hyperparameter file..."

python -c "
import yaml
from pathlib import Path

# Load SQuAD v2 results
squad_files = [
    'analysis/squad_v2_full_finetune_optimal.yaml',
    'analysis/squad_v2_lora_optimal.yaml'
]

squad_config = {
    'task': 'squad_v2',
    'optimization_method': 'optuna_tpe',
    'total_trials': 60,
    'trials_per_method': 30,
    'optimal_hyperparameters': {}
}

for file_path in squad_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        method = config['method']
        squad_config['optimal_hyperparameters'][method] = {
            'hyperparameters': config['best_hyperparameters'],
            'expected_performance': config['expected_performance'],
            'optimization_summary': config['optimization_summary']
        }
        print(f'✅ Loaded optimal hyperparameters for SQuAD v2 {method}')

# Save SQuAD v2-specific config
with open('analysis/squad_v2_optimal_hyperparameters.yaml', 'w') as f:
    yaml.dump(squad_config, f, default_flow_style=False)
print('📄 SQuAD v2 optimal hyperparameters saved')
"

# Verify optimal configurations were generated
if [ ! -f "analysis/squad_v2_optimal_hyperparameters.yaml" ]; then
    echo "❌ SQuAD v2 optimal hyperparameters file not found!"
    exit 1
fi

echo "📋 OPTIMAL HYPERPARAMETERS IDENTIFIED (from Optuna TPE):"
echo "------------------------------------------------------------"

echo "🎯 SQuAD v2 (Question Answering):"
python -c "
import yaml
with open('analysis/squad_v2_optimal_hyperparameters.yaml') as f:
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
echo "🎯 PHASE 1B COMPLETE: SQuAD v2 optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: HYPERPARAMETER VALIDATION (Quick Test)
# ============================================================================
echo "🧪 PHASE 1C: HYPERPARAMETER VALIDATION"
echo "Quick validation test with optimal hyperparameters (single seed)"
echo "------------------------------------------------------------"

# Extract optimal hyperparameters for SQuAD v2
python -c "
import yaml

# SQuAD v2 hyperparameters
with open('analysis/squad_v2_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if method in optimal_hp:
        hp = optimal_hp[method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        
        print(f'export SQUADV2_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export SQUADV2_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export SQUADV2_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export SQUADV2_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        print(f'export SQUADV2_{method_upper}_WD={hp.get(\"weight_decay\", 0.01)}')
        
        if method == 'lora':
            print(f'export SQUADV2_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export SQUADV2_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export SQUADV2_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > squad_v2_hyperparams.sh

source squad_v2_hyperparams.sh

# Change wandb project for production runs
export WANDB_PROJECT=NLP-Phase1-Production

# SQuAD v2 Validation Tests (Quick validation only)
echo "🎯 [1/2] SQuAD v2 Full Fine-tuning Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - SQuAD v2 full fine-tuning validation with Optuna hyperparameters..."
echo "    LR=$SQUADV2_FULLFINETUNE_LR, BS=$SQUADV2_FULLFINETUNE_BS, WU=$SQUADV2_FULLFINETUNE_WU, EP=$SQUADV2_FULLFINETUNE_EP"

if python experiments/full_finetune.py \
    --task squad_v2 --mode single --seed 42 \
    --learning-rate $SQUADV2_FULLFINETUNE_LR \
    --batch-size $SQUADV2_FULLFINETUNE_BS \
    --warmup-ratio $SQUADV2_FULLFINETUNE_WU \
    --epochs $SQUADV2_FULLFINETUNE_EP \
    --weight-decay $SQUADV2_FULLFINETUNE_WD \
    --no-base-representations \
    > logs/phase1_optuna/vm1/squad_v2_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SQuAD v2 full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [2/2] SQuAD v2 LoRA Validation (1 seed, quick test)"
echo "  ⚡ $(date +'%H:%M') - SQuAD v2 LoRA validation with Optuna hyperparameters..."
echo "    LR=$SQUADV2_LORA_LR, BS=$SQUADV2_LORA_BS, WU=$SQUADV2_LORA_WU, R=$SQUADV2_LORA_R, α=$SQUADV2_LORA_A"

if python experiments/lora_finetune.py \
    --task squad_v2 --mode single --seed 42 \
    --learning-rate $SQUADV2_LORA_LR \
    --batch-size $SQUADV2_LORA_BS \
    --warmup-ratio $SQUADV2_LORA_WU \
    --epochs $SQUADV2_LORA_EP \
    --weight-decay $SQUADV2_LORA_WD \
    --lora-r $SQUADV2_LORA_R \
    --lora-alpha $SQUADV2_LORA_A \
    --lora-dropout $SQUADV2_LORA_D \
    > logs/phase1_optuna/vm1/squad_v2_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SQuAD v2 LoRA validation FAILED"
    exit 1
fi

echo "🎯 PHASE 1C COMPLETE: All SQuAD v2 hyperparameter validations finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM1 OPTUNA HYPERPARAMETER OPTIMIZATION COMPLETE! $(date)"
echo "========================================================="
echo "✅ Phase 1A: Bayesian optimization completed (60 trials total)"
echo "✅ Phase 1B: Optimal hyperparameters identified via TPE"
echo "✅ Phase 1C: Hyperparameter validation completed (2 validation tests)"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Optuna"
echo "📄 Optimal configs: analysis/squad_v2_optimal_hyperparameters.yaml"
echo "📋 Ready for Phase 2: Production experiments with optimal hyperparameters"
echo ""
echo "🧠 ACADEMIC RIGOR: Bayesian optimization (TPE) > Grid search efficiency"
echo "   • 30 trials/config vs 100+ for grid search"
echo "   • Median pruning eliminates poor trials early"
echo "   • Quick validation confirms hyperparameters work"
echo "   • SQuAD v2 focus: 3x computational weight vs classification"
echo "   • Full production experiments moved to Phase 2"