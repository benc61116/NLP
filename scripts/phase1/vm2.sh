#!/bin/bash
# Phase 1 - VM2: OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# Academic-grade Bayesian optimization for SQuAD v2 + RTE + Infrastructure setup
set -e  # Exit on error

echo "🚀 PHASE 1 - VM2: OPTUNA BAYESIAN OPTIMIZATION + INFRASTRUCTURE"
echo "==============================================================="
echo "Academic-grade hyperparameter optimization workflow:"
echo "1. Bayesian optimization (TPE) for SQuAD v2 + RTE (30 trials each)"
echo "2. Optimal hyperparameter extraction"
echo "3. Production experiments using optimal hyperparameters"
echo "4. Shared infrastructure setup for all VMs"
echo "==============================================================="

# Setup environment
export WANDB_PROJECT=NLP-Phase1-Optuna
export WANDB_ENTITY=galavny-tel-aviv-university
export PYTHONPATH=/home/benc6116/workspace/NLP:$PYTHONPATH

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
echo "SQuAD v2 = 3x computational weight vs classification tasks"
echo "------------------------------------------------------------"

# SQuAD v2 Optimization (Heavy computational load)
echo "⚡ [1/4] SQuAD v2 Full Fine-tuning Optimization (30 trials)"
echo "   🔥 Note: SQuAD v2 has 3x computational weight vs classification"
if python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method full_finetune \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/squad_v2_full_finetune_optimal.yaml \
    > logs/phase1_optuna/vm2/squad_v2_full_optuna.log 2>&1; then
    echo "✅ SQuAD v2 full fine-tuning optimization completed (30 trials)"
else
    echo "❌ SQuAD v2 full fine-tuning optimization FAILED"
    exit 1
fi

echo "⚡ [2/4] SQuAD v2 LoRA Optimization (30 trials)"
if python experiments/optuna_optimization.py \
    --task squad_v2 \
    --method lora \
    --n-trials 30 \
    --wandb-project NLP-Phase1-Optuna \
    --output-file analysis/squad_v2_lora_optimal.yaml \
    > logs/phase1_optuna/vm2/squad_v2_lora_optuna.log 2>&1; then
    echo "✅ SQuAD v2 LoRA optimization completed (30 trials)"
else
    echo "❌ SQuAD v2 LoRA optimization FAILED"
    exit 1
fi

# RTE Optimization (Lighter computational load)
echo "⚡ [3/4] RTE Full Fine-tuning Optimization (30 trials)"
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

echo "⚡ [4/4] RTE LoRA Optimization (30 trials)"
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

echo "🎯 PHASE 1A COMPLETE: All VM2 Optuna optimizations finished!"
echo "Total trials: 120 (4 × 30 trials with TPE sampler + median pruning)"
echo ""

# ============================================================================
# PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION
# ============================================================================
echo "📊 PHASE 1B: OPTIMAL HYPERPARAMETER EXTRACTION"
echo "Consolidating VM2 Optuna results into task-specific optimal configurations"
echo "------------------------------------------------------------"

# Create task-specific optimal hyperparameters files (better practice)
echo "⚡ Creating task-specific optimal hyperparameter files..."

python -c "
import yaml
from pathlib import Path

# Process SQuAD v2 results
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
if [ ! -f "analysis/squad_v2_optimal_hyperparameters.yaml" ] || [ ! -f "analysis/rte_optimal_hyperparameters.yaml" ]; then
    echo "❌ Task-specific optimal hyperparameters files not found!"
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
echo "🎯 RTE (Classification):"
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

# Extract optimal hyperparameters for RTE
python -c "
import yaml

# RTE hyperparameters
with open('analysis/rte_optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)

optimal_hp = config['optimal_hyperparameters']
for method in ['full_finetune', 'lora']:
    if method in optimal_hp:
        hp = optimal_hp[method]['hyperparameters']
        method_upper = method.upper().replace('_', '')
        
        print(f'export RTE_{method_upper}_LR={hp[\"learning_rate\"]}')
        print(f'export RTE_{method_upper}_BS={hp[\"per_device_train_batch_size\"]}')
        print(f'export RTE_{method_upper}_WU={hp[\"warmup_ratio\"]}')
        print(f'export RTE_{method_upper}_EP={hp[\"num_train_epochs\"]}')
        print(f'export RTE_{method_upper}_WD={hp.get(\"weight_decay\", 0.01)}')
        
        if method == 'lora':
            print(f'export RTE_{method_upper}_R={hp.get(\"lora_r\", 8)}')
            print(f'export RTE_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
            print(f'export RTE_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > rte_hyperparams.sh

source squad_v2_hyperparams.sh
source rte_hyperparams.sh

# Change wandb project for production runs
export WANDB_PROJECT=NLP-Phase1-Production

# SQuAD v2 Validation Tests (Quick validation only)
echo "🎯 [1/4] SQuAD v2 Full Fine-tuning Validation (1 seed, quick test)"
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
    > logs/phase1_optuna/vm2/squad_v2_full_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 full fine-tuning validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SQuAD v2 full fine-tuning validation FAILED"
    exit 1
fi

echo "🎯 [2/4] SQuAD v2 LoRA Validation (1 seed, quick test)"
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
    > logs/phase1_optuna/vm2/squad_v2_lora_validation.log 2>&1; then
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA validation COMPLETED"
else
    echo "  ❌ $(date +'%H:%M') - SQuAD v2 LoRA validation FAILED"
    exit 1
fi

# RTE Validation Tests (Quick validation only)
echo "🎯 [3/4] RTE Full Fine-tuning Validation (1 seed, quick test)"
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

echo "🎯 [4/4] RTE LoRA Validation (1 seed, quick test)"
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

echo "🎯 PHASE 1C COMPLETE: All VM2 hyperparameter validations finished!"
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
    'analysis/optimal_hyperparameters.yaml',  # VM1 unified file (if available)
    'analysis/squad_v2_optimal_hyperparameters.yaml',  # VM2 SQuAD v2
    'analysis/rte_optimal_hyperparameters.yaml'  # VM2 RTE
]

unified_summary = {
    'phase1_summary': {
        'optimization_method': 'optuna_tpe_bayesian',
        'total_trials_across_all_vms': 240,  # 4×30 VM1 + 4×30 VM2
        'trials_per_task_method': 30,
        'vm_distribution': {
            'vm1': ['mrpc', 'sst2'],
            'vm2': ['squad_v2', 'rte']
        }
    },
    'task_optimal_hyperparameters': {}
}

# Process each file
for file_path in hyperparams_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            config = yaml.safe_load(f)
        
        # Handle different file formats
        if 'optimal_hyperparameters' in config:
            # VM1 format (multiple tasks)
            for task, methods in config['optimal_hyperparameters'].items():
                unified_summary['task_optimal_hyperparameters'][task] = methods
        elif 'task' in config:
            # VM2 format (single task)
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
        'vm2_tasks': ['squad_v2', 'rte'],
        'optimization_method': 'optuna_tpe',
        'total_trials_vm2': 120,
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
echo "🎉 VM2 OPTUNA HYPERPARAMETER OPTIMIZATION + INFRASTRUCTURE COMPLETE! $(date)"
echo "========================================================================="
echo "✅ Phase 1A: Bayesian optimization completed (120 trials VM2)"
echo "✅ Phase 1B: Task-specific optimal hyperparameters identified"
echo "✅ Phase 1C: Hyperparameter validation completed (4 validation tests)"
echo "✅ Phase 1D: Shared infrastructure setup completed"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Optuna"
echo "📄 Task configs: analysis/squad_v2_optimal_hyperparameters.yaml"
echo "📄 Task configs: analysis/rte_optimal_hyperparameters.yaml"
echo "📄 Unified summary: results/phase1_summaries/all_optimal_hyperparameters.yaml"
echo "📋 Ready for Phase 2: Production experiments with optimal hyperparameters"
echo ""
echo "🧠 ACADEMIC RIGOR: Bayesian optimization (TPE) > Grid search efficiency"
echo "   • 30 trials/config vs 100+ for grid search"
echo "   • Median pruning eliminates poor trials early"
echo "   • Quick validation confirms hyperparameters work"
echo "   • Task-specific optimization for SQuAD v2 vs classification"
echo "   • Full production experiments moved to Phase 2"
