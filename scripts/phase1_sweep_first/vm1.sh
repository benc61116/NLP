#!/bin/bash
# Phase 1 - VM1: SWEEP-FIRST METHODOLOGY Implementation
# Proper academic approach: 1) Sweeps → 2) Analysis → 3) Production experiments with optimal hyperparameters
set -e  # Exit on error

echo "🚀 PHASE 1 - VM1: SWEEP-FIRST METHODOLOGY"
echo "=========================================="
echo "Academic-grade hyperparameter optimization workflow:"
echo "1. Hyperparameter sweeps for MRPC + SST-2"
echo "2. Analysis to identify optimal hyperparameters"  
echo "3. Production experiments using optimal hyperparameters"
echo "=========================================="

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
mkdir -p logs/phase1_sweep_first/vm1

echo "📅 Started at: $(date)"
echo ""

# ============================================================================
# PHASE 1A: HYPERPARAMETER SWEEPS (Academic Requirement)
# ============================================================================
echo "🔬 PHASE 1A: HYPERPARAMETER SWEEPS"
echo "Find optimal hyperparameters BEFORE running production experiments"
echo "------------------------------------------------------------"

# MRPC Sweeps
echo "⚡ [1/4] MRPC Full Fine-tuning Hyperparameter Sweep"
if python experiments/full_finetune.py --task mrpc --mode sweep --no-base-representations > logs/phase1_sweep_first/vm1/mrpc_full_sweep.log 2>&1; then
    echo "✅ MRPC full fine-tuning sweep completed"
else
    echo "❌ MRPC full fine-tuning sweep FAILED"
    exit 1
fi

echo "⚡ [2/4] MRPC LoRA Hyperparameter Sweep"
if python experiments/lora_finetune.py --task mrpc --mode sweep > logs/phase1_sweep_first/vm1/mrpc_lora_sweep.log 2>&1; then
    echo "✅ MRPC LoRA sweep completed"
else
    echo "❌ MRPC LoRA sweep FAILED"
    exit 1
fi

# SST-2 Sweeps
echo "⚡ [3/4] SST-2 Full Fine-tuning Hyperparameter Sweep"
if python experiments/full_finetune.py --task sst2 --mode sweep --no-base-representations > logs/phase1_sweep_first/vm1/sst2_full_sweep.log 2>&1; then
    echo "✅ SST-2 full fine-tuning sweep completed"
else
    echo "❌ SST-2 full fine-tuning sweep FAILED"
    exit 1
fi

echo "⚡ [4/4] SST-2 LoRA Hyperparameter Sweep"
if python experiments/lora_finetune.py --task sst2 --mode sweep > logs/phase1_sweep_first/vm1/sst2_lora_sweep.log 2>&1; then
    echo "✅ SST-2 LoRA sweep completed"
else
    echo "❌ SST-2 LoRA sweep FAILED"
    exit 1
fi

echo "🎯 PHASE 1A COMPLETE: All hyperparameter sweeps finished!"
echo ""

# ============================================================================
# PHASE 1B: SWEEP ANALYSIS (Identify Optimal Hyperparameters)
# ============================================================================
echo "📊 PHASE 1B: SWEEP ANALYSIS"
echo "Analyzing sweep results to identify optimal hyperparameters"
echo "------------------------------------------------------------"

mkdir -p analysis

echo "⚡ Analyzing sweep results and identifying optimal configurations..."
if python scripts/analyze_sweeps.py --export-optimal-configs --output-dir analysis > logs/phase1_sweep_first/vm1/sweep_analysis.log 2>&1; then
    echo "✅ Sweep analysis completed"
    echo "📄 Optimal hyperparameters saved to: analysis/optimal_hyperparameters.yaml"
else
    echo "❌ Sweep analysis FAILED"
    echo "Check logs/phase1_sweep_first/vm1/sweep_analysis.log for details"
    exit 1
fi

# Verify optimal configurations were generated
if [ ! -f "analysis/optimal_hyperparameters.yaml" ]; then
    echo "❌ Optimal hyperparameters file not found!"
    exit 1
fi

echo "📋 OPTIMAL HYPERPARAMETERS IDENTIFIED:"
echo "------------------------------------------------------------"
python -c "
import yaml
with open('analysis/optimal_hyperparameters.yaml') as f:
    config = yaml.safe_load(f)
optimal_hp = config['optimal_hyperparameters']
for task, methods in optimal_hp.items():
    print(f'{task.upper()}:')
    for method, info in methods.items():
        hp = info['hyperparameters']
        perf = info['expected_performance']
        lr = hp['learning_rate']
        bs = hp['per_device_train_batch_size']
        wu = hp['warmup_ratio']
        acc = perf.get('eval_accuracy', perf.get('eval_f1', 0))
        print(f'  {method:12}: LR={lr:8} BS={bs:2} WU={wu:.2f} → Perf={acc:.3f}')
"

echo "🎯 PHASE 1B COMPLETE: Optimal hyperparameters identified!"
echo ""

# ============================================================================
# PHASE 1C: PRODUCTION EXPERIMENTS (Using Optimal Hyperparameters)
# ============================================================================
echo "🚀 PHASE 1C: PRODUCTION EXPERIMENTS"
echo "Running experiments with optimal hyperparameters (multiple seeds)"
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
            
            if method == 'lora':
                print(f'export {task_upper}_{method_upper}_R={hp.get(\"lora_r\", 8)}')
                print(f'export {task_upper}_{method_upper}_A={hp.get(\"lora_alpha\", 16)}')
                print(f'export {task_upper}_{method_upper}_D={hp.get(\"lora_dropout\", 0.05)}')
" > optimal_hyperparams.sh

source optimal_hyperparams.sh

# MRPC Production Experiments with Optimal Hyperparameters
echo "🎯 [1/4] MRPC Full Fine-tuning Production (3 seeds with optimal hyperparameters)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) with optimal hyperparameters..."
    echo "    LR=$MRPC_FULLFINETUNE_LR, BS=$MRPC_FULLFINETUNE_BS, WU=$MRPC_FULLFINETUNE_WU"
    
    if python experiments/full_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_FULLFINETUNE_LR \
        --batch-size $MRPC_FULLFINETUNE_BS \
        --warmup-ratio $MRPC_FULLFINETUNE_WU \
        --epochs $MRPC_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_sweep_first/vm1/mrpc_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 [2/4] MRPC LoRA Production (3 seeds with optimal hyperparameters)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - MRPC LoRA (seed $seed) with optimal hyperparameters..."
    echo "    LR=$MRPC_LORA_LR, BS=$MRPC_LORA_BS, WU=$MRPC_LORA_WU, R=$MRPC_LORA_R"
    
    if python experiments/lora_finetune.py \
        --task mrpc --mode single --seed $seed \
        --learning-rate $MRPC_LORA_LR \
        --batch-size $MRPC_LORA_BS \
        --warmup-ratio $MRPC_LORA_WU \
        --epochs $MRPC_LORA_EP \
        --lora-r $MRPC_LORA_R \
        --lora-alpha $MRPC_LORA_A \
        --lora-dropout $MRPC_LORA_D \
        > logs/phase1_sweep_first/vm1/mrpc_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - MRPC LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - MRPC LoRA (seed $seed) FAILED"
        exit 1
    fi
done

# SST-2 Production Experiments with Optimal Hyperparameters
echo "🎯 [3/4] SST-2 Full Fine-tuning Production (3 seeds with optimal hyperparameters)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) with optimal hyperparameters..."
    echo "    LR=$SST2_FULLFINETUNE_LR, BS=$SST2_FULLFINETUNE_BS, WU=$SST2_FULLFINETUNE_WU"
    
    if python experiments/full_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_FULLFINETUNE_LR \
        --batch-size $SST2_FULLFINETUNE_BS \
        --warmup-ratio $SST2_FULLFINETUNE_WU \
        --epochs $SST2_FULLFINETUNE_EP \
        --no-base-representations \
        > logs/phase1_sweep_first/vm1/sst2_full_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 [4/4] SST-2 LoRA Production (3 seeds with optimal hyperparameters)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - SST-2 LoRA (seed $seed) with optimal hyperparameters..."
    echo "    LR=$SST2_LORA_LR, BS=$SST2_LORA_BS, WU=$SST2_LORA_WU, R=$SST2_LORA_R"
    
    if python experiments/lora_finetune.py \
        --task sst2 --mode single --seed $seed \
        --learning-rate $SST2_LORA_LR \
        --batch-size $SST2_LORA_BS \
        --warmup-ratio $SST2_LORA_WU \
        --epochs $SST2_LORA_EP \
        --lora-r $SST2_LORA_R \
        --lora-alpha $SST2_LORA_A \
        --lora-dropout $SST2_LORA_D \
        > logs/phase1_sweep_first/vm1/sst2_lora_optimal_seed${seed}.log 2>&1; then
        echo "  ✅ $(date +'%H:%M') - SST-2 LoRA (seed $seed) COMPLETED"
    else
        echo "  ❌ $(date +'%H:%M') - SST-2 LoRA (seed $seed) FAILED"
        exit 1
    fi
done

echo "🎯 PHASE 1C COMPLETE: All production experiments finished!"
echo ""

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "🎉 VM1 SWEEP-FIRST METHODOLOGY COMPLETE! $(date)"
echo "=========================================="
echo "✅ Phase 1A: Hyperparameter sweeps completed (4 sweeps)"
echo "✅ Phase 1B: Optimal hyperparameters identified"
echo "✅ Phase 1C: Production experiments completed (12 experiments)"
echo ""
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "📄 Optimal configs: analysis/optimal_hyperparameters.yaml"
echo "📋 Results ready for Phase 2a analysis when all VMs complete"
echo ""
echo "This implements proper academic methodology: Sweep → Analyze → Optimize!"
