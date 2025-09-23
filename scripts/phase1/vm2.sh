#!/bin/bash
# Phase 1 - VM2: Mixed Heavy Tasks (SQuAD v2 + RTE Full & LoRA)
set -e  # Exit on error

echo "Starting Phase 1 on VM2: Mixed Heavy Tasks (SQuAD v2 + RTE Full & LoRA)..."

# Setup environment
export WANDB_PROJECT=NLP-Phase1-Training
export WANDB_ENTITY=galavny-tel-aviv-university

# Clear GPU memory cache to ensure maximum available memory
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
mkdir -p logs/phase1/vm2

# HuggingFace authentication check
echo "Checking TinyLlama model access..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    print('✅ TinyLlama model accessible')
except Exception as e:
    print(f'❌ TinyLlama access failed: {e}')
    print('Please check internet connection')
    exit(1)
" 2>&1 | tee logs/phase1/vm2/auth_check.log

echo "✅ TinyLlama model check passed! Starting Phase 1 experiments..."
echo "📅 Started at: $(date)"
echo ""

# SQuAD v2 full fine-tuning with multiple seeds (HEAVY LOAD)
echo "🔬 [1/4] SQuAD v2 Full Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 full fine-tuning (seed $seed)..."
    python experiments/full_finetune.py --task squad_v2 --mode single --seed $seed > logs/phase1/vm2/squad_v2_full_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 full fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 hyperparameter sweep..."
python experiments/full_finetune.py --task squad_v2 --mode sweep > logs/phase1/vm2/squad_v2_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SQuAD v2 hyperparameter sweep complete"
echo "🎯 [1/4] SQuAD v2 Full Fine-tuning COMPLETE"
echo ""

# SQuAD v2 LoRA fine-tuning with multiple seeds (HEAVY LOAD)
echo "🔬 [2/4] SQuAD v2 LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task squad_v2 --mode single --seed $seed > logs/phase1/vm2/squad_v2_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task squad_v2 --mode sweep > logs/phase1/vm2/squad_v2_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA hyperparameter sweep complete"
echo "🎯 [2/4] SQuAD v2 LoRA Fine-tuning COMPLETE"
echo ""

# RTE full fine-tuning with multiple seeds (LIGHT LOAD)
echo "🔬 [3/4] RTE Full Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting RTE full fine-tuning (seed $seed)..."
    python experiments/full_finetune.py --task rte --mode single --seed $seed > logs/phase1/vm2/rte_full_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - RTE full fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting RTE hyperparameter sweep..."
python experiments/full_finetune.py --task rte --mode sweep > logs/phase1/vm2/rte_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - RTE hyperparameter sweep complete"
echo "🎯 [3/4] RTE Full Fine-tuning COMPLETE"
echo ""

# RTE LoRA fine-tuning with multiple seeds (LIGHT LOAD)
echo "🔬 [4/4] RTE LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task rte --mode single --seed $seed > logs/phase1/vm2/rte_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - RTE LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting RTE LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task rte --mode sweep > logs/phase1/vm2/rte_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - RTE LoRA hyperparameter sweep complete"
echo "🎯 [4/4] RTE LoRA Fine-tuning COMPLETE"

echo ""
echo "🎉 VM2 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"