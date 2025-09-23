#!/bin/bash
# Phase 1 - VM1: Classification Tasks (MRPC + SST-2 Full & LoRA)
set -e  # Exit on error

echo "Starting Phase 1 on VM1: Classification Tasks (MRPC + SST-2 Full & LoRA)..."

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
mkdir -p logs/phase1/vm1

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
" 2>&1 | tee logs/phase1/vm1/auth_check.log

# Skip validation demo - start full experiments directly
echo "✅ TinyLlama model check passed! Starting Phase 1 experiments..."
echo "📅 Started at: $(date)"
echo ""

# MRPC full fine-tuning with multiple seeds
echo "🔬 [1/4] MRPC Full Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting MRPC full fine-tuning (seed $seed)..."
    python experiments/full_finetune.py --task mrpc --mode single --seed $seed > logs/phase1/vm1/mrpc_full_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - MRPC full fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting MRPC hyperparameter sweep..."
python experiments/full_finetune.py --task mrpc --mode sweep > logs/phase1/vm1/mrpc_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - MRPC hyperparameter sweep complete"
echo "🎯 [1/4] MRPC Full Fine-tuning COMPLETE"
echo ""

# MRPC LoRA fine-tuning with multiple seeds
echo "🔬 [2/4] MRPC LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting MRPC LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task mrpc --mode single --seed $seed > logs/phase1/vm1/mrpc_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - MRPC LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting MRPC LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task mrpc --mode sweep > logs/phase1/vm1/mrpc_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - MRPC LoRA hyperparameter sweep complete"
echo "🎯 [2/4] MRPC LoRA Fine-tuning COMPLETE"
echo ""

# SST-2 full fine-tuning with multiple seeds
echo "🔬 [3/4] SST-2 Full Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SST-2 full fine-tuning (seed $seed)..."
    python experiments/full_finetune.py --task sst2 --mode single --seed $seed > logs/phase1/vm1/sst2_full_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SST-2 hyperparameter sweep..."
python experiments/full_finetune.py --task sst2 --mode sweep > logs/phase1/vm1/sst2_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SST-2 hyperparameter sweep complete"
echo "🎯 [3/4] SST-2 Full Fine-tuning COMPLETE"
echo ""

# SST-2 LoRA fine-tuning with multiple seeds
echo "🔬 [4/4] SST-2 LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SST-2 LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task sst2 --mode single --seed $seed > logs/phase1/vm1/sst2_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SST-2 LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SST-2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task sst2 --mode sweep > logs/phase1/vm1/sst2_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SST-2 LoRA hyperparameter sweep complete"
echo "🎯 [4/4] SST-2 LoRA Fine-tuning COMPLETE"

echo ""
echo "🎉 VM1 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"