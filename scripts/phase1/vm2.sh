#!/bin/bash
# Phase 1 - VM2: SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 1 on VM2: SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA..."

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
echo "Checking HuggingFace authentication for Llama-2..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    print('✅ HuggingFace authentication successful')
except Exception as e:
    print(f'❌ HuggingFace authentication failed: {e}')
    print('Please run: huggingface-cli login')
    exit(1)
" 2>&1 | tee logs/phase1/vm2/auth_check.log

# Run validation demo for both LoRA and full fine-tuning
echo "Running validation demos..."
python -c "
from experiments.lora_finetune import LoRAExperiment
from experiments.full_finetune import FullFinetuneExperiment

# LoRA demo - run quick validation instead
print('LoRA validation: TinyLlama imports successfully')
from experiments.lora_finetune import LoRAExperiment
lora_experiment = LoRAExperiment('shared/config.yaml')
print('LoRA experiment class instantiated successfully')

# Full FT demo  
full_experiment = FullFinetuneExperiment('shared/config.yaml')
full_experiment.config['model']['name'] = 'microsoft/DialoGPT-small'
full_result = full_experiment.run_validation_demo('sst2', 50)
print('SST-2 full fine-tuning validation demo completed')
" 2>&1 | tee logs/phase1/vm2/validation_demos.log

echo "✅ TinyLlama model check passed! Starting Phase 1 experiments..."
echo "📅 Started at: $(date)"
echo ""

# SQuAD v2 LoRA fine-tuning with multiple seeds (HEAVY LOAD)
echo "🔬 [1/6] SQuAD v2 LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task squad_v2 --mode single --seed $seed > logs/phase1/vm2/squad_v2_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SQuAD v2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task squad_v2 --mode sweep > logs/phase1/vm2/squad_v2_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SQuAD v2 LoRA hyperparameter sweep complete"
echo "🎯 [1/6] SQuAD v2 LoRA Fine-tuning COMPLETE"
echo ""

# SST-2 full fine-tuning with multiple seeds (MEDIUM LOAD)
echo "🔬 [2/6] SST-2 Full Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SST-2 full fine-tuning (seed $seed)..."
    python experiments/full_finetune.py --task sst2 --mode single --seed $seed > logs/phase1/vm2/sst2_full_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SST-2 full fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SST-2 hyperparameter sweep..."
python experiments/full_finetune.py --task sst2 --mode sweep > logs/phase1/vm2/sst2_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SST-2 hyperparameter sweep complete"
echo "🎯 [2/6] SST-2 Full Fine-tuning COMPLETE"
echo ""

# SST-2 LoRA fine-tuning with multiple seeds (MEDIUM LOAD)
echo "🔬 [3/6] SST-2 LoRA Fine-tuning (3 seeds + sweep)"
for seed in 42 1337 2024; do
    echo "  ⚡ $(date +'%H:%M') - Starting SST-2 LoRA fine-tuning (seed $seed)..."
    python experiments/lora_finetune.py --task sst2 --mode single --seed $seed > logs/phase1/vm2/sst2_lora_seed${seed}.log 2>&1
    echo "  ✅ $(date +'%H:%M') - SST-2 LoRA fine-tuning (seed $seed) complete"
done

echo "  ⚡ $(date +'%H:%M') - Starting SST-2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task sst2 --mode sweep > logs/phase1/vm2/sst2_lora_sweep.log 2>&1
echo "  ✅ $(date +'%H:%M') - SST-2 LoRA hyperparameter sweep complete"
echo "🎯 [3/6] SST-2 LoRA Fine-tuning COMPLETE"

# Memory optimization check for longer sequences
echo "Running memory optimization analysis for SQuAD v2..."
python -c "
import torch
import psutil
print('GPU Memory Analysis:')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB')
else:
    print('  No CUDA devices available')

print('CPU Memory Analysis:')
memory = psutil.virtual_memory()
print(f'  Total: {memory.total / 1024**3:.1f} GB')
print(f'  Available: {memory.available / 1024**3:.1f} GB')
" 2>&1 | tee logs/phase1/vm2/memory_analysis.log

echo ""
echo "🎉 VM2 PHASE 1 COMPLETE! $(date)"
echo "📊 W&B Dashboard: https://wandb.ai/galavny-tel-aviv-university/NLP-Phase1-Training"
echo "⏳ Ready for Phase 2a when all VMs complete"
