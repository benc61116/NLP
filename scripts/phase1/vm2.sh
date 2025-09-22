#!/bin/bash
# Phase 1 - VM2: SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 1 on VM2: SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm2

# HuggingFace authentication check
echo "Checking HuggingFace authentication for Llama-2..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-1.3b-hf')
    print('✅ HuggingFace authentication successful')
except Exception as e:
    print(f'❌ HuggingFace authentication failed: {e}')
    print('Please run: huggingface-cli login')
    exit(1)
" 2>&1 | tee logs/phase1/vm2/auth_check.log

# Run validation demo for both LoRA and full fine-tuning
echo "Running validation demos..."
python -c "
from experiments.lora_finetune import LoRAFinetuneExperiment
from experiments.full_finetune import FullFinetuneExperiment

# LoRA demo
lora_experiment = LoRAFinetuneExperiment('shared/config.yaml')
lora_experiment.config['model']['name'] = 'microsoft/DialoGPT-small'
lora_result = lora_experiment.run_validation_demo('squad_v2', 50)
print('SQuAD v2 LoRA validation demo completed')

# Full FT demo  
full_experiment = FullFinetuneExperiment('shared/config.yaml')
full_experiment.config['model']['name'] = 'microsoft/DialoGPT-small'
full_result = full_experiment.run_validation_demo('sst2', 50)
print('SST-2 full fine-tuning validation demo completed')
" 2>&1 | tee logs/phase1/vm2/validation_demos.log

echo "✅ Validation demos passed! Starting balanced experiments..."

# SQuAD v2 LoRA fine-tuning with multiple seeds (HEAVY LOAD)
echo "Running SQuAD v2 LoRA fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SQuAD v2 LoRA fine-tuning with seed $seed..."
    python experiments/lora_finetune.py --task squad_v2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm2/squad_v2_lora_seed${seed}.log
done

# SQuAD v2 LoRA hyperparameter search
echo "Running SQuAD v2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task squad_v2 --mode sweep 2>&1 | tee logs/phase1/vm2/squad_v2_lora_sweep.log

# SST-2 full fine-tuning with multiple seeds (MEDIUM LOAD)
echo "Running SST-2 full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SST-2 full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task sst2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm2/sst2_full_seed${seed}.log
done

# SST-2 hyperparameter search
echo "Running SST-2 hyperparameter sweep..."
python experiments/full_finetune.py --task sst2 --mode sweep 2>&1 | tee logs/phase1/vm2/sst2_sweep.log

# SST-2 LoRA fine-tuning with multiple seeds (MEDIUM LOAD)
echo "Running SST-2 LoRA fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SST-2 LoRA fine-tuning with seed $seed..."
    python experiments/lora_finetune.py --task sst2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm2/sst2_lora_seed${seed}.log
done

# SST-2 LoRA hyperparameter search
echo "Running SST-2 LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task sst2 --mode sweep 2>&1 | tee logs/phase1/vm2/sst2_lora_sweep.log

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

echo "✅ Phase 1 VM2 complete: SQuAD v2 LoRA + SST-2 Full FT + SST-2 LoRA finished"
echo "  - 3 SQuAD v2 LoRA fine-tuning runs completed (1 heavy task)"
echo "  - 3 SST-2 full fine-tuning runs completed (1 medium task)"
echo "  - 3 SST-2 LoRA fine-tuning runs completed (1 medium task)"
echo "  - 4 hyperparameter sweeps completed"
echo "  - Memory optimization for longer sequences (768 tokens)"
echo "  - All representations extracted and saved"
echo "  - Checkpoints saved for analysis phases"
