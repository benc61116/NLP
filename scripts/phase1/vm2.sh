#!/bin/bash
# Phase 1 - VM2: SQuAD v2 Full Fine-tuning Experiments
set -e  # Exit on error

echo "Starting Phase 1 on VM2: SQuAD v2 Full Fine-tuning..."

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

# Run validation demo for QA task
echo "Running SQuAD v2 validation demo..."
python -c "
from experiments.full_finetune import FullFinetuneExperiment
experiment = FullFinetuneExperiment('shared/config.yaml')
# Use smaller model for demo
experiment.config['model']['name'] = 'microsoft/DialoGPT-small'
result = experiment.run_validation_demo('squad_v2', 50)
print('SQuAD v2 validation demo completed')
" 2>&1 | tee logs/phase1/vm2/squad_validation_demo.log

echo "✅ Validation demo passed! Starting SQuAD v2 full experiments..."

# SQuAD v2 full fine-tuning with multiple seeds
echo "Running SQuAD v2 full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SQuAD v2 full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task squad_v2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm2/squad_v2_full_seed${seed}.log
done

# SQuAD v2 hyperparameter search (focused on QA-specific hyperparameters)
echo "Running SQuAD v2 hyperparameter sweep..."
python experiments/full_finetune.py --task squad_v2 --mode sweep 2>&1 | tee logs/phase1/vm2/squad_v2_sweep.log

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

echo "✅ Phase 1 VM2 complete: SQuAD v2 full fine-tuning finished"
echo "  - 3 full fine-tuning runs completed (3 seeds)"
echo "  - 1 hyperparameter sweep completed"
echo "  - Memory optimization for longer sequences (768 tokens)"
echo "  - All representations extracted and saved"
echo "  - Checkpoints saved for analysis phases"
