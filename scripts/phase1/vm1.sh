#!/bin/bash
# Phase 1 - VM1: SQuAD v2 Full FT + MRPC Full FT + MRPC LoRA (Balanced Load)
set -e  # Exit on error

echo "Starting Phase 1 on VM1: SQuAD v2 Full FT + MRPC Full FT + MRPC LoRA..."

# Setup environment
export WANDB_PROJECT=NLP-Phase1-Training
export WANDB_ENTITY=galavny-tel-aviv-university

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

# Run validation demo first
echo "Running full fine-tuning validation demo..."
python run_full_finetune_demo.py 2>&1 | tee logs/phase1/vm1/validation_demo.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Validation demo failed! Check implementation."
    exit 1
fi

echo "✅ Validation demo passed! Starting full experiments..."

# SQuAD v2 full fine-tuning with multiple seeds (HEAVY LOAD)
echo "Running SQuAD v2 full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running SQuAD v2 full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task squad_v2 --mode single --seed $seed 2>&1 | tee logs/phase1/vm1/squad_v2_full_seed${seed}.log
done

# SQuAD v2 hyperparameter search
echo "Running SQuAD v2 hyperparameter sweep..."
python experiments/full_finetune.py --task squad_v2 --mode sweep 2>&1 | tee logs/phase1/vm1/squad_v2_sweep.log

# MRPC full fine-tuning with multiple seeds (LIGHT LOAD)
echo "Running MRPC full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running MRPC full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task mrpc --mode single --seed $seed 2>&1 | tee logs/phase1/vm1/mrpc_full_seed${seed}.log
done

# MRPC hyperparameter search
echo "Running MRPC hyperparameter sweep..."
python experiments/full_finetune.py --task mrpc --mode sweep 2>&1 | tee logs/phase1/vm1/mrpc_sweep.log

# MRPC LoRA fine-tuning with multiple seeds (LIGHT LOAD)
echo "Running MRPC LoRA fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running MRPC LoRA fine-tuning with seed $seed..."
    python experiments/lora_finetune.py --task mrpc --mode single --seed $seed 2>&1 | tee logs/phase1/vm1/mrpc_lora_seed${seed}.log
done

# MRPC LoRA hyperparameter search
echo "Running MRPC LoRA hyperparameter sweep..."
python experiments/lora_finetune.py --task mrpc --mode sweep 2>&1 | tee logs/phase1/vm1/mrpc_lora_sweep.log

echo "✅ Phase 1 VM1 complete: SQuAD v2 Full FT + MRPC Full FT + MRPC LoRA finished"
echo "  - 3 SQuAD v2 full fine-tuning runs completed (1 heavy task)"
echo "  - 3 MRPC full fine-tuning runs completed (1 light task)"
echo "  - 3 MRPC LoRA fine-tuning runs completed (1 light task)"
echo "  - 3 hyperparameter sweeps completed"
echo "  - All representations extracted and saved"
echo "  - Checkpoints saved for analysis phases"
