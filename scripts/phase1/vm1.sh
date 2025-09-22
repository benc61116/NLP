#!/bin/bash
# Phase 1 - VM1: MRPC + RTE Full Fine-tuning Experiments
set -e  # Exit on error

echo "Starting Phase 1 on VM1: MRPC + RTE Full Fine-tuning..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm1

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
" 2>&1 | tee logs/phase1/vm1/auth_check.log

# Run validation demo first
echo "Running full fine-tuning validation demo..."
python run_full_finetune_demo.py 2>&1 | tee logs/phase1/vm1/validation_demo.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Validation demo failed! Check implementation."
    exit 1
fi

echo "✅ Validation demo passed! Starting full experiments..."

# MRPC full fine-tuning with multiple seeds
echo "Running MRPC full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running MRPC full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task mrpc --mode single --seed $seed 2>&1 | tee logs/phase1/vm1/mrpc_full_seed${seed}.log
done

# MRPC hyperparameter search
echo "Running MRPC hyperparameter sweep..."
python experiments/full_finetune.py --task mrpc --mode sweep 2>&1 | tee logs/phase1/vm1/mrpc_sweep.log

# RTE full fine-tuning with multiple seeds
echo "Running RTE full fine-tuning experiments..."
for seed in 42 1337 2024; do
    echo "  Running RTE full fine-tuning with seed $seed..."
    python experiments/full_finetune.py --task rte --mode single --seed $seed 2>&1 | tee logs/phase1/vm1/rte_full_seed${seed}.log
done

# RTE hyperparameter search
echo "Running RTE hyperparameter sweep..."
python experiments/full_finetune.py --task rte --mode sweep 2>&1 | tee logs/phase1/vm1/rte_sweep.log

echo "✅ Phase 1 VM1 complete: MRPC + RTE full fine-tuning finished"
echo "  - 6 full fine-tuning runs completed (3 seeds × 2 tasks)"
echo "  - 2 hyperparameter sweeps completed"
echo "  - All representations extracted and saved"
echo "  - Checkpoints saved for analysis phases"
