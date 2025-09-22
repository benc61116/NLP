#!/bin/bash
# Phase 2a - VM1: LoRA Fine-tuning for All Tasks
set -e  # Exit on error

echo "Starting Phase 2a on VM1: LoRA Fine-tuning for All Tasks..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm1

# Run all tasks with LoRA method
echo "Running LoRA experiments for all tasks..."
python -m shared.experiment_runner --tasks mrpc sst2 rte squad_v2 --methods lora --skip-sanity-checks 2>&1 | tee logs/phase2a/vm1/lora_all_tasks.log

echo "✅ Phase 2a VM1 complete: LoRA experiments for all tasks finished"
