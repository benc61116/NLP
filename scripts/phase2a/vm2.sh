#!/bin/bash
# Phase 2a - VM2: Full Fine-tuning for All Tasks
set -e  # Exit on error

echo "Starting Phase 2a on VM2: Full Fine-tuning for All Tasks..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase2a/vm2

# Run all tasks with full fine-tuning method
echo "Running full fine-tuning experiments for all tasks..."
python -m shared.experiment_runner --tasks mrpc sst2 rte squad_v2 --methods full --skip-sanity-checks 2>&1 | tee logs/phase2a/vm2/full_all_tasks.log

echo "✅ Phase 2a VM2 complete: Full fine-tuning experiments for all tasks finished"
