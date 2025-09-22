#!/bin/bash
# Phase 1 - VM3: SST-2 Training + All Baselines
set -e  # Exit on error

echo "Starting Phase 1 training on VM3: SST-2 + Baselines..."

# SST-2 experiments
echo "Running SST-2 full fine-tuning..."
python experiments/full_finetune.py --task sst2

echo "Running SST-2 LoRA fine-tuning..."
python experiments/lora_finetune.py --task sst2

# Baseline experiments for all tasks
echo "Running baseline experiments for all tasks..."
python experiments/baselines.py --all-tasks

echo "Phase 1 VM3 complete"
