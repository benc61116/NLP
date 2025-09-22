#!/bin/bash
# Phase 1 - VM2: SQuAD v2 Training
set -e  # Exit on error

echo "Starting Phase 1 training on VM2: SQuAD v2..."

# SQuAD v2 experiments
echo "Running SQuAD v2 full fine-tuning..."
python experiments/full_finetune.py --task squad_v2

echo "Running SQuAD v2 LoRA fine-tuning..."
python experiments/lora_finetune.py --task squad_v2

echo "Phase 1 VM2 complete"
