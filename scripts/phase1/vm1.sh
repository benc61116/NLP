#!/bin/bash
# Phase 1 - VM1: MRPC + RTE Training
set -e  # Exit on error

echo "Starting Phase 1 training on VM1: MRPC + RTE..."

# MRPC experiments
echo "Running MRPC full fine-tuning..."
python experiments/full_finetune.py --task mrpc

echo "Running MRPC LoRA fine-tuning..."
python experiments/lora_finetune.py --task mrpc

# RTE experiments  
echo "Running RTE full fine-tuning..."
python experiments/full_finetune.py --task rte

echo "Running RTE LoRA fine-tuning..."
python experiments/lora_finetune.py --task rte

echo "Phase 1 VM1 complete"
