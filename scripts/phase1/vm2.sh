#!/bin/bash
# Phase 1 - VM2: RTE + SQuAD v2 Experiments
set -e  # Exit on error

echo "Starting Phase 1 on VM2: RTE + SQuAD v2 Experiments..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm2

# RTE experiments (VM2 allocation)
echo "Running RTE experiments..."
python -m shared.experiment_runner --tasks rte --methods lora full 2>&1 | tee logs/phase1/vm2/rte_experiments.log

# SQuAD v2 experiments (VM2 allocation)
echo "Running SQuAD v2 experiments..."
python -m shared.experiment_runner --tasks squad_v2 --methods lora full 2>&1 | tee logs/phase1/vm2/squad_v2_experiments.log

echo "✅ Phase 1 VM2 complete: RTE + SQuAD v2 experiments finished"
