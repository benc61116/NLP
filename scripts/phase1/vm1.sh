#!/bin/bash
# Phase 1 - VM1: Sanity Checks + MRPC + SST-2 Experiments
set -e  # Exit on error

echo "Starting Phase 1 on VM1: Sanity Checks + Classification Tasks..."

# Setup environment
export WANDB_PROJECT=NLP
export WANDB_ENTITY=galavny-tel-aviv-university

# Create logs directory
mkdir -p logs/phase1/vm1

# First run comprehensive sanity checks
echo "Running comprehensive sanity checks..."
python -m shared.sanity_checks 2>&1 | tee logs/phase1/vm1/sanity_checks.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Sanity checks failed! Aborting experiments."
    exit 1
fi

echo "✅ Sanity checks passed! Starting experiments..."

# MRPC experiments (VM1 allocation)
echo "Running MRPC experiments..."
python -m shared.experiment_runner --tasks mrpc --methods lora full 2>&1 | tee logs/phase1/vm1/mrpc_experiments.log

# SST-2 experiments (VM1 allocation)  
echo "Running SST-2 experiments..."
python -m shared.experiment_runner --tasks sst2 --methods lora full 2>&1 | tee logs/phase1/vm1/sst2_experiments.log

echo "✅ Phase 1 VM1 complete: MRPC + SST-2 experiments finished"
