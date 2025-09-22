#!/bin/bash
# Phase 2a - VM1: Classification Drift Analysis
set -e  # Exit on error

echo "Starting Phase 2a analysis on VM1: Classification Drift Analysis..."

# Drift analysis for classification tasks
echo "Running drift analysis for classification tasks (MRPC, SST-2, RTE)..."
python experiments/drift_analysis.py --tasks mrpc,sst2,rte

echo "Phase 2a VM1 complete"
