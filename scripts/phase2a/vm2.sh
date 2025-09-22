#!/bin/bash
# Phase 2a - VM2: QA Drift Analysis + Deployment Benchmarking
set -e  # Exit on error

echo "Starting Phase 2a analysis on VM2: QA Drift + Deployment..."

# Drift analysis for QA task
echo "Running drift analysis for QA task (SQuAD v2)..."
python experiments/drift_analysis.py --tasks squad_v2

# Deployment benchmarking
echo "Running deployment benchmarking..."
python experiments/deployment_bench.py

echo "Phase 2a VM2 complete"
