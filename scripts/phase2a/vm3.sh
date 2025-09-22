#!/bin/bash
# Phase 2a - VM3: Correlation Analysis
set -e  # Exit on error

echo "Starting Phase 2a analysis on VM3: Correlation Analysis..."

# Correlation analysis between performance and drift
echo "Running correlation analysis..."
python analysis/correlation_analysis.py

echo "Phase 2a VM3 complete"
