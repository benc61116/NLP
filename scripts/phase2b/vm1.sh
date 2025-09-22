#!/bin/bash
# Phase 2b - VM1: Final Synthesis
set -e  # Exit on error

echo "Starting Phase 2b synthesis on VM1: Statistical Analysis + Visualization..."

# Statistical analysis
echo "Running statistical analysis..."
python analysis/statistical_analysis.py

# Visualization
echo "Generating visualizations..."
python analysis/visualization.py

# Report generation
echo "Generating final report..."
python analysis/report_generation.py

echo "Phase 2b VM1 complete - All experiments finished!"
