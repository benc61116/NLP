#!/usr/bin/env python3
"""
Publication-Quality Drift Analysis Visualizations

Creates comprehensive visualizations for the LoRA vs Full Fine-tuning
representation drift analysis research.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DriftVisualizer:
    """Create publication-quality drift analysis visualizations."""
    
    def __init__(self, results_file: Path, output_dir: Path):
        """Initialize visualizer with drift analysis results."""
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_file) as f:
            self.results = json.load(f)
        
        logger.info(f"Initialized DriftVisualizer")
        logger.info(f"  Results: {self.results_file}")
        logger.info(f"  Output: {self.output_dir}")
    
    def create_layer_drift_heatmap(self):
        """Create layer-wise drift comparison heatmap."""
        logger.info("Creating layer-wise drift heatmap...")
        
        # Extract data for heatmap
        tasks = []
        methods = ['full_finetune', 'lora']
        drift_data = {method: [] for method in methods}
        
        # Collect drift data for each task and method
        for task, task_results in self.results['task_results'].items():
            if 'all_results' in task_results:
                tasks.append(task.upper())
                
                for method in methods:
                    method_results = task_results['all_results'][method]
                    # Average across seeds
                    method_drifts = []
                    for seed_result in method_results:
                        if 'summary_statistics' in seed_result:
                            drift = seed_result['summary_statistics'].get('mean_drift', np.nan)
                            if not np.isnan(drift):
                                method_drifts.append(drift)
                    
                    drift_data[method].append(np.mean(method_drifts) if method_drifts else np.nan)
        
        if not tasks:
            logger.warning("No valid data for heatmap")
            return
        
        # Create heatmap data
        heatmap_data = np.array([drift_data['full_finetune'], drift_data['lora']]).T
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = sns.heatmap(
            heatmap_data,
            xticklabels=['Full Fine-tuning', 'LoRA'],
            yticklabels=tasks,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean Representation Drift'},
            ax=ax
        )
        
        plt.title('Representational Drift Comparison\nLoRA vs Full Fine-tuning (Lower = Better Preservation)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Fine-tuning Method', fontsize=12)
        plt.ylabel('Task', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'layer_drift_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Layer drift heatmap saved: {output_file}")
    
    def create_drift_reduction_plot(self):
        """Create drift reduction comparison plot."""
        logger.info("Creating drift reduction comparison plot...")
        
        # Extract drift reduction data
        tasks = []
        reductions = []
        errors = []
        hypothesis_supported = []
        
        for task, task_results in self.results['task_results'].items():
            if 'comparison_statistics' in task_results:
                stats = task_results['comparison_statistics']
                mean_reduction = stats.get('mean_drift_reduction_percent', np.nan)
                std_reduction = stats.get('std_drift_reduction_percent', 0)
                
                if not np.isnan(mean_reduction):
                    tasks.append(task.upper())
                    reductions.append(mean_reduction)
                    errors.append(std_reduction)
                    hypothesis_supported.append(mean_reduction > 20.0)
        
        if not tasks:
            logger.warning("No valid data for drift reduction plot")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on hypothesis support
        colors = ['forestgreen' if supported else 'orangered' for supported in hypothesis_supported]
        
        bars = ax.bar(range(len(tasks)), reductions, yerr=errors, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add reference line at 20%
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                  label='Research Hypothesis: 20% reduction')
        
        # Customize plot
        ax.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drift Reduction (%)', fontsize=12, fontweight='bold')
        ax.set_title('LoRA Representation Preservation vs Full Fine-tuning\n(Higher = Better Preservation)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, reduction, error, supported) in enumerate(zip(bars, reductions, errors, hypothesis_supported)):
            height = bar.get_height()
            label_y = height + error + max(reductions) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold',
                   color='darkgreen' if supported else 'darkred')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'drift_reduction_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Drift reduction plot saved: {output_file}")
    
    def create_layer_wise_evolution_plot(self):
        """Create layer-wise drift evolution plot."""
        logger.info("Creating layer-wise drift evolution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        task_idx = 0
        for task, task_results in self.results['task_results'].items():
            if task_idx >= 4:  # Only plot first 4 tasks
                break
                
            if 'all_results' in task_results:
                ax = axes[task_idx]
                
                # Prepare data for this task
                layers = list(range(24))
                
                for method in ['full_finetune', 'lora']:
                    method_results = task_results['all_results'][method]
                    
                    # Average drift across seeds for each layer
                    layer_drifts = []
                    layer_errors = []
                    
                    for layer_idx in range(24):
                        layer_name = f'layer_{layer_idx}'
                        seed_drifts = []
                        
                        for seed_result in method_results:
                            if 'layer_wise_drift' in seed_result and layer_name in seed_result['layer_wise_drift']:
                                drift = seed_result['layer_wise_drift'][layer_name]['drift']
                                seed_drifts.append(drift)
                        
                        if seed_drifts:
                            layer_drifts.append(np.mean(seed_drifts))
                            layer_errors.append(np.std(seed_drifts))
                        else:
                            layer_drifts.append(np.nan)
                            layer_errors.append(0)
                    
                    # Plot this method
                    label = method.replace('_', ' ').title()
                    color = 'red' if method == 'full_finetune' else 'blue'
                    ax.plot(layers, layer_drifts, marker='o', label=label, color=color, alpha=0.8)
                    ax.fill_between(layers, 
                                   np.array(layer_drifts) - np.array(layer_errors),
                                   np.array(layer_drifts) + np.array(layer_errors),
                                   alpha=0.2, color=color)
                
                # Customize subplot
                ax.set_title(f'{task.upper()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Transformer Layer')
                ax.set_ylabel('Representation Drift')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                task_idx += 1
        
        # Remove unused subplots
        for i in range(task_idx, 4):
            fig.delaxes(axes[i])
        
        plt.suptitle('Layer-wise Representation Drift Evolution\nAcross Tasks and Methods', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'layer_wise_drift_evolution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Layer-wise evolution plot saved: {output_file}")
    
    def create_statistical_summary_plot(self):
        """Create statistical summary and hypothesis testing visualization."""
        logger.info("Creating statistical summary plot...")
        
        # Extract hypothesis testing data
        hypothesis_data = []
        
        for task, task_results in self.results['task_results'].items():
            if 'comparison_statistics' in task_results:
                stats = task_results['comparison_statistics']
                
                if 'significance_test' in stats:
                    sig_test = stats['significance_test']
                    hypothesis_data.append({
                        'task': task.upper(),
                        'drift_reduction': stats['mean_drift_reduction_percent'],
                        'p_value': sig_test['p_value'],
                        'significant': sig_test['significant_at_05'],
                        'hypothesis_supported': sig_test['hypothesis_20_percent']
                    })
        
        if not hypothesis_data:
            logger.warning("No statistical data available for summary plot")
            return
        
        # Create statistical summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Drift reduction vs significance
        tasks = [d['task'] for d in hypothesis_data]
        reductions = [d['drift_reduction'] for d in hypothesis_data]
        p_values = [d['p_value'] for d in hypothesis_data]
        significant = [d['significant'] for d in hypothesis_data]
        
        # Scatter plot with significance coloring
        colors = ['green' if sig else 'red' for sig in significant]
        scatter = ax1.scatter(reductions, [-np.log10(p) for p in p_values], 
                            c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add task labels
        for i, task in enumerate(tasks):
            ax1.annotate(task, (reductions[i], -np.log10(p_values[i])), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Reference lines
        ax1.axvline(x=20, color='blue', linestyle='--', alpha=0.7, label='Hypothesis: 20% reduction')
        ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='Significance: p=0.05')
        
        ax1.set_xlabel('Drift Reduction (%)', fontsize=12)
        ax1.set_ylabel('-log₁₀(p-value)', fontsize=12)
        ax1.set_title('Statistical Significance vs Effect Size', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Summary statistics
        supported_count = sum(1 for d in hypothesis_data if d['hypothesis_supported'])
        significant_count = sum(1 for d in hypothesis_data if d['significant'])
        total_tasks = len(hypothesis_data)
        
        categories = ['Hypothesis\nSupported\n(>20% reduction)', 'Statistically\nSignificant\n(p<0.05)']
        values = [supported_count/total_tasks * 100, significant_count/total_tasks * 100]
        
        bars = ax2.bar(categories, values, color=['forestgreen', 'steelblue'], alpha=0.8)
        ax2.set_ylabel('Percentage of Tasks (%)', fontsize=12)
        ax2.set_title('Research Hypothesis Validation', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{value:.1f}%\n({int(value*total_tasks/100)}/{total_tasks})',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'statistical_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Statistical summary plot saved: {output_file}")
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard with all key visualizations."""
        logger.info("Creating comprehensive dashboard...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main drift reduction plot (top, spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        self._plot_main_drift_reduction(ax_main)
        
        # 2. Statistical significance plot (top right)
        ax_stats = fig.add_subplot(gs[0, 2])
        self._plot_statistical_significance(ax_stats)
        
        # 3. Layer-wise drift heatmap (middle, full width)
        ax_heatmap = fig.add_subplot(gs[1, :])
        self._plot_layer_wise_heatmap(ax_heatmap)
        
        # 4. Task-specific comparisons (bottom row)
        self._plot_task_comparisons(fig, gs[2, :])
        
        # Overall title
        fig.suptitle('LoRA vs Full Fine-tuning: Comprehensive Drift Analysis\nRepresentational Preservation Across Tasks', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save dashboard
        output_file = self.output_dir / 'comprehensive_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Comprehensive dashboard saved: {output_file}")
    
    def _plot_main_drift_reduction(self, ax):
        """Plot main drift reduction comparison."""
        tasks = []
        reductions = []
        errors = []
        colors = []
        
        for task, task_results in self.results['task_results'].items():
            if 'comparison_statistics' in task_results:
                stats = task_results['comparison_statistics']
                reduction = stats.get('mean_drift_reduction_percent', np.nan)
                error = stats.get('std_drift_reduction_percent', 0)
                
                if not np.isnan(reduction):
                    tasks.append(task.upper())
                    reductions.append(reduction)
                    errors.append(error)
                    colors.append('forestgreen' if reduction > 20 else 'orangered')
        
        if tasks:
            bars = ax.bar(tasks, reductions, yerr=errors, capsize=5, 
                         color=colors, alpha=0.8, edgecolor='black')
            
            ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, linewidth=2,
                      label='Research Target: 20% reduction')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            ax.set_ylabel('Drift Reduction (%)', fontweight='bold')
            ax.set_title('LoRA Representation Preservation vs Full Fine-tuning', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, reduction, error in zip(bars, reductions, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error + 1,
                       f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_statistical_significance(self, ax):
        """Plot statistical significance summary."""
        # Count significant results
        total_tasks = len([t for t in self.results['task_results'].keys() 
                          if 'comparison_statistics' in self.results['task_results'][t]])
        
        significant_count = 0
        hypothesis_count = 0
        
        for task_results in self.results['task_results'].values():
            if 'comparison_statistics' in task_results:
                stats = task_results['comparison_statistics']
                if 'significance_test' in stats:
                    sig_test = stats['significance_test']
                    if sig_test['significant_at_05']:
                        significant_count += 1
                    if sig_test['hypothesis_20_percent']:
                        hypothesis_count += 1
        
        # Create pie chart
        labels = ['Hypothesis\nSupported', 'Not Supported']
        sizes = [hypothesis_count, total_tasks - hypothesis_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.1, 0)
        
        ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
               startangle=90, textprops={'fontweight': 'bold'})
        ax.set_title(f'Research Hypothesis\nValidation\n(n={total_tasks} tasks)', fontweight='bold')
    
    def _plot_layer_wise_heatmap(self, ax):
        """Plot layer-wise drift heatmap."""
        # This is a simplified version - can be expanded
        ax.text(0.5, 0.5, 'Layer-wise Drift Heatmap\n(To be implemented with layer data)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Layer-wise Representation Drift Analysis', fontweight='bold')
    
    def _plot_task_comparisons(self, fig, gs):
        """Plot individual task comparisons."""
        # Simplified implementation - can be expanded
        ax = fig.add_subplot(gs)
        ax.text(0.5, 0.5, 'Individual Task Comparisons\n(To be implemented)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Task-Specific Drift Patterns', fontweight='bold')
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        logger.info("🎨 Generating all drift analysis visualizations...")
        
        try:
            self.create_layer_drift_heatmap()
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
        
        try:
            self.create_drift_reduction_plot()
        except Exception as e:
            logger.error(f"Failed to create drift reduction plot: {e}")
        
        try:
            self.create_statistical_summary_plot()
        except Exception as e:
            logger.error(f"Failed to create statistical summary: {e}")
        
        try:
            self.create_comprehensive_dashboard()
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
        
        logger.info("✅ All visualizations generated!")
        logger.info(f"📁 Output directory: {self.output_dir}")


def main():
    """Main function for visualization generation."""
    parser = argparse.ArgumentParser(description="Generate drift analysis visualizations")
    parser.add_argument("--results-file", type=str, 
                       default="results/drift_analysis/drift_analysis_results.json",
                       help="Path to drift analysis results JSON file")
    parser.add_argument("--output-dir", type=str, default="results/drift_visualizations",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Check if results file exists
    results_file = Path(args.results_file)
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Run analyze_drift.py first to generate analysis results")
        return
    
    # Initialize visualizer
    visualizer = DriftVisualizer(
        results_file=results_file,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    logger.info("🎨 Drift analysis visualization generation complete!")


if __name__ == "__main__":
    main()
