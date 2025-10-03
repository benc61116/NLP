#!/usr/bin/env python3
"""
Comprehensive Drift Analysis for LoRA vs Full Fine-tuning Research

This script implements the core research analysis comparing representational
drift between LoRA and full fine-tuning methods across all tasks.
"""

import os
import sys
import numpy as np
import torch
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.metrics import RepresentationMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """Comprehensive drift analysis comparing LoRA vs Full Fine-tuning."""
    
    def __init__(self, representations_dir: Path, base_representations_dir: Path, output_dir: Path):
        """Initialize drift analyzer.
        
        Args:
            representations_dir: Directory with Phase 3 extracted representations
            base_representations_dir: Directory with base model representations
            output_dir: Output directory for analysis results
        """
        self.representations_dir = Path(representations_dir)
        self.base_representations_dir = Path(base_representations_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DriftAnalyzer")
        logger.info(f"  Representations: {self.representations_dir}")
        logger.info(f"  Base representations: {self.base_representations_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def load_base_representations(self, task: str) -> Dict[str, np.ndarray]:
        """Load base model representations for a task (extracted in Phase 0)."""
        # Try multiple possible locations
        possible_dirs = [
            self.base_representations_dir / f"base_pretrained_{task}" / "step_000000",
            self.base_representations_dir / task / "step_000000",
            Path("base_representations") / "representations" / f"base_pretrained_{task}" / "step_000000",
            Path("base_representations") / task / "step_000000",
        ]
        
        base_task_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                base_task_dir = dir_path
                break
        
        if base_task_dir is None:
            raise FileNotFoundError(
                f"Base representations not found for {task}. "
                f"Tried locations: {possible_dirs}"
            )
        
        representations = {}
        
        # Load all layer representations (TinyLlama has 22 layers: 0-21)
        for layer_idx in range(22):
            layer_file = base_task_dir / f"layer_{layer_idx}.pt"
            if layer_file.exists():
                tensor = torch.load(layer_file, map_location='cpu')
                # Convert bfloat16 to float32 for numpy compatibility
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                representations[f'layer_{layer_idx}'] = tensor.numpy()
        
        # Load final hidden states if available
        final_file = base_task_dir / "final_hidden_states.pt"
        if final_file.exists():
            tensor = torch.load(final_file, map_location='cpu')
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            representations['final_hidden_states'] = tensor.numpy()
        
        logger.info(f"Loaded base representations for {task} from {base_task_dir}: {len(representations)} layers")
        return representations
    
    def load_finetuned_representations(self, task: str, method: str, seed: int) -> Dict[str, np.ndarray]:
        """Load fine-tuned model representations."""
        # Try both possible naming conventions
        possible_dirs = [
            self.representations_dir / f"{method}_seed{seed}_{task}" / "step_000000",
            self.representations_dir / f"{task}_{method}_seed{seed}" / "step_000000",
        ]
        
        repr_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                repr_dir = dir_path
                break
        
        if repr_dir is None:
            raise FileNotFoundError(
                f"Fine-tuned representations not found for {task}/{method}/seed{seed}. "
                f"Tried: {possible_dirs}"
            )
        
        representations = {}
        
        # Load all layer representations (TinyLlama has 22 layers: 0-21)
        for layer_idx in range(22):
            layer_file = repr_dir / f"layer_{layer_idx}.pt"
            if layer_file.exists():
                tensor = torch.load(layer_file, map_location='cpu')
                # Convert bfloat16 to float32 for numpy compatibility
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                representations[f'layer_{layer_idx}'] = tensor.numpy()
        
        # Load final hidden states if available
        final_file = repr_dir / "final_hidden_states.pt"
        if final_file.exists():
            tensor = torch.load(final_file, map_location='cpu')
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            representations['final_hidden_states'] = tensor.numpy()
        
        logger.info(f"Loaded {method} representations for {task} (seed {seed}) from {repr_dir}: {len(representations)} layers")
        return representations
    
    def analyze_task_drift(self, task: str, method: str, seed: int) -> Dict[str, float]:
        """Analyze drift for a specific task/method/seed combination."""
        logger.info(f"Analyzing drift: {task}/{method}/seed{seed}")
        
        # Load representations
        base_reprs = self.load_base_representations(task)
        ft_reprs = self.load_finetuned_representations(task, method, seed)
        
        results = {
            'task': task,
            'method': method,
            'seed': seed,
            'layer_wise_drift': {},
            'summary_statistics': {}
        }
        
        # Analyze each layer
        layer_drifts = []
        layer_ckas = []
        layer_cosines = []
        
        for layer_idx in range(22):
            layer_name = f'layer_{layer_idx}'
            
            if layer_name in base_reprs and layer_name in ft_reprs:
                base_layer = base_reprs[layer_name]
                ft_layer = ft_reprs[layer_name]
                
                # Ensure shapes match (handle padding differences)
                min_samples = min(base_layer.shape[0], ft_layer.shape[0])
                base_layer = base_layer[:min_samples]
                ft_layer = ft_layer[:min_samples]
                
                # CRITICAL: Mean-pool over token dimension (axis=1) to avoid OOM
                # Shapes: (samples, tokens, hidden_dim) -> (samples, hidden_dim)
                if base_layer.ndim == 3:
                    base_layer = base_layer.mean(axis=1)
                if ft_layer.ndim == 3:
                    ft_layer = ft_layer.mean(axis=1)
                
                # Compute CKA and cosine similarity
                try:
                    cka = RepresentationMetrics.compute_centered_kernel_alignment(base_layer, ft_layer)
                    cos_sim = RepresentationMetrics.compute_cosine_similarity(base_layer, ft_layer)
                    drift = 1.0 - cka
                    
                    results['layer_wise_drift'][layer_name] = {
                        'cka': float(cka),
                        'cosine_similarity': float(cos_sim),
                        'drift': float(drift)
                    }
                    
                    layer_drifts.append(drift)
                    layer_ckas.append(cka)
                    layer_cosines.append(cos_sim)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute metrics for {layer_name}: {e}")
                    results['layer_wise_drift'][layer_name] = {
                        'cka': 0.0,
                        'cosine_similarity': 0.0, 
                        'drift': 1.0,
                        'error': str(e)
                    }
        
        # Compute summary statistics
        if layer_drifts:
            results['summary_statistics'] = {
                'mean_drift': float(np.mean(layer_drifts)),
                'std_drift': float(np.std(layer_drifts)),
                'median_drift': float(np.median(layer_drifts)),
                'max_drift': float(np.max(layer_drifts)),
                'min_drift': float(np.min(layer_drifts)),
                'mean_cka': float(np.mean(layer_ckas)),
                'mean_cosine': float(np.mean(layer_cosines)),
                'num_layers_analyzed': len(layer_drifts)
            }
        
        return results
    
    def compare_methods(self, task: str, seeds: List[int] = [42, 1337, 2024]) -> Dict:
        """Compare drift between LoRA and Full FT for a task across seeds."""
        logger.info(f"Comparing methods for {task} across {len(seeds)} seeds")
        
        # Analyze both methods across all seeds
        all_results = {'full_finetune': [], 'lora': []}
        
        for seed in seeds:
            for method in ['full_finetune', 'lora']:
                try:
                    result = self.analyze_task_drift(task, method, seed)
                    all_results[method].append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze {task}/{method}/seed{seed}: {e}")
                    # Add placeholder result
                    all_results[method].append({
                        'task': task,
                        'method': method,
                        'seed': seed,
                        'summary_statistics': {'mean_drift': np.nan},
                        'error': str(e)
                    })
        
        # Calculate drift reduction statistics
        drift_reductions = []
        valid_comparisons = 0
        
        for i, seed in enumerate(seeds):
            ft_result = all_results['full_finetune'][i]
            lora_result = all_results['lora'][i]
            
            ft_drift = ft_result.get('summary_statistics', {}).get('mean_drift', np.nan)
            lora_drift = lora_result.get('summary_statistics', {}).get('mean_drift', np.nan)
            
            if not (np.isnan(ft_drift) or np.isnan(lora_drift)) and ft_drift > 0:
                reduction = (ft_drift - lora_drift) / ft_drift * 100
                drift_reductions.append(reduction)
                valid_comparisons += 1
        
        # Compute comparison statistics
        comparison_stats = {}
        if drift_reductions:
            comparison_stats = {
                'mean_drift_reduction_percent': float(np.mean(drift_reductions)),
                'std_drift_reduction_percent': float(np.std(drift_reductions)),
                'median_drift_reduction_percent': float(np.median(drift_reductions)),
                'drift_reductions_by_seed': drift_reductions,
                'valid_comparisons': valid_comparisons,
                'total_seeds': len(seeds)
            }
            
            # Statistical significance test
            if len(drift_reductions) >= 2:
                # Test if drift reduction is significantly greater than 0
                t_stat, p_value = stats.ttest_1samp(drift_reductions, 0, alternative='greater')
                comparison_stats['significance_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_05': bool(p_value < 0.05),
                    'hypothesis_20_percent': bool(float(np.mean(drift_reductions)) > 20.0)
                }
        
        return {
            'task': task,
            'comparison_type': 'lora_vs_full_finetune',
            'all_results': all_results,
            'comparison_statistics': comparison_stats,
            'research_hypothesis': {
                'target_drift_reduction': 20.0,  # 20% target from research plan
                'achieved_reduction': comparison_stats.get('mean_drift_reduction_percent', np.nan),
                'hypothesis_supported': bool(comparison_stats.get('mean_drift_reduction_percent', 0) > 20.0)
            }
        }
    
    def analyze_all_tasks(self, tasks: List[str] = ['mrpc', 'sst2', 'rte', 'squad_v2'], 
                         seeds: List[int] = [42, 1337, 2024]) -> Dict:
        """Analyze drift across all tasks and generate comprehensive report."""
        logger.info(f"Running comprehensive drift analysis across {len(tasks)} tasks")
        
        all_task_results = {}
        
        for task in tasks:
            logger.info(f"Processing task: {task}")
            try:
                task_results = self.compare_methods(task, seeds)
                all_task_results[task] = task_results
                
                # Log key results
                stats = task_results.get('comparison_statistics', {})
                reduction = stats.get('mean_drift_reduction_percent', np.nan)
                logger.info(f"  {task}: {reduction:.1f}% drift reduction (LoRA vs Full FT)")
                
            except Exception as e:
                logger.error(f"Failed to analyze {task}: {e}")
                all_task_results[task] = {'error': str(e)}
        
        # Cross-task summary
        valid_reductions = []
        task_summary = {}
        
        for task, results in all_task_results.items():
            if 'comparison_statistics' in results:
                reduction = results['comparison_statistics'].get('mean_drift_reduction_percent', np.nan)
                if not np.isnan(reduction):
                    valid_reductions.append(reduction)
                    task_summary[task] = {
                        'drift_reduction': reduction,
                        'hypothesis_supported': bool(reduction > 20.0)
                    }
        
        # Overall research conclusions
        overall_results = {
            'tasks_analyzed': tasks,
            'seeds_per_task': seeds,
            'task_results': all_task_results,
            'cross_task_summary': {
                'mean_drift_reduction_all_tasks': float(np.mean(valid_reductions)) if valid_reductions else np.nan,
                'std_drift_reduction_all_tasks': float(np.std(valid_reductions)) if valid_reductions else np.nan,
                'tasks_supporting_hypothesis': sum(1 for s in task_summary.values() if s['hypothesis_supported']),
                'total_tasks_analyzed': len(task_summary),
                'overall_hypothesis_supported': bool(len(task_summary) > 0 and sum(1 for s in task_summary.values() if s['hypothesis_supported']) >= len(task_summary) * 0.75)
            }
        }
        
        # Save comprehensive results
        results_file = self.output_dir / "drift_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        logger.info(f"✅ Comprehensive drift analysis saved to: {results_file}")
        
        # Generate summary report
        self.generate_summary_report(overall_results)
        
        return overall_results
    
    def generate_summary_report(self, results: Dict):
        """Generate human-readable summary report."""
        report_file = self.output_dir / "drift_analysis_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE DRIFT ANALYSIS REPORT\n")
            f.write("LoRA vs Full Fine-tuning Representational Preservation\n")
            f.write("="*80 + "\n\n")
            
            # Cross-task summary
            cross_task = results['cross_task_summary']
            f.write(f"OVERALL RESULTS:\n")
            f.write(f"  Mean drift reduction: {cross_task['mean_drift_reduction_all_tasks']:.2f}%\n")
            f.write(f"  Tasks supporting hypothesis (>20%): {cross_task['tasks_supporting_hypothesis']}/{cross_task['total_tasks_analyzed']}\n")
            f.write(f"  Overall hypothesis supported: {cross_task['overall_hypothesis_supported']}\n\n")
            
            # Task-by-task results
            f.write("TASK-BY-TASK RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for task, task_results in results['task_results'].items():
                if 'comparison_statistics' in task_results:
                    stats = task_results['comparison_statistics']
                    f.write(f"{task.upper()}:\n")
                    f.write(f"  Drift reduction: {stats['mean_drift_reduction_percent']:.2f}% ± {stats['std_drift_reduction_percent']:.2f}%\n")
                    
                    if 'significance_test' in stats:
                        sig = stats['significance_test']
                        f.write(f"  Statistical significance: p={sig['p_value']:.4f} {'✅' if sig['significant_at_05'] else '❌'}\n")
                        f.write(f"  Hypothesis (>20%): {'✅' if sig['hypothesis_20_percent'] else '❌'}\n")
                    
                    f.write(f"  Valid comparisons: {stats['valid_comparisons']}/{stats['total_seeds']}\n")
                else:
                    f.write(f"{task.upper()}: ❌ Analysis failed\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("RESEARCH CONCLUSIONS:\n")
            
            if cross_task['overall_hypothesis_supported']:
                f.write("✅ HYPOTHESIS SUPPORTED: LoRA preserves representations better than full fine-tuning\n")
            else:
                f.write("❌ HYPOTHESIS NOT SUPPORTED: LoRA does not show consistent representation preservation advantage\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"📄 Summary report saved to: {report_file}")


def main():
    """Main function for drift analysis."""
    parser = argparse.ArgumentParser(description="Analyze representational drift between LoRA and Full FT")
    parser.add_argument("--task", choices=["mrpc", "sst2", "rte", "all"], 
                       default="all", help="Task to analyze (default: all)")
    parser.add_argument("--representations-dir", type=str, default="results/phase3_representations/representations",
                       help="Directory with extracted representations")
    parser.add_argument("--base-representations-dir", type=str, default="base_representations/representations", 
                       help="Directory with base model representations")
    parser.add_argument("--output-dir", type=str, default="results/drift_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024],
                       help="Seeds to analyze")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DriftAnalyzer(
        representations_dir=args.representations_dir,
        base_representations_dir=args.base_representations_dir,
        output_dir=args.output_dir
    )
    
    if args.task == "all":
        # Analyze all tasks (classification only)
        tasks = ["mrpc", "sst2", "rte"]
        results = analyzer.analyze_all_tasks(tasks, args.seeds)
        
        logger.info("✅ Comprehensive drift analysis completed for all tasks")
        logger.info(f"📊 Results: {analyzer.output_dir}/drift_analysis_results.json")
        logger.info(f"📄 Report: {analyzer.output_dir}/drift_analysis_summary.txt")
        
    else:
        # Analyze single task
        task_results = analyzer.compare_methods(args.task, args.seeds)
        
        # Save single task results
        results_file = analyzer.output_dir / f"drift_analysis_{args.task}.json"
        with open(results_file, 'w') as f:
            json.dump(task_results, f, indent=2)
        
        logger.info(f"✅ Drift analysis completed for {args.task}")
        logger.info(f"📊 Results: {results_file}")


if __name__ == "__main__":
    main()
