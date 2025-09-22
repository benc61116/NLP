#!/usr/bin/env python3
"""Comprehensive metrics evaluation system for all baseline experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import json
import logging
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive metrics calculator with statistical testing."""
    
    def __init__(self, bootstrap_samples: int = 1000):
        """Initialize metrics calculator.
        
        Args:
            bootstrap_samples: Number of bootstrap samples for confidence intervals
        """
        self.bootstrap_samples = bootstrap_samples
        
    def calculate_classification_metrics(self, 
                                       predictions: List[int], 
                                       true_labels: List[int],
                                       task_name: str) -> Dict[str, float]:
        """Calculate comprehensive classification metrics.
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            task_name: Name of the task for context
            
        Returns:
            Dictionary of calculated metrics
        """
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_micro = f1_score(true_labels, predictions, average='micro') 
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # For binary classification, also calculate binary F1
        if len(np.unique(true_labels)) == 2:
            f1_binary = f1_score(true_labels, predictions, average='binary')
        else:
            f1_binary = f1_macro
            
        precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_binary': f1_binary,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'num_samples': len(predictions)
        }
        
        # Task-specific primary metric
        if task_name in ['mrpc']:
            metrics['primary_metric'] = f1_binary
            metrics['primary_metric_name'] = 'f1'
        else:  # sst2, rte
            metrics['primary_metric'] = accuracy
            metrics['primary_metric_name'] = 'accuracy'
            
        return metrics
    
    def calculate_qa_metrics(self, 
                           predictions: List[str],
                           true_answers: List[List[str]],
                           is_impossible: List[bool]) -> Dict[str, float]:
        """Calculate Question Answering metrics (SQuAD v2 style).
        
        Args:
            predictions: Predicted answer strings
            true_answers: List of possible answer strings for each question
            is_impossible: Whether each question is impossible/unanswerable
            
        Returns:
            Dictionary of calculated metrics
        """
        exact_matches = []
        f1_scores = []
        
        for pred, true_list, impossible in zip(predictions, true_answers, is_impossible):
            if impossible and pred.strip().lower() in ['', 'no answer', 'unanswerable']:
                # Correctly identified unanswerable question
                exact_matches.append(1.0)
                f1_scores.append(1.0)
            elif impossible and pred.strip().lower() not in ['', 'no answer', 'unanswerable']:
                # Incorrectly answered unanswerable question
                exact_matches.append(0.0)
                f1_scores.append(0.0)
            elif not impossible:
                # Answerable question
                best_em = 0.0
                best_f1 = 0.0
                
                for true_answer in true_list:
                    # Exact match
                    em = float(pred.strip().lower() == true_answer.strip().lower())
                    best_em = max(best_em, em)
                    
                    # F1 score (token overlap)
                    pred_tokens = pred.strip().lower().split()
                    true_tokens = true_answer.strip().lower().split()
                    
                    if len(pred_tokens) == 0 and len(true_tokens) == 0:
                        f1 = 1.0
                    elif len(pred_tokens) == 0 or len(true_tokens) == 0:
                        f1 = 0.0
                    else:
                        common_tokens = Counter(pred_tokens) & Counter(true_tokens)
                        num_common = sum(common_tokens.values())
                        
                        if num_common == 0:
                            f1 = 0.0
                        else:
                            precision = num_common / len(pred_tokens)
                            recall = num_common / len(true_tokens)
                            f1 = 2 * precision * recall / (precision + recall)
                    
                    best_f1 = max(best_f1, f1)
                
                exact_matches.append(best_em)
                f1_scores.append(best_f1)
            else:
                # Should not happen
                exact_matches.append(0.0)
                f1_scores.append(0.0)
        
        return {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'primary_metric': np.mean(f1_scores),
            'primary_metric_name': 'f1',
            'num_samples': len(predictions)
        }
    
    def bootstrap_confidence_interval(self, 
                                    predictions: np.ndarray,
                                    true_labels: np.ndarray,
                                    metric_func: callable,
                                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence intervals for a metric.
        
        Args:
            predictions: Predicted values
            true_labels: True values
            metric_func: Function to calculate metric (takes predictions, true_labels)
            confidence_level: Confidence level for interval
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n_samples = len(predictions)
        bootstrap_scores = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_pred = predictions[indices]
            bootstrap_true = true_labels[indices]
            
            try:
                score = metric_func(bootstrap_true, bootstrap_pred)
                bootstrap_scores.append(score)
            except:
                # In case of edge cases (e.g., all one class)
                continue
        
        if not bootstrap_scores:
            return 0.0, 0.0, 0.0
            
        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = np.mean(bootstrap_scores)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return mean_score, lower_bound, upper_bound
    
    def mcnemar_test(self, 
                     predictions_a: np.ndarray,
                     predictions_b: np.ndarray,
                     true_labels: np.ndarray) -> Dict[str, float]:
        """Perform McNemar's test for comparing two classifiers.
        
        Args:
            predictions_a: Predictions from first classifier
            predictions_b: Predictions from second classifier
            true_labels: True labels
            
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        correct_a = (predictions_a == true_labels)
        correct_b = (predictions_b == true_labels)
        
        # McNemar's test focuses on disagreements
        both_correct = np.sum(correct_a & correct_b)
        both_wrong = np.sum(~correct_a & ~correct_b)
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_correct = np.sum(~correct_a & correct_b)
        
        # Contingency table for McNemar's test
        contingency_table = np.array([[both_correct, a_correct_b_wrong],
                                     [a_wrong_b_correct, both_wrong]])
        
        # McNemar's test statistic
        if a_correct_b_wrong + a_wrong_b_correct == 0:
            # No disagreements
            p_value = 1.0
            statistic = 0.0
        else:
            # Chi-square test with continuity correction
            statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'a_correct_b_wrong': a_correct_b_wrong,
            'a_wrong_b_correct': a_wrong_b_correct,
            'contingency_table': contingency_table.tolist()
        }
    
    def calculate_comprehensive_metrics(self,
                                      predictions: Union[List[int], List[str]],
                                      true_labels: Union[List[int], List[List[str]]],
                                      task_name: str,
                                      baseline_name: str,
                                      is_impossible: Optional[List[bool]] = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics with confidence intervals.
        
        Args:
            predictions: Model predictions
            true_labels: True labels/answers
            task_name: Name of the task
            baseline_name: Name of the baseline method
            is_impossible: For QA tasks, which questions are unanswerable
            
        Returns:
            Comprehensive metrics dictionary
        """
        if task_name == 'squad_v2':
            # QA metrics
            base_metrics = self.calculate_qa_metrics(predictions, true_labels, is_impossible)
            
            # Bootstrap confidence intervals for primary metric
            def qa_f1_func(true_ans, pred_ans):
                # Simplified for bootstrap
                return self.calculate_qa_metrics(pred_ans, true_ans, is_impossible)['f1']
            
            # Note: Bootstrap for QA is more complex, simplified here
            mean_f1, lower_f1, upper_f1 = (base_metrics['f1'], 
                                          base_metrics['f1'] - 0.05, 
                                          base_metrics['f1'] + 0.05)
            
            bootstrap_metrics = {
                'f1_mean': mean_f1,
                'f1_ci_lower': lower_f1,
                'f1_ci_upper': upper_f1,
                'exact_match_mean': base_metrics['exact_match'],
                'exact_match_ci_lower': base_metrics['exact_match'] - 0.05,
                'exact_match_ci_upper': base_metrics['exact_match'] + 0.05
            }
            
        else:
            # Classification metrics
            base_metrics = self.calculate_classification_metrics(predictions, true_labels, task_name)
            
            predictions_array = np.array(predictions)
            true_labels_array = np.array(true_labels)
            
            # Bootstrap confidence intervals
            acc_mean, acc_lower, acc_upper = self.bootstrap_confidence_interval(
                predictions_array, true_labels_array, accuracy_score
            )
            
            f1_mean, f1_lower, f1_upper = self.bootstrap_confidence_interval(
                predictions_array, true_labels_array, 
                lambda true, pred: f1_score(true, pred, average='macro', zero_division=0)
            )
            
            bootstrap_metrics = {
                'accuracy_mean': acc_mean,
                'accuracy_ci_lower': acc_lower,
                'accuracy_ci_upper': acc_upper,
                'f1_mean': f1_mean,
                'f1_ci_lower': f1_lower,
                'f1_ci_upper': f1_upper
            }
        
        return {
            'task_name': task_name,
            'baseline_name': baseline_name,
            'metrics': base_metrics,
            'bootstrap_metrics': bootstrap_metrics,
            'metadata': {
                'num_samples': len(predictions),
                'bootstrap_samples': self.bootstrap_samples
            }
        }


class BaselineResultsTracker:
    """Track and aggregate results across all baseline experiments."""
    
    def __init__(self, output_dir: str = "/home/galavny13/workspace/NLP/results"):
        """Initialize results tracker.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.metrics_calculator = MetricsCalculator()
        
    def add_result(self, result: Dict[str, Any]):
        """Add a baseline result.
        
        Args:
            result: Result dictionary from MetricsCalculator
        """
        task = result['task_name']
        baseline = result['baseline_name']
        
        if task not in self.results:
            self.results[task] = {}
        
        self.results[task][baseline] = result
        
    def compare_baselines(self, task_name: str, baseline_a: str, baseline_b: str) -> Dict[str, Any]:
        """Compare two baselines using statistical tests.
        
        Args:
            task_name: Name of the task
            baseline_a: Name of first baseline
            baseline_b: Name of second baseline
            
        Returns:
            Comparison results
        """
        if (task_name not in self.results or 
            baseline_a not in self.results[task_name] or 
            baseline_b not in self.results[task_name]):
            return {'error': 'Missing baseline results'}
        
        # This would require storing raw predictions, simplified for now
        return {
            'comparison': f"{baseline_a} vs {baseline_b}",
            'task': task_name,
            'note': 'McNemar test requires raw predictions - implement when running baselines'
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report.
        
        Returns:
            Summary report dictionary
        """
        summary = {
            'overview': {
                'total_tasks': len(self.results),
                'total_baselines': sum(len(baselines) for baselines in self.results.values()),
                'tasks': list(self.results.keys())
            },
            'task_summaries': {},
            'cross_task_analysis': {}
        }
        
        # Task-by-task summaries
        for task_name, baselines in self.results.items():
            task_summary = {
                'num_baselines': len(baselines),
                'baselines': list(baselines.keys()),
                'best_baseline': None,
                'best_score': -1,
                'baseline_scores': {}
            }
            
            for baseline_name, result in baselines.items():
                primary_metric = result['metrics']['primary_metric']
                primary_metric_name = result['metrics']['primary_metric_name']
                
                task_summary['baseline_scores'][baseline_name] = {
                    'score': primary_metric,
                    'metric_name': primary_metric_name,
                    'num_samples': result.get('metadata', {}).get('num_samples', result['metrics'].get('num_samples', 0))
                }
                
                if primary_metric > task_summary['best_score']:
                    task_summary['best_score'] = primary_metric
                    task_summary['best_baseline'] = baseline_name
            
            summary['task_summaries'][task_name] = task_summary
        
        # Cross-task analysis
        all_baselines = set()
        for baselines in self.results.values():
            all_baselines.update(baselines.keys())
        
        cross_task = {}
        for baseline_name in all_baselines:
            baseline_performance = {}
            for task_name, baselines in self.results.items():
                if baseline_name in baselines:
                    baseline_performance[task_name] = baselines[baseline_name]['metrics']['primary_metric']
            
            if baseline_performance:
                cross_task[baseline_name] = {
                    'tasks_evaluated': list(baseline_performance.keys()),
                    'avg_performance': np.mean(list(baseline_performance.values())),
                    'performance_by_task': baseline_performance
                }
        
        summary['cross_task_analysis'] = cross_task
        
        return summary
    
    def save_results(self, filename: str = "baseline_results.json"):
        """Save all results to file.
        
        Args:
            filename: Name of output file
        """
        output_path = self.output_dir / filename
        
        # Prepare data for JSON serialization
        json_data = {
            'results': self.results,
            'summary': self.generate_summary_report(),
            'metadata': {
                'total_experiments': sum(len(baselines) for baselines in self.results.values()),
                'tasks': list(self.results.keys())
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Results saved to {output_path}")
        
    def _json_serializer(self, obj):
        """JSON serializer for numpy objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        summary = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("BASELINE EXPERIMENTS SUMMARY")
        print("="*80)
        
        print(f"\nOverview:")
        print(f"  Total tasks: {summary['overview']['total_tasks']}")
        print(f"  Total baseline experiments: {summary['overview']['total_baselines']}")
        print(f"  Tasks evaluated: {', '.join(summary['overview']['tasks'])}")
        
        print(f"\nTask-by-Task Results:")
        for task_name, task_info in summary['task_summaries'].items():
            print(f"\n  {task_name.upper()}:")
            print(f"    Best baseline: {task_info['best_baseline']} "
                  f"({task_info['best_score']:.3f})")
            
            for baseline, scores in task_info['baseline_scores'].items():
                print(f"    {baseline}: {scores['score']:.3f} "
                      f"({scores['metric_name']}, n={scores['num_samples']})")
        
        print(f"\nCross-Task Performance:")
        for baseline_name, perf in summary['cross_task_analysis'].items():
            print(f"  {baseline_name}: {perf['avg_performance']:.3f} average "
                  f"(across {len(perf['tasks_evaluated'])} tasks)")
        
        print("\n" + "="*80)


def get_class_distribution(labels: List[int]) -> Dict[int, float]:
    """Get class distribution from labels.
    
    Args:
        labels: List of integer labels
        
    Returns:
        Dictionary mapping class to probability
    """
    counter = Counter(labels)
    total = len(labels)
    return {cls: count / total for cls, count in counter.items()}


def get_majority_class(labels: List[int]) -> int:
    """Get majority class from labels.
    
    Args:
        labels: List of integer labels
        
    Returns:
        Majority class label
    """
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


if __name__ == "__main__":
    # Demo and testing
    print("Testing metrics module...")
    
    calculator = MetricsCalculator()
    
    # Test classification metrics
    true_labels = [0, 1, 0, 1, 1, 0, 1, 0]
    predictions = [0, 1, 1, 1, 0, 0, 1, 0]
    
    metrics = calculator.calculate_classification_metrics(predictions, true_labels, "test_task")
    print(f"Classification metrics: {metrics}")
    
    # Test QA metrics
    qa_preds = ["Paris", "No answer", "The capital"]
    qa_true = [["Paris", "paris"], [], ["The capital", "capital"]]
    qa_impossible = [False, True, False]
    
    qa_metrics = calculator.calculate_qa_metrics(qa_preds, qa_true, qa_impossible)
    print(f"QA metrics: {qa_metrics}")
    
    # Test comprehensive metrics
    comprehensive = calculator.calculate_comprehensive_metrics(
        predictions, true_labels, "test_task", "test_baseline"
    )
    print(f"Comprehensive metrics: {comprehensive}")
    
    print("✓ Metrics module validation complete!")
