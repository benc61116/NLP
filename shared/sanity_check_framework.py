#!/usr/bin/env python3
"""Enhanced sanity check framework with two-stage validation."""

import os
import sys
import logging
import subprocess
import time
import torch
import yaml
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class SanityCheckFramework:
    """Enhanced sanity check framework with overfitting and production validation."""
    
    def __init__(self, config_path: str = None):
        """Initialize the sanity check framework."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
        self.methods = ['lora', 'full_finetune']
        
        logger.info(f"Initialized sanity check framework for {len(self.tasks)} tasks, {len(self.methods)} methods")
    
    def run_overfitting_sanity_check(self, task: str, method: str, seed: int = 42) -> Dict[str, Any]:
        """Run aggressive overfitting sanity check - can the model learn at all?"""
        logger.info(f"Running overfitting sanity check: {task} with {method}")
        
        # Get task-specific timeout
        sanity_config = self.config.get('sanity_check', {})
        task_specific = sanity_config.get('task_specific', {})
        task_config = task_specific.get(task, {})
        timeout = task_config.get('timeout', 300)  # Default 5 minutes
        
        logger.info(f"Using {timeout}s timeout for {task}")
        
        # Determine script to use
        script = "experiments/lora_finetune.py" if method == "lora" else "experiments/full_finetune.py"
        
        # Build command with sanity check flag (current implementation)
        cmd = [
            sys.executable, script,
            "--task", task,
            "--mode", "single",
            "--seed", str(seed),
            "--sanity-check"
        ]
        
        return self._run_sanity_command(cmd, "overfitting", task, method, timeout=timeout)
    
    def run_production_stability_check(self, task: str, method: str, seed: int = 42) -> Dict[str, Any]:
        """Run production stability check - will production config be stable?"""
        logger.info(f"Running production stability check: {task} with {method}")
        
        # Determine script to use
        script = "experiments/lora_finetune.py" if method == "lora" else "experiments/full_finetune.py"
        
        # Build command with production stability flag
        cmd = [
            sys.executable, script,
            "--task", task,
            "--mode", "single", 
            "--seed", str(seed),
            "--production-stability"  # New flag we need to implement
        ]
        
        return self._run_sanity_command(cmd, "production_stability", task, method, timeout=600)
    
    def _run_sanity_command(self, cmd: List[str], check_type: str, task: str, method: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a sanity check command and analyze results."""
        start_time = time.time()
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Analyze the results
            subprocess_success = result.returncode == 0
            
            # Extract key metrics from output
            metrics = self._extract_metrics_from_output(result.stdout, result.stderr)
            
            # Detect gradient explosion or other issues
            issues = self._detect_issues(result.stdout, result.stderr, metrics, check_type)
            
            # CRITICAL FIX: Factor in detected issues for success determination
            critical_issues = [issue for issue in issues if any(critical in issue.lower() for critical in [
                'nan', 'inf', 'gradient explosion', 'no significant learning', 'loss increased'
            ])]
            
            # Success requires both: no crash AND no critical issues
            success = subprocess_success and len(critical_issues) == 0
            
            if critical_issues:
                logger.warning(f"🚨 Critical issues detected for {task}/{method}:")
                for issue in critical_issues:
                    logger.warning(f"   - {issue}")
                logger.warning("   Marking sanity check as FAILED due to critical issues")
            
            result_dict = {
                'success': success,
                'duration_seconds': round(duration, 1),
                'return_code': result.returncode,
                'metrics': metrics,
                'issues': issues,
                'check_type': check_type,
                'task': task,
                'method': method,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                logger.info(f"✅ {check_type} sanity check PASSED for {task}/{method} ({duration:.1f}s)")
                if metrics:
                    logger.info(f"   Final loss: {metrics.get('final_loss', 'N/A')}")
                    logger.info(f"   Max grad norm: {metrics.get('max_grad_norm', 'N/A')}")
            else:
                logger.error(f"❌ {check_type} sanity check FAILED for {task}/{method}")
                logger.error(f"   Return code: {result.returncode}")
                if issues:
                    for issue in issues:
                        logger.error(f"   Issue: {issue}")
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"❌ {check_type} sanity check TIMED OUT for {task}/{method} after {duration:.1f}s")
            return {
                'success': False,
                'duration_seconds': round(duration, 1),
                'return_code': -1,
                'metrics': {},
                'issues': ['Timeout - process killed'],
                'check_type': check_type,
                'task': task,
                'method': method,
                'stdout': '',
                'stderr': 'Process timed out'
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {check_type} sanity check ERROR for {task}/{method}: {e}")
            return {
                'success': False,
                'duration_seconds': round(duration, 1),
                'return_code': -2,
                'metrics': {},
                'issues': [f'Exception: {str(e)}'],
                'check_type': check_type,
                'task': task,
                'method': method,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _extract_metrics_from_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract key metrics from the training output."""
        metrics = {}
        
        # Look for loss values
        lines = stdout.split('\n') + stderr.split('\n')
        
        losses = []
        grad_norms = []
        
        for line in lines:
            # Extract training loss
            if 'loss' in line.lower() and ('train' in line.lower() or 'step' in line.lower()):
                try:
                    # Look for patterns like "loss: 0.123" or "train_loss': 0.123"
                    import re
                    loss_match = re.search(r'loss[\'"]?\s*[:=]\s*([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
                    if loss_match:
                        loss_val = float(loss_match.group(1))
                        if loss_val < 100:  # Filter out obviously wrong values
                            losses.append(loss_val)
                except:
                    continue
            
            # Extract gradient norms
            if 'grad_norm' in line.lower() or 'gradient' in line.lower():
                try:
                    import re
                    grad_match = re.search(r'grad_norm[\'"]?\s*[:=]\s*([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
                    if grad_match:
                        grad_val = float(grad_match.group(1))
                        grad_norms.append(grad_val)
                except:
                    continue
        
        # Calculate metrics
        if losses:
            metrics['initial_loss'] = losses[0]
            metrics['final_loss'] = losses[-1]
            metrics['min_loss'] = min(losses)
            metrics['loss_reduction'] = losses[0] - losses[-1] if len(losses) > 1 else 0
            metrics['loss_values'] = losses[-5:]  # Last 5 values
        
        if grad_norms:
            metrics['max_grad_norm'] = max(grad_norms)
            metrics['avg_grad_norm'] = sum(grad_norms) / len(grad_norms)
            metrics['final_grad_norm'] = grad_norms[-1]
        
        return metrics
    
    def _detect_issues(self, stdout: str, stderr: str, metrics: Dict[str, Any], check_type: str = "") -> List[str]:
        """Detect potential issues from the training output."""
        issues = []
        
        output = (stdout + stderr).lower()
        
        # Check for explicit errors (improved to avoid false positives)
        error_patterns = [
            'nan values',
            'inf values', 
            'gradient explosion',
            'overflow error',
            'underflow error',
            'out of memory',
            'cuda error',
            'runtime error',
            'assertion error'
        ]
        
        for pattern in error_patterns:
            if pattern in output:
                issues.append(f"Detected {pattern} in output")
        
        # Check gradient norms
        if 'max_grad_norm' in metrics:
            if metrics['max_grad_norm'] > 10.0:
                issues.append(f"High gradient norm: {metrics['max_grad_norm']:.2f}")
            elif metrics['max_grad_norm'] > 100.0:
                issues.append(f"Gradient explosion: {metrics['max_grad_norm']:.2f}")
        
        # Check loss behavior (only training loss matters for sanity checks)
        if 'final_loss' in metrics and 'initial_loss' in metrics:
            # For sanity checks, only care if TRAINING loss increased (eval loss can diverge during overfitting)
            loss_increased = metrics['final_loss'] > metrics['initial_loss']
            # Only flag as issue if it's a significant increase (> 10% to avoid noise)
            if loss_increased and (metrics['final_loss'] - metrics['initial_loss']) > 0.1 * metrics['initial_loss']:
                issues.append(f"Training loss increased significantly: {metrics['initial_loss']:.3f} → {metrics['final_loss']:.3f}")
            elif metrics['final_loss'] > 20.0:  # Increased threshold for high loss
                issues.append(f"High final training loss: {metrics['final_loss']:.3f}")
        
        # Check for no learning (different thresholds for different check types)  
        if 'loss_reduction' in metrics:
            # Different expectations for overfitting vs production stability
            if check_type == "overfitting":
                # For overfitting checks, expect aggressive learning
                if abs(metrics['loss_reduction']) < 0.05:
                    issues.append("No significant learning detected")
            elif check_type == "production_stability":
                # For production stability, expect modest but detectable learning
                if abs(metrics['loss_reduction']) < 0.01:
                    issues.append("No significant learning detected")
            else:
                # Default threshold
                if abs(metrics['loss_reduction']) < 0.02:
                    issues.append("No significant learning detected")
        
        return issues
    
    def run_comprehensive_sanity_checks(self, task: str, seeds: List[int] = [42, 1337, 2024]) -> Dict[str, Any]:
        """Run comprehensive sanity checks for a task with both methods and multiple seeds."""
        logger.info(f"Running comprehensive sanity checks for {task}")
        logger.info("=" * 60)
        
        results = {
            'task': task,
            'overall_success': True,
            'method_results': {},
            'summary': {
                'total_checks': 0,
                'passed_checks': 0,
                'failed_checks': 0,
                'issues_found': []
            }
        }
        
        for method in self.methods:
            logger.info(f"\nTesting method: {method}")
            method_results = {
                'overfitting_checks': [],
                'production_checks': [],
                'method_success': True
            }
            
            for seed in seeds:
                logger.info(f"  Seed {seed}:")
                
                # Stage 1: Overfitting sanity check
                overfitting_result = self.run_overfitting_sanity_check(task, method, seed)
                method_results['overfitting_checks'].append(overfitting_result)
                results['summary']['total_checks'] += 1
                
                if overfitting_result['success']:
                    results['summary']['passed_checks'] += 1
                    logger.info(f"    ✅ Overfitting check passed")
                else:
                    results['summary']['failed_checks'] += 1
                    method_results['method_success'] = False
                    logger.error(f"    ❌ Overfitting check failed")
                    continue  # Skip production check if overfitting fails
                
                # Stage 2: Production stability check
                production_result = self.run_production_stability_check(task, method, seed)
                method_results['production_checks'].append(production_result)
                results['summary']['total_checks'] += 1
                
                if production_result['success']:
                    results['summary']['passed_checks'] += 1
                    logger.info(f"    ✅ Production stability check passed")
                else:
                    results['summary']['failed_checks'] += 1
                    method_results['method_success'] = False
                    logger.error(f"    ❌ Production stability check failed")
                
                # Collect issues
                results['summary']['issues_found'].extend(
                    overfitting_result['issues'] + production_result['issues']
                )
            
            results['method_results'][method] = method_results
            if not method_results['method_success']:
                results['overall_success'] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"SANITY CHECK SUMMARY FOR {task.upper()}")
        logger.info("=" * 60)
        
        if results['overall_success']:
            logger.info("🎉 ALL SANITY CHECKS PASSED")
        else:
            logger.error("❌ SANITY CHECK FAILURES DETECTED")
        
        logger.info(f"Total checks: {results['summary']['total_checks']}")
        logger.info(f"Passed: {results['summary']['passed_checks']}")
        logger.info(f"Failed: {results['summary']['failed_checks']}")
        
        if results['summary']['issues_found']:
            logger.warning("Issues found:")
            for issue in set(results['summary']['issues_found']):  # Deduplicate
                logger.warning(f"  - {issue}")
        
        return results
    
    def run_all_tasks_sanity_checks(self) -> Dict[str, Any]:
        """Run comprehensive sanity checks for all tasks."""
        logger.info("Starting comprehensive sanity checks for all tasks")
        logger.info("=" * 80)
        
        overall_results = {
            'overall_success': True,
            'task_results': {},
            'global_summary': {
                'total_tasks': len(self.tasks),
                'passed_tasks': 0,
                'failed_tasks': 0,
                'total_checks': 0,
                'passed_checks': 0,
                'failed_checks': 0
            }
        }
        
        for task in self.tasks:
            task_results = self.run_comprehensive_sanity_checks(task)
            overall_results['task_results'][task] = task_results
            
            # Update global summary
            if task_results['overall_success']:
                overall_results['global_summary']['passed_tasks'] += 1
            else:
                overall_results['global_summary']['failed_tasks'] += 1
                overall_results['overall_success'] = False
            
            overall_results['global_summary']['total_checks'] += task_results['summary']['total_checks']
            overall_results['global_summary']['passed_checks'] += task_results['summary']['passed_checks']
            overall_results['global_summary']['failed_checks'] += task_results['summary']['failed_checks']
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("GLOBAL SANITY CHECK SUMMARY")
        logger.info("=" * 80)
        
        if overall_results['overall_success']:
            logger.info("🎉 ALL TASKS PASSED SANITY CHECKS - READY FOR PHASE 1")
        else:
            logger.error("❌ SANITY CHECK FAILURES - FIX ISSUES BEFORE PHASE 1")
        
        summary = overall_results['global_summary']
        logger.info(f"Tasks: {summary['passed_tasks']}/{summary['total_tasks']} passed")
        logger.info(f"Individual checks: {summary['passed_checks']}/{summary['total_checks']} passed")
        
        return overall_results

def main():
    """Run the enhanced sanity check framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced sanity check framework")
    parser.add_argument("--task", type=str, choices=['mrpc', 'sst2', 'rte', 'squad_v2'],
                       help="Run checks for specific task only")
    parser.add_argument("--method", type=str, choices=['lora', 'full_finetune'],
                       help="Run checks for specific method only")
    parser.add_argument("--check-type", type=str, choices=['overfitting', 'production', 'both'],
                       default='both', help="Type of sanity check to run")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    framework = SanityCheckFramework()
    
    if args.task:
        # Run checks for specific task
        results = framework.run_comprehensive_sanity_checks(args.task)
        success = results['overall_success']
    else:
        # Run checks for all tasks
        results = framework.run_all_tasks_sanity_checks()
        success = results['overall_success']
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
