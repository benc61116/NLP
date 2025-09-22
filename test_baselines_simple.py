#!/usr/bin/env python3
"""Simple test script for baseline experiments without W&B."""

import os
import sys
sys.path.append('/home/galavny13/workspace/NLP')

# Disable W&B for testing
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
from experiments.baselines import BaselineExperiments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_majority_class_baseline():
    """Test majority class baseline."""
    logger.info("Testing majority class baseline...")
    
    experiments = BaselineExperiments()
    
    # Test with simulated data
    task_name = 'mrpc'
    
    try:
        result = experiments.majority_class_baseline(task_name)
        logger.info(f"✓ Majority class baseline test passed")
        logger.info(f"  Result: {result['metrics']['primary_metric']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ Majority class baseline test failed: {e}")
        return False

def test_random_baseline():
    """Test random baseline."""
    logger.info("Testing random baseline...")
    
    experiments = BaselineExperiments()
    
    # Test with simulated data
    task_name = 'sst2'
    
    try:
        result = experiments.random_baseline(task_name, num_seeds=2)
        logger.info(f"✓ Random baseline test passed")
        logger.info(f"  Result: {result['aggregated_metrics']['primary_metric_mean']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ Random baseline test failed: {e}")
        return False

def test_zero_shot_baseline():
    """Test zero-shot baseline."""
    logger.info("Testing zero-shot baseline...")
    
    experiments = BaselineExperiments()
    
    # Test with simulated data
    task_name = 'sst2'
    
    try:
        result = experiments.zero_shot_llama_baseline(task_name, num_prompt_templates=1)
        logger.info(f"✓ Zero-shot baseline test passed")
        logger.info(f"  Result: {result['metrics']['primary_metric']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ Zero-shot baseline test failed: {e}")
        return False

def test_sota_baseline():
    """Test SOTA baseline."""
    logger.info("Testing SOTA baseline...")
    
    experiments = BaselineExperiments()
    
    # Test with simulated data
    task_name = 'mrpc'
    
    try:
        result = experiments.sota_baseline(task_name)
        logger.info(f"✓ SOTA baseline test passed")
        logger.info(f"  Result: {result['metrics']['primary_metric']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ SOTA baseline test failed: {e}")
        return False

def test_metrics_system():
    """Test metrics system."""
    logger.info("Testing metrics system...")
    
    from shared.metrics import MetricsCalculator
    
    calculator = MetricsCalculator()
    
    # Test classification metrics
    predictions = [0, 1, 0, 1, 1]
    true_labels = [0, 1, 1, 1, 0]
    
    try:
        result = calculator.calculate_comprehensive_metrics(
            predictions, true_labels, 'test_task', 'test_baseline'
        )
        logger.info(f"✓ Metrics system test passed")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.3f}")
        return True
    except Exception as e:
        logger.error(f"✗ Metrics system test failed: {e}")
        return False

def run_validation_demo():
    """Run comprehensive validation demo."""
    logger.info("="*60)
    logger.info("BASELINE VALIDATION DEMO")
    logger.info("="*60)
    
    tests = [
        ("Metrics System", test_metrics_system),
        ("Majority Class Baseline", test_majority_class_baseline),
        ("Random Baseline", test_random_baseline),
        ("Zero-Shot Baseline", test_zero_shot_baseline),
        ("SOTA Baseline", test_sota_baseline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)
        
        if test_func():
            passed += 1
        
    logger.info("\n" + "="*60)
    logger.info(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ ALL TESTS PASSED - Baseline implementation is working!")
    else:
        logger.info(f"✗ {total - passed} tests failed - Check implementation")
    
    logger.info("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_validation_demo()
    exit(0 if success else 1)
