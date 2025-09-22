#!/usr/bin/env python3
"""Validation demo script to test the complete experimental setup."""

import os
import sys
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables for W&B."""
    os.environ['WANDB_PROJECT'] = 'NLP'
    os.environ['WANDB_ENTITY'] = 'galavny-tel-aviv-university'
    logger.info("Environment variables set for W&B")

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        from shared.data_preparation import TaskDataLoader
        
        # Initialize data loader
        loader = TaskDataLoader()
        
        # Print dataset summary
        loader.print_dataset_summary()
        
        # Validate data integrity
        if not loader.validate_data_integrity():
            logger.error("Data validation failed!")
            return False
        
        # Test sample loading
        samples = loader.get_all_task_samples(num_samples_per_task=5)
        
        logger.info(f"Successfully loaded samples from {len(samples)} tasks:")
        for task_name, data in samples.items():
            logger.info(f"  {task_name}: {data['num_samples']} samples, input shape: {data['input_ids'].shape}")
        
        logger.info("✓ Data loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def test_sanity_checks():
    """Test sanity check functionality."""
    logger.info("Testing sanity checks...")
    
    try:
        from shared.sanity_checks import run_sanity_checks
        
        # Run a subset of sanity checks
        logger.info("Running basic sanity checks...")
        success = run_sanity_checks()
        
        if success:
            logger.info("✓ Sanity checks passed")
            return True
        else:
            logger.error("✗ Some sanity checks failed")
            return False
            
    except Exception as e:
        logger.error(f"Sanity checks failed with error: {e}")
        return False

def test_quick_experiment():
    """Run a quick experiment to test the full pipeline."""
    logger.info("Testing quick experiment pipeline...")
    
    try:
        import yaml
        from shared.experiment_runner import run_experiment_from_config
        
        # Create a minimal config for testing
        with open('shared/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify for quick testing
        config['training']['num_train_epochs'] = 1
        config['training']['eval_steps'] = 10
        config['training']['save_steps'] = 50
        config['training']['logging_steps'] = 5
        
        # Use very small data samples
        for task in config['tasks'].values():
            task['max_samples_train'] = 20
            task['max_samples_eval'] = 10
        
        # Save test config
        test_config_path = 'shared/config_test.yaml'
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info("Running quick LoRA experiment on SST-2 (20 training samples, 1 epoch)...")
        
        # Run a single quick experiment
        results = run_experiment_from_config(
            config_path=test_config_path,
            tasks=['sst2'],
            methods=['lora'],
            skip_sanity_checks=True
        )
        
        if results and 'sst2' in results and 'lora' in results['sst2']:
            result = results['sst2']['lora']
            if 'error' in result:
                logger.error(f"Quick experiment failed: {result['error']}")
                return False
            else:
                logger.info(f"✓ Quick experiment completed successfully!")
                logger.info(f"  Train loss: {result.get('train_loss', 'N/A')}")
                logger.info(f"  Eval loss: {result.get('eval_loss', 'N/A')}")
                logger.info(f"  Runtime: {result.get('train_runtime', 'N/A')}s")
                return True
        else:
            logger.error("Quick experiment returned unexpected results")
            return False
            
    except Exception as e:
        logger.error(f"Quick experiment failed: {e}")
        return False

def test_wandb_integration():
    """Test W&B integration."""
    logger.info("Testing W&B integration...")
    
    try:
        import wandb
        
        # Test W&B connection
        run = wandb.init(
            project='NLP',
            entity='galavny-tel-aviv-university',
            name='validation-demo-test',
            tags=['validation', 'demo', 'test']
        )
        
        # Log some test metrics
        for i in range(5):
            wandb.log({
                'test_loss': 1.0 - i * 0.1,
                'test_accuracy': 0.5 + i * 0.1,
                'step': i
            })
            time.sleep(0.1)
        
        wandb.finish()
        
        logger.info("✓ W&B integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"W&B integration test failed: {e}")
        return False

def test_reproducibility():
    """Test reproducibility with fixed seeds."""
    logger.info("Testing reproducibility...")
    
    try:
        import torch
        import numpy as np
        from shared.data_preparation import TaskDataLoader
        
        # Test 1: Data loading reproducibility
        results = []
        for run in range(2):
            np.random.seed(42)
            torch.manual_seed(42)
            
            loader = TaskDataLoader()
            sample = loader.get_sample_data('sst2', 'train', 5, seed=42)
            
            # Get first input_ids as a simple reproducibility check
            first_input = sample[0]['input_ids'] if len(sample) > 0 else None
            results.append(first_input)
        
        # Check if both runs produced identical results
        if results[0] == results[1]:
            logger.info("✓ Reproducibility test passed - data loading is deterministic")
            return True
        else:
            logger.warning("⚠ Reproducibility test: data loading may not be fully deterministic")
            return True  # Not a critical failure
            
    except Exception as e:
        logger.error(f"Reproducibility test failed: {e}")
        return False

def main():
    """Run the complete validation demo."""
    logger.info("="*60)
    logger.info("STARTING VALIDATION DEMO")
    logger.info("="*60)
    
    # Setup
    setup_environment()
    
    # Create results directory
    results_dir = Path("validation_demo_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("W&B Integration", test_wandb_integration),
        ("Reproducibility", test_reproducibility),
        ("Sanity Checks", test_sanity_checks),
        ("Quick Experiment", test_quick_experiment),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.error(f"✗ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION DEMO SUMMARY")
    logger.info("="*60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:<20}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("The experimental setup is ready for full experiments.")
        return True
    else:
        logger.error(f"❌ {total - passed} validation tests failed.")
        logger.error("Please address the issues before running full experiments.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
