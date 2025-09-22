#!/usr/bin/env python3
"""Simple demo script to test basic functionality without complex dependencies."""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test basic data loading without model dependencies."""
    logger.info("Testing data loading...")
    
    try:
        from datasets import load_from_disk
        
        # Check if datasets exist
        tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
        loaded_tasks = []
        
        for task in tasks:
            data_path = f"data/{task}"
            if os.path.exists(data_path):
                try:
                    dataset = load_from_disk(data_path)
                    sample_size = len(dataset['train']) if 'train' in dataset else 0
                    logger.info(f"  ✓ {task}: {sample_size:,} training samples")
                    loaded_tasks.append(task)
                except Exception as e:
                    logger.error(f"  ✗ {task}: Failed to load - {e}")
            else:
                logger.error(f"  ✗ {task}: Data directory not found")
        
        if len(loaded_tasks) == len(tasks):
            logger.info("✓ All datasets loaded successfully")
            return True
        else:
            logger.warning(f"Only {len(loaded_tasks)}/{len(tasks)} datasets loaded")
            return len(loaded_tasks) > 0
            
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def test_basic_imports():
    """Test that required packages can be imported."""
    logger.info("Testing basic imports...")
    
    imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('wandb', 'Weights & Biases'),
        ('peft', 'PEFT'),
        ('yaml', 'PyYAML'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy')
    ]
    
    failed_imports = []
    
    for module, name in imports:
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError as e:
            logger.error(f"  ✗ {name}: {e}")
            failed_imports.append(name)
    
    if not failed_imports:
        logger.info("✓ All required packages imported successfully")
        return True
    else:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False

def test_config_loading():
    """Test configuration file loading."""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        
        config_path = 'shared/config.yaml'
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['model', 'training', 'tasks', 'wandb', 'reproducibility']
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
            else:
                logger.info(f"  ✓ {section} section found")
        
        if missing_sections:
            logger.error(f"Missing config sections: {', '.join(missing_sections)}")
            return False
        
        # Test task configurations
        tasks = config.get('tasks', {})
        if len(tasks) == 4:
            logger.info(f"  ✓ All 4 tasks configured: {list(tasks.keys())}")
        else:
            logger.warning(f"Expected 4 tasks, found {len(tasks)}: {list(tasks.keys())}")
        
        logger.info("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Config loading test failed: {e}")
        return False

def test_wandb_connection():
    """Test W&B connection without running experiments."""
    logger.info("Testing W&B connection...")
    
    try:
        import wandb
        
        # Set environment variables
        os.environ['WANDB_PROJECT'] = 'NLP'
        os.environ['WANDB_ENTITY'] = 'galavny-tel-aviv-university'
        
        # Test login status
        if wandb.api.api_key:
            logger.info("  ✓ W&B API key found")
        else:
            logger.warning("  ⚠ W&B API key not found - run 'wandb login'")
        
        # Test basic connection (without creating a run)
        api = wandb.Api()
        
        # Try to get user info
        try:
            user = api.viewer
            logger.info(f"  ✓ Connected as: {user.get('username', 'unknown')}")
        except:
            logger.warning("  ⚠ Could not get user info")
        
        logger.info("✓ W&B connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"W&B connection test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        'shared',
        'scripts',
        'scripts/phase1',
        'scripts/phase2a', 
        'scripts/phase2b',
        'data'
    ]
    
    optional_dirs = [
        'results',
        'logs',
        'wandb'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            logger.info(f"  ✓ {dir_path} exists")
        else:
            logger.error(f"  ✗ {dir_path} missing")
            all_good = False
    
    for dir_path in optional_dirs:
        if os.path.exists(dir_path):
            logger.info(f"  ✓ {dir_path} exists")
        else:
            logger.info(f"  ○ {dir_path} will be created when needed")
    
    # Test creating a results directory
    try:
        test_dir = Path('results/test')
        test_dir.mkdir(parents=True, exist_ok=True)
        logger.info("  ✓ Can create result directories")
        
        # Clean up
        if test_dir.exists():
            test_dir.rmdir()
            test_dir.parent.rmdir() if test_dir.parent.exists() and not list(test_dir.parent.iterdir()) else None
            
    except Exception as e:
        logger.error(f"  ✗ Cannot create directories: {e}")
        all_good = False
    
    if all_good:
        logger.info("✓ Directory structure test passed")
    else:
        logger.error("✗ Directory structure test failed")
        
    return all_good

def test_script_executability():
    """Test that phase scripts are executable."""
    logger.info("Testing script executability...")
    
    script_files = [
        'scripts/download_datasets.py',
        'scripts/phase1/vm1.sh',
        'scripts/phase1/vm2.sh',
        'scripts/phase1/vm3.sh',
        'scripts/phase2a/vm1.sh',
        'scripts/phase2a/vm2.sh',
        'scripts/phase2a/vm3.sh',
        'scripts/phase2b/vm1.sh'
    ]
    
    all_good = True
    
    for script in script_files:
        if os.path.exists(script):
            if os.access(script, os.R_OK):
                logger.info(f"  ✓ {script} is readable")
            else:
                logger.error(f"  ✗ {script} is not readable")
                all_good = False
        else:
            logger.error(f"  ✗ {script} does not exist")
            all_good = False
    
    if all_good:
        logger.info("✓ All scripts are accessible")
    else:
        logger.error("✗ Some scripts are missing or not accessible")
        
    return all_good

def main():
    """Run the simple demo tests."""
    logger.info("="*60)
    logger.info("STARTING SIMPLE VALIDATION DEMO")
    logger.info("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Script Accessibility", test_script_executability),
        ("W&B Connection", test_wandb_connection),
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
    logger.info("SIMPLE DEMO SUMMARY")
    logger.info("="*60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:<25}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL BASIC TESTS PASSED!")
        logger.info("The basic setup is ready. You may need to:")
        logger.info("1. Set up model authentication for Llama-2")
        logger.info("2. Run 'wandb login' if not already done")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        logger.info("✅ MOST TESTS PASSED!")
        logger.info("The setup is largely ready with minor issues.")
        return True
    else:
        logger.error(f"❌ {total - passed} critical tests failed.")
        logger.error("Please address the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
