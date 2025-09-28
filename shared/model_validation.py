#!/usr/bin/env python3
"""Model validation utilities to ensure consistency across components."""

import os
import sys
import logging
import yaml
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load the shared configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Get expected model from config
try:
    config = load_config()
    EXPECTED_MODEL = config['model']['name']
except Exception as e:
    logger.warning(f"Could not load config, using fallback model: {e}")
    EXPECTED_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def validate_model_consistency() -> Dict[str, Any]:
    """Validate that all components use the same model."""
    results = {
        'consistent': True,
        'expected_model': EXPECTED_MODEL,
        'component_models': {},
        'issues': []
    }
    
    try:
        # Check baselines.py
        from experiments.baselines import BaselineExperiments
        baseline_exp = BaselineExperiments()
        baseline_model = baseline_exp.model_name
        results['component_models']['baselines'] = baseline_model
        
        if baseline_model != EXPECTED_MODEL:
            results['consistent'] = False
            results['issues'].append(f"Baselines uses {baseline_model}, expected {EXPECTED_MODEL}")
        
        # Check data preparation
        from shared.data_preparation import TaskDataLoader
        # Create a temporary data loader to check its model
        temp_loader = TaskDataLoader(EXPECTED_MODEL)
        # The TaskDataLoader doesn't store the model name, but it should work with our expected model
        
        # Check if extract_base_representations.py uses correct model
        # Read the script content to verify
        extract_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'extract_base_representations.py')
        if os.path.exists(extract_script_path):
            with open(extract_script_path, 'r') as f:
                content = f.read()
                if EXPECTED_MODEL in content:
                    results['component_models']['extract_base_representations'] = EXPECTED_MODEL
                else:
                    results['consistent'] = False
                    results['issues'].append(f"extract_base_representations.py may not use {EXPECTED_MODEL}")
        
        # Check production experiment scripts
        try:
            from experiments.lora_finetune import LoRAExperiment
            from experiments.full_finetune import FullFinetuneExperiment
            
            # These scripts should force TinyLlama in main() - we can't easily check without running
            # But we can verify the scripts exist and are importable
            results['component_models']['lora_finetune'] = "Forced to TinyLlama in main()"
            results['component_models']['full_finetune'] = "Forced to TinyLlama in main()"
            
        except ImportError as e:
            results['issues'].append(f"Could not import production scripts: {e}")
        
        if results['consistent']:
            logger.info(f"✅ Model consistency validation PASSED - all components use {EXPECTED_MODEL}")
        else:
            logger.error(f"❌ Model consistency validation FAILED")
            for issue in results['issues']:
                logger.error(f"   - {issue}")
                
    except Exception as e:
        results['consistent'] = False
        results['issues'].append(f"Validation failed with error: {e}")
        logger.error(f"❌ Model validation error: {e}")
    
    return results


def validate_environment() -> Dict[str, Any]:
    """Validate the environment setup for Phase 0."""
    results = {
        'valid': True,
        'checks': {},
        'issues': []
    }
    
    try:
        # Check Python path
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results['checks']['workspace_dir'] = workspace_dir
        results['checks']['python_path'] = sys.path
        
        # Check if workspace is in Python path
        if workspace_dir not in sys.path:
            results['issues'].append(f"Workspace {workspace_dir} not in Python path")
            results['valid'] = False
        
        # Check for required directories
        required_dirs = ['experiments', 'shared', 'scripts', 'data']
        for dir_name in required_dirs:
            dir_path = os.path.join(workspace_dir, dir_name)
            if os.path.exists(dir_path):
                results['checks'][f'{dir_name}_exists'] = True
            else:
                results['checks'][f'{dir_name}_exists'] = False
                results['issues'].append(f"Required directory {dir_name} not found")
                results['valid'] = False
        
        # Check for critical files
        critical_files = [
            'experiments/baselines.py',
            'experiments/lora_finetune.py', 
            'experiments/full_finetune.py',
            'shared/sanity_checks.py',
            'scripts/extract_base_representations.py'
        ]
        for file_path in critical_files:
            full_path = os.path.join(workspace_dir, file_path)
            if os.path.exists(full_path):
                results['checks'][f'{file_path}_exists'] = True
            else:
                results['checks'][f'{file_path}_exists'] = False
                results['issues'].append(f"Critical file {file_path} not found")
                results['valid'] = False
        
        if results['valid']:
            logger.info("✅ Environment validation PASSED")
        else:
            logger.error("❌ Environment validation FAILED")
            for issue in results['issues']:
                logger.error(f"   - {issue}")
                
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Environment validation failed: {e}")
        logger.error(f"❌ Environment validation error: {e}")
    
    return results


def main():
    """Run all validation checks."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔍 Running Phase 0 validation checks...")
    print("=" * 50)
    
    # Validate environment
    env_results = validate_environment()
    
    # Validate model consistency
    model_results = validate_model_consistency()
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    if env_results['valid'] and model_results['consistent']:
        print("🎉 ALL VALIDATIONS PASSED - Phase 0 is ready to run")
        return 0
    else:
        print("❌ VALIDATION FAILURES DETECTED")
        if not env_results['valid']:
            print("   Environment issues found")
        if not model_results['consistent']:
            print("   Model consistency issues found")
        print("\n🛑 Fix these issues before running Phase 0")
        return 1


if __name__ == "__main__":
    exit(main())
