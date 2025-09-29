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
    """Validate that all components use the same model AND can actually load/work."""
    results = {
        'consistent': True,
        'expected_model': EXPECTED_MODEL,
        'component_models': {},
        'functionality_tests': {},
        'issues': []
    }
    
    try:
        logger.info(f"🔍 Testing model consistency and functionality for {EXPECTED_MODEL}")
        
        # Test 1: Actually load the model and test it works
        logger.info("🧪 Testing model loading and basic functionality...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(EXPECTED_MODEL)
            model = AutoModelForCausalLM.from_pretrained(EXPECTED_MODEL, torch_dtype=torch.float16, device_map="auto")
            
            # Test basic generation
            test_prompt = "What is artificial intelligence?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results['functionality_tests']['model_loading'] = True
            results['functionality_tests']['generation_test'] = len(generated_text) > len(test_prompt)
            results['functionality_tests']['model_size_gb'] = model.get_memory_footprint() / (1024**3)
            
            logger.info(f"✅ Model loads successfully (Memory: {results['functionality_tests']['model_size_gb']:.2f}GB)")
            logger.info(f"✅ Generation test: '{test_prompt}' → '{generated_text[:100]}...'")
            
            # Clean up memory
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            results['consistent'] = False
            results['issues'].append(f"Model loading/generation failed: {e}")
            results['functionality_tests']['model_loading'] = False
            logger.error(f"❌ Model functionality test failed: {e}")
        
        # Test 2: Check baseline experiments can actually initialize
        logger.info("🧪 Testing baseline experiments functionality...")
        try:
            from experiments.baselines import BaselineExperiments
            baseline_exp = BaselineExperiments()
            baseline_model = baseline_exp.model_name
            results['component_models']['baselines'] = baseline_model
            
            if baseline_model != EXPECTED_MODEL:
                results['consistent'] = False
                results['issues'].append(f"Baselines uses {baseline_model}, expected {EXPECTED_MODEL}")
            
            # Test that baselines can actually load small dataset
            try:
                train_data, val_data = baseline_exp.get_dataset_splits("squad_v2")
                results['functionality_tests']['baseline_data_loading'] = val_data['num_samples'] > 0
                logger.info(f"✅ Baseline data loading works (val samples: {val_data['num_samples']})")
            except Exception as e:
                results['functionality_tests']['baseline_data_loading'] = False
                results['issues'].append(f"Baseline data loading failed: {e}")
                
        except Exception as e:
            results['issues'].append(f"Baseline experiments test failed: {e}")
            results['functionality_tests']['baselines'] = False
        
        # Test 3: Check data preparation actually works
        logger.info("🧪 Testing data preparation functionality...")
        try:
            from shared.data_preparation import TaskDataLoader
            data_loader = TaskDataLoader(EXPECTED_MODEL)
            
            # Test loading a small amount of data
            squad_data = data_loader.prepare_qa_data(split="validation", num_samples=10)
            results['functionality_tests']['data_preparation'] = squad_data['num_samples'] > 0
            logger.info(f"✅ Data preparation works (loaded {squad_data['num_samples']} validation samples)")
            
        except Exception as e:
            results['functionality_tests']['data_preparation'] = False
            results['issues'].append(f"Data preparation test failed: {e}")
        
        # Test 4: Check if key scripts exist and use correct model (improved)
        extract_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'extract_base_representations.py')
        if os.path.exists(extract_script_path):
            with open(extract_script_path, 'r') as f:
                content = f.read()
                # More thorough check - look for model assignment patterns
                model_patterns = [
                    f'"{EXPECTED_MODEL}"',
                    f"'{EXPECTED_MODEL}'",
                    f"model_name = {EXPECTED_MODEL}",
                    f"MODEL_NAME = {EXPECTED_MODEL}"
                ]
                found_model = any(pattern in content for pattern in model_patterns)
                
                if found_model:
                    results['component_models']['extract_base_representations'] = EXPECTED_MODEL
                    logger.info("✅ Extract script uses correct model")
                else:
                    results['consistent'] = False
                    results['issues'].append(f"extract_base_representations.py may not use {EXPECTED_MODEL}")
        
        # Test 5: Check production experiment scripts (improved)
        try:
            from experiments.lora_finetune import LoRAExperiment
            from experiments.full_finetune import FullFinetuneExperiment
            
            # Actually test initialization with config files
            logger.info("🧪 Testing LoRA experiment initialization...")
            lora_exp = LoRAExperiment(config_path="shared/config.yaml")
            # Verify it uses the expected model from config
            expected_model_in_config = lora_exp.config['model']['name']
            if expected_model_in_config == EXPECTED_MODEL:
                results['functionality_tests']['lora_init'] = True
                results['component_models']['lora_finetune'] = EXPECTED_MODEL
                logger.info("✅ LoRA experiment initializes correctly")
            else:
                results['functionality_tests']['lora_init'] = False
                results['issues'].append(f"LoRA config uses {expected_model_in_config}, expected {EXPECTED_MODEL}")
            
            logger.info("🧪 Testing Full Fine-tune experiment initialization...")
            full_exp = FullFinetuneExperiment(config_path="shared/config.yaml")
            # Verify it uses the expected model from config  
            expected_model_in_config = full_exp.config['model']['name']
            if expected_model_in_config == EXPECTED_MODEL:
                results['functionality_tests']['full_init'] = True
                results['component_models']['full_finetune'] = EXPECTED_MODEL
                logger.info("✅ Full fine-tune experiment initializes correctly")
            else:
                results['functionality_tests']['full_init'] = False
                results['issues'].append(f"Full FT config uses {expected_model_in_config}, expected {EXPECTED_MODEL}")
            
        except Exception as e:
            results['issues'].append(f"Production script initialization failed: {e}")
            results['functionality_tests']['lora_init'] = False
            results['functionality_tests']['full_init'] = False
        
        # Overall assessment
        failed_tests = [k for k, v in results['functionality_tests'].items() if v is False]
        if failed_tests:
            results['consistent'] = False
            results['issues'].append(f"Functionality tests failed: {failed_tests}")
        
        if results['consistent']:
            logger.info(f"✅ Model consistency AND functionality validation PASSED")
            logger.info(f"   All components use {EXPECTED_MODEL} and work correctly")
        else:
            logger.error(f"❌ Model consistency/functionality validation FAILED")
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
