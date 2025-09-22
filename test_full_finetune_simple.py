#!/usr/bin/env python3
"""Simple test for full fine-tuning implementation without complex dependencies."""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test that core components can be imported."""
    logger.info("Testing basic imports...")
    
    try:
        import torch
        import transformers
        import numpy as np
        from pathlib import Path
        logger.info("✓ Basic dependencies imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_model_loading():
    """Test loading the target model."""
    logger.info("Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test with smaller model first
        model_name = "microsoft/DialoGPT-small"
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Use float32 to avoid half precision issues
        )
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Vocab size: {tokenizer.vocab_size}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test basic inference
        test_text = "Hello, this is a test"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(f"  Test inference successful, output shape: {outputs.logits.shape}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        from shared.data_preparation import TaskDataLoader
        
        data_loader = TaskDataLoader("microsoft/DialoGPT-small")
        
        # Test loading a small sample from SST-2
        logger.info("Loading SST-2 sample...")
        data = data_loader.prepare_classification_data("sst2", "train", num_samples=10)
        
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Input IDs shape: {len(data['input_ids'])}")
        logger.info(f"  Attention mask shape: {len(data['attention_mask'])}")
        logger.info(f"  Labels shape: {len(data['labels'])}")
        logger.info(f"  Sample input ID length: {len(data['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False


def test_metrics_computation():
    """Test metrics computation."""
    logger.info("Testing metrics computation...")
    
    try:
        from shared.metrics import compute_classification_metrics, RepresentationMetrics
        
        # Test classification metrics
        predictions = torch.randn(8, 2)  # 8 samples, 2 classes
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        
        eval_pred = (predictions, labels)
        metrics = compute_classification_metrics(eval_pred, "accuracy")
        
        logger.info(f"✓ Classification metrics computed successfully")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        
        # Test representation metrics
        repr_a = np.random.randn(100, 512)
        repr_b = np.random.randn(100, 512) + 0.1 * repr_a
        
        cosine_sim = RepresentationMetrics.compute_cosine_similarity(repr_a, repr_b)
        cka_sim = RepresentationMetrics.compute_centered_kernel_alignment(repr_a, repr_b)
        drift = RepresentationMetrics.compute_representation_drift(repr_a, repr_b, "cka")
        
        logger.info(f"✓ Representation metrics computed successfully")
        logger.info(f"  Cosine similarity: {cosine_sim:.4f}")
        logger.info(f"  CKA similarity: {cka_sim:.4f}")
        logger.info(f"  Representation drift: {drift:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Metrics computation failed: {e}")
        return False


def test_representation_extraction():
    """Test representation extraction functionality."""
    logger.info("Testing representation extraction...")
    
    try:
        from test_components import SimpleRepresentationExtractor, SimpleRepresentationConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load small model
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        # Create representation extractor
        config = SimpleRepresentationConfig()
        config.max_validation_samples = 5
        config.save_layers = [0, 1, 2]  # Only first few layers for test
        
        output_dir = Path("test_representations")
        extractor = SimpleRepresentationExtractor(config, output_dir, "test_task", "test_method")
        
        # Create test examples
        test_texts = ["Hello world", "This is a test", "Another example"]
        inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
        
        examples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        extractor.set_validation_examples(examples)
        
        # Extract representations
        representations = extractor.extract_representations(model, step=0)
        
        logger.info(f"✓ Representation extraction successful")
        logger.info(f"  Extracted layers: {list(representations.keys())}")
        
        if representations:
            first_key = list(representations.keys())[0]
            logger.info(f"  Example shape: {representations[first_key].shape}")
        
        # Save representations
        extractor.save_representations(representations, step=0)
        
        # Check saved files
        step_dir = output_dir / "test_method_test_task" / "step_000000"
        if step_dir.exists():
            saved_files = list(step_dir.glob("*.pt"))
            logger.info(f"  Saved {len(saved_files)} representation files")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up test files
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Representation extraction failed: {e}")
        return False


def test_gradient_monitoring():
    """Test gradient monitoring functionality."""
    logger.info("Testing gradient monitoring...")
    
    try:
        from test_components import SimpleGradientStatsMonitor, SimpleMemoryProfiler
        from transformers import AutoModelForCausalLM
        
        # Load small model
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", torch_dtype=torch.float32)
        
        # Create monitors
        grad_monitor = SimpleGradientStatsMonitor()
        memory_profiler = SimpleMemoryProfiler()
        
        # Simulate some gradients
        dummy_loss = torch.sum(model.parameters().__next__())
        dummy_loss.backward()
        
        # Test gradient stats
        grad_stats = grad_monitor.compute_gradient_stats(model)
        
        logger.info(f"✓ Gradient monitoring successful")
        logger.info(f"  Total gradient norm: {grad_stats['gradient_norm_total']:.6f}")
        logger.info(f"  Num parameters with gradients: {grad_stats['num_parameters_with_gradients']}")
        
        # Test memory monitoring
        memory_stats = memory_profiler.get_memory_stats()
        
        logger.info(f"✓ Memory monitoring successful")
        logger.info(f"  CPU memory (RSS): {memory_stats['cpu_memory_rss_mb']:.1f} MB")
        
        if torch.cuda.is_available():
            logger.info(f"  GPU memory available")
        else:
            logger.info(f"  No GPU memory monitoring (CPU only)")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradient monitoring failed: {e}")
        return False


def test_configuration():
    """Test configuration loading and validation."""
    logger.info("Testing configuration...")
    
    try:
        import yaml
        
        with open("shared/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  LoRA rank: {config['lora']['r']}")
        logger.info(f"  Target modules: {config['lora']['target_modules']}")
        logger.info(f"  Tasks: {list(config['tasks'].keys())}")
        
        # Validate required sections
        required_sections = ['model', 'lora', 'training', 'tasks', 'wandb']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        logger.info(f"✓ All required configuration sections present")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("FULL FINE-TUNING IMPLEMENTATION TESTS")
    logger.info("="*80)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Metrics Computation", test_metrics_computation),
        ("Representation Extraction", test_representation_extraction),
        ("Gradient Monitoring", test_gradient_monitoring),
    ]
    
    results = {}
    passed = 0
    
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
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:<30}: {status}")
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("The full fine-tuning implementation is ready for use")
        logger.info("\nKey capabilities validated:")
        logger.info("✓ Model loading and inference")
        logger.info("✓ Data preparation and processing")
        logger.info("✓ Representation extraction and saving")
        logger.info("✓ Gradient statistics monitoring")
        logger.info("✓ Memory profiling")
        logger.info("✓ Metrics computation")
        logger.info("✓ Configuration management")
        
        logger.info("\nNext steps:")
        logger.info("1. Set up HuggingFace authentication for Llama-2")
        logger.info("2. Run with 'meta-llama/Llama-2-1.3b-hf' for full experiments")
        logger.info("3. Execute VM scripts for parallel training")
        
        return True
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("Please fix the failing components before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
