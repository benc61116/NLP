#!/usr/bin/env python3
"""
Validation Script: Multi-Adapter Correctness Check

This script validates that multi-adapter deployment produces identical predictions
to single-adapter deployment, addressing the concern that adapter swapping might
affect output correctness.

Validation Method:
1. Load single LoRA adapter + base model → get predictions
2. Load multi-adapter setup (all adapters) → activate same adapter → get predictions  
3. Compare: predictions should be bitwise identical (or accuracy should match)

This validates the fundamental assumption that adapter swapping is functionally
equivalent to single-adapter deployment.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiAdapterValidator:
    """Validates that multi-adapter deployment produces correct outputs."""
    
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        models_dir: str = "results/downloaded_models",
        device: str = None
    ):
        self.base_model_name = base_model_name
        self.models_dir = Path(models_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_test_samples(self, task: str, num_samples: int = 100) -> List[Tuple[str, int]]:
        """Get test samples for a task."""
        if task == "mrpc":
            dataset = load_dataset("glue", "mrpc", split="test")
            samples = [(f"{ex['sentence1']} {ex['sentence2']}", ex['label']) 
                      for ex in dataset.select(range(min(num_samples, len(dataset))))]
        elif task == "sst2":
            dataset = load_dataset("glue", "sst2", split="validation")
            samples = [(ex['sentence'], ex['label']) 
                      for ex in dataset.select(range(min(num_samples, len(dataset))))]
        elif task == "rte":
            dataset = load_dataset("glue", "rte", split="test")
            samples = [(f"{ex['sentence1']} {ex['sentence2']}", ex['label']) 
                      for ex in dataset.select(range(min(num_samples, len(dataset))))]
        else:
            raise ValueError(f"Unknown task: {task}")
            
        return samples
        
    def get_predictions_single_adapter(
        self,
        task: str,
        seed: int,
        samples: List[Tuple[str, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions using single-adapter deployment."""
        logger.info(f"Single-adapter: Loading {task}_seed{seed}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Load LoRA adapter
        adapter_path = self.models_dir / f"lora_adapter_{task}_seed{seed}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
            
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        # Get predictions
        all_logits = []
        all_preds = []
        
        with torch.no_grad():
            for text, _ in samples:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                logits = outputs.logits.float().cpu().numpy()[0]  # Convert bfloat16 to float32
                pred = np.argmax(logits)
                
                all_logits.append(logits)
                all_preds.append(pred)
                
        # Cleanup
        del model, base_model
        torch.cuda.empty_cache()
        
        return np.array(all_logits), np.array(all_preds)
        
    def get_predictions_multi_adapter(
        self,
        tasks: List[str],
        seeds: List[int],
        active_task: str,
        active_seed: int,
        samples: List[Tuple[str, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions using multi-adapter deployment with adapter switching."""
        logger.info(f"Multi-adapter: Loading {len(tasks)} adapters, activating {active_task}_seed{active_seed}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Load first adapter
        first_adapter = self.models_dir / f"lora_adapter_{tasks[0]}_seed{seeds[0]}"
        model = PeftModel.from_pretrained(base_model, str(first_adapter), adapter_name=f"{tasks[0]}_seed{seeds[0]}")
        
        # Load remaining adapters
        for task, seed in zip(tasks[1:], seeds[1:]):
            adapter_path = self.models_dir / f"lora_adapter_{task}_seed{seed}"
            if not adapter_path.exists():
                logger.warning(f"Adapter not found: {adapter_path}, skipping")
                continue
            model.load_adapter(str(adapter_path), adapter_name=f"{task}_seed{seed}")
            
        # Activate the target adapter
        active_adapter_name = f"{active_task}_seed{active_seed}"
        model.set_adapter(active_adapter_name)
        model.eval()
        
        logger.info(f"Active adapter: {model.active_adapter}")
        
        # Get predictions
        all_logits = []
        all_preds = []
        
        with torch.no_grad():
            for text, _ in samples:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                logits = outputs.logits.float().cpu().numpy()[0]  # Convert bfloat16 to float32
                pred = np.argmax(logits)
                
                all_logits.append(logits)
                all_preds.append(pred)
                
        # Cleanup
        del model, base_model
        torch.cuda.empty_cache()
        
        return np.array(all_logits), np.array(all_preds)
        
    def validate_task(
        self,
        task: str,
        seed: int,
        multi_adapter_tasks: List[str],
        multi_adapter_seeds: List[int],
        num_samples: int = 100
    ) -> Dict:
        """
        Validate that single-adapter and multi-adapter produce identical results.
        
        Args:
            task: Task to test (e.g., "mrpc")
            seed: Seed of the adapter to test
            multi_adapter_tasks: List of all tasks in multi-adapter setup
            multi_adapter_seeds: List of all seeds in multi-adapter setup
            num_samples: Number of test samples
            
        Returns:
            Dictionary with validation results
        """
        logger.info("="*80)
        logger.info(f"VALIDATING: {task}_seed{seed}")
        logger.info("="*80)
        
        # Get test samples
        samples = self.get_test_samples(task, num_samples)
        logger.info(f"Testing on {len(samples)} samples")
        
        # Get single-adapter predictions
        logger.info("\n1. Single-adapter predictions...")
        single_logits, single_preds = self.get_predictions_single_adapter(task, seed, samples)
        
        # Get multi-adapter predictions
        logger.info("\n2. Multi-adapter predictions...")
        multi_logits, multi_preds = self.get_predictions_multi_adapter(
            multi_adapter_tasks, multi_adapter_seeds, task, seed, samples
        )
        
        # Compare results
        logger.info("\n3. Comparing results...")
        
        # Check if predictions are identical
        preds_identical = np.array_equal(single_preds, multi_preds)
        pred_mismatch_count = np.sum(single_preds != multi_preds)
        pred_mismatch_pct = (pred_mismatch_count / len(samples)) * 100
        
        # Check if logits are identical (or very close)
        logits_close = np.allclose(single_logits, multi_logits, rtol=1e-5, atol=1e-8)
        max_logit_diff = np.max(np.abs(single_logits - multi_logits))
        mean_logit_diff = np.mean(np.abs(single_logits - multi_logits))
        
        # Calculate accuracy if we have labels
        true_labels = np.array([label for _, label in samples])
        single_accuracy = np.mean(single_preds == true_labels) * 100
        multi_accuracy = np.mean(multi_preds == true_labels) * 100
        accuracy_diff = abs(single_accuracy - multi_accuracy)
        
        results = {
            "task": task,
            "seed": seed,
            "num_samples": len(samples),
            "num_adapters_loaded": len(multi_adapter_tasks),
            "predictions_identical": preds_identical,
            "prediction_mismatches": int(pred_mismatch_count),
            "prediction_mismatch_pct": float(pred_mismatch_pct),
            "logits_close": logits_close,
            "max_logit_difference": float(max_logit_diff),
            "mean_logit_difference": float(mean_logit_diff),
            "single_adapter_accuracy": float(single_accuracy),
            "multi_adapter_accuracy": float(multi_accuracy),
            "accuracy_difference": float(accuracy_diff),
        }
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Task: {task}_seed{seed}")
        logger.info(f"Samples tested: {len(samples)}")
        logger.info(f"Adapters loaded in multi-adapter: {len(multi_adapter_tasks)}")
        logger.info("")
        logger.info(f"✓ Predictions identical: {preds_identical}")
        if not preds_identical:
            logger.warning(f"  ⚠️  Mismatches: {pred_mismatch_count}/{len(samples)} ({pred_mismatch_pct:.2f}%)")
        logger.info(f"✓ Logits close (rtol=1e-5): {logits_close}")
        logger.info(f"  Max logit difference: {max_logit_diff:.6f}")
        logger.info(f"  Mean logit difference: {mean_logit_diff:.6f}")
        logger.info("")
        logger.info(f"Single-adapter accuracy: {single_accuracy:.2f}%")
        logger.info(f"Multi-adapter accuracy: {multi_accuracy:.2f}%")
        logger.info(f"Accuracy difference: {accuracy_diff:.4f}%")
        
        if preds_identical and logits_close:
            logger.info("\n✅ VALIDATION PASSED: Multi-adapter produces identical results!")
        elif preds_identical:
            logger.info("\n✅ VALIDATION PASSED: Predictions identical (logits have minor numerical differences)")
        else:
            logger.error("\n❌ VALIDATION FAILED: Predictions differ between single and multi-adapter!")
            
        return results


def main():
    """Run validation experiment."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate multi-adapter correctness")
    parser.add_argument("--models-dir", type=str, default="results/downloaded_models")
    parser.add_argument("--output-dir", type=str, default="deployment")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    # Create validator
    validator = MultiAdapterValidator(models_dir=args.models_dir)
    
    # Test configuration: 3 adapters loaded, test each one
    tasks = ["mrpc", "sst2", "rte"]
    seeds = [42, 42, 42]  # Use seed 42 for all
    
    all_results = []
    
    # Validate each task
    for task, seed in zip(tasks, seeds):
        try:
            results = validator.validate_task(
                task=task,
                seed=seed,
                multi_adapter_tasks=tasks,
                multi_adapter_seeds=seeds,
                num_samples=args.num_samples
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to validate {task}_seed{seed}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "multi_adapter_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"\n✓ Results saved to {output_file}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    all_passed = all(r["predictions_identical"] for r in all_results)
    
    for r in all_results:
        status = "✅ PASS" if r["predictions_identical"] else "❌ FAIL"
        logger.info(f"{status} - {r['task']}_seed{r['seed']}: "
                   f"Acc diff={r['accuracy_difference']:.4f}%, "
                   f"Max logit diff={r['max_logit_difference']:.6f}")
                   
    if all_passed:
        logger.info("\n🎉 ALL VALIDATIONS PASSED: Multi-adapter is functionally equivalent!")
    else:
        logger.error("\n⚠️  SOME VALIDATIONS FAILED: Multi-adapter may not be equivalent!")


if __name__ == "__main__":
    main()

