#!/usr/bin/env python3
"""
Evaluate saved full fine-tuning models to extract REAL performance metrics.
"""

import sys
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_full_finetune_models() -> List[Dict]:
    """Find all saved full fine-tuning classification models."""
    models = []
    results_dir = Path("results")
    
    # Known models from our search
    model_paths = [
        ("results/full_finetune_20251001_185252/full_ft_sst2_seed42/final_model", "sst2", 42),
        ("results/full_finetune_20251002_054948/full_ft_sst2_seed1337/final_model", "sst2", 1337),
        ("results/full_finetune_20251002_165023/full_ft_sst2_seed2024/final_model", "sst2", 2024),
        ("results/full_finetune_20251001_173755/full_ft_mrpc_seed1337/final_model", "mrpc", 1337),
        ("results/full_finetune_20251001_181529/full_ft_mrpc_seed2024/final_model", "mrpc", 2024),
        ("results/full_finetune_20251001_164533/full_ft_mrpc_seed42/final_model", "mrpc", 42),
        ("results/full_finetune_20251003_034934/full_ft_rte_seed42/final_model", "rte", 42),
        ("results/full_finetune_20251003_041630/full_ft_rte_seed1337/final_model", "rte", 1337),
        ("results/full_finetune_20251003_044300/full_ft_rte_seed2024/final_model", "rte", 2024),
    ]
    
    for model_path, task, seed in model_paths:
        path = Path(model_path)
        if path.exists():
            models.append({
                "task": task,
                "seed": seed,
                "method": "full_finetune",
                "model_path": path
            })
    
    return models


def evaluate_model(model_info: Dict) -> Dict:
    """Evaluate a saved model and return REAL metrics."""
    task = model_info["task"]
    seed = model_info["seed"]
    model_path = model_info["model_path"]
    
    logger.info(f"📊 Evaluating {task.upper()} (seed={seed})...")
    logger.info(f"   Model: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16
        ).to('cuda')
        
        logger.info(f"   ✓ Model loaded")
        
        # Load validation data directly from GLUE
        if task == "mrpc":
            dataset = load_dataset("glue", "mrpc", split="validation")
        elif task == "sst2":
            dataset = load_dataset("glue", "sst2", split="validation")
        elif task == "rte":
            dataset = load_dataset("glue", "rte", split="validation")
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Tokenize dataset
        def tokenize_function(examples):
            if task == "mrpc":
                return tokenizer(examples["sentence1"], examples["sentence2"], 
                               truncation=True, padding="max_length", max_length=128)
            elif task == "sst2":
                return tokenizer(examples["sentence"], 
                               truncation=True, padding="max_length", max_length=128)
            elif task == "rte":
                return tokenizer(examples["sentence1"], examples["sentence2"],
                               truncation=True, padding="max_length", max_length=128)
        
        val_dataset = dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.rename_column("label", "labels")
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        logger.info(f"   ✓ Loaded {len(val_dataset)} validation samples")
        
        # Create trainer for evaluation
        training_args = TrainingArguments(
            output_dir="tmp_eval",
            per_device_eval_batch_size=32,
            dataloader_num_workers=0,
        )
        
        # Determine primary metric for this task
        if task == "mrpc":
            primary_metric = "f1"
        else:
            primary_metric = "accuracy"
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: compute_classification_metrics(eval_pred, primary_metric)
        )
        
        # Run evaluation
        eval_results = trainer.evaluate(val_dataset)
        
        # Extract metrics
        accuracy = eval_results.get('eval_accuracy')
        f1 = eval_results.get('eval_f1') or eval_results.get('eval_f1_binary')
        eval_loss = eval_results.get('eval_loss')
        
        logger.info(f"   ✅ {task.upper()} (seed={seed}):")
        logger.info(f"      Accuracy: {accuracy:.4f}" if accuracy else "      Accuracy: N/A")
        logger.info(f"      F1: {f1:.4f}" if f1 else "      F1: N/A")
        logger.info(f"      Loss: {eval_loss:.4f}" if eval_loss else "      Loss: N/A")
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        
        return {
            "task": task,
            "seed": seed,
            "method": "full_finetune",
            "accuracy": accuracy,
            "f1": f1,
            "eval_loss": eval_loss,
            "model_path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("EXTRACTING REAL PERFORMANCE METRICS FROM SAVED MODELS")
    logger.info("="*80)
    
    output_dir = Path("results/drift_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all models
    models = find_full_finetune_models()
    
    logger.info(f"\n✓ Found {len(models)} full fine-tuning models to evaluate")
    for m in sorted(models, key=lambda x: (x['task'], x['seed'])):
        logger.info(f"   • {m['task'].upper()} (seed={m['seed']})")
    
    if not models:
        logger.error("\n❌ No models found!")
        return
    
    # Evaluate each model
    logger.info("\n" + "="*80)
    logger.info("EVALUATING MODELS")
    logger.info("="*80)
    
    all_results = []
    for model_info in sorted(models, key=lambda x: (x['task'], x['seed'])):
        result = evaluate_model(model_info)
        if result:
            all_results.append(result)
        print()  # Blank line between models
    
    if not all_results:
        logger.error("\n❌ No results collected!")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_file = output_dir / "full_finetune_performance_REAL.csv"
    df.to_csv(csv_file, index=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FULL FINE-TUNING PERFORMANCE SUMMARY (REAL DATA)")
    logger.info("="*80)
    
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        
        # Get primary metric
        if task == "mrpc":
            metric_col = "f1"
            metric_name = "F1"
        else:
            metric_col = "accuracy"
            metric_name = "Accuracy"
        
        mean_val = task_df[metric_col].mean()
        std_val = task_df[metric_col].std()
        
        logger.info(f"\n{task.upper()}:")
        logger.info(f"   {metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(task_df)})")
        for idx, row in task_df.iterrows():
            logger.info(f"      seed {row['seed']}: {row[metric_col]:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Saved REAL metrics to: {csv_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

