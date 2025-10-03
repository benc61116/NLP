#!/usr/bin/env python3
"""
Re-evaluate saved full fine-tuning models to extract performance metrics.
This script loads saved models and evaluates them on their validation sets.
"""

import sys
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.full_finetune import FullFinetuneExperiment
from shared.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_full_finetune_models(results_dir: Path) -> List[Dict]:
    """Find all saved full fine-tuning models."""
    models = []
    
    for exp_dir in results_dir.glob("full_finetune_*"):
        for model_dir in exp_dir.glob("full_ft_*_seed*"):
            # Parse task and seed from directory name
            parts = model_dir.name.split("_")
            if len(parts) >= 4:
                task = "_".join(parts[2:-1])  # e.g., "mrpc" or "squad_v2"
                seed_str = parts[-1]  # e.g., "seed42"
                seed = int(seed_str.replace("seed", ""))
                
                final_model = model_dir / "final_model"
                if final_model.exists():
                    models.append({
                        "task": task,
                        "seed": seed,
                        "method": "full_finetune",
                        "model_path": final_model
                    })
    
    return models


def evaluate_model(model_info: Dict, experiment: FullFinetuneExperiment) -> Dict:
    """Evaluate a saved model and return metrics."""
    task = model_info["task"]
    seed = model_info["seed"]
    model_path = model_info["model_path"]
    
    logger.info(f"Evaluating {task} (seed={seed})...")
    logger.info(f"  Model: {model_path}")
    
    try:
        # Load model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = experiment.config['model']['name']
        target_dtype = getattr(torch, experiment.config['model']['dtype'])
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            torch_dtype=target_dtype
        ).to('cuda')
        
        # Load validation data
        from shared.data_loader import DataLoader
        data_loader = DataLoader(config_path="shared/config.yaml")
        
        val_dataset = data_loader.load_and_preprocess_data(
            task, 
            split="validation",
            max_samples=None  # Full validation set
        )
        
        # Evaluate
        from transformers import Trainer, TrainingArguments
        from shared.training_utils import DataCollatorWithPadding
        
        training_args = TrainingArguments(
            output_dir="tmp_eval",
            per_device_eval_batch_size=16,
            dataloader_num_workers=0,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
        )
        
        # Run evaluation
        eval_results = trainer.evaluate(val_dataset)
        
        logger.info(f"  ✓ Evaluated {task} (seed={seed})")
        logger.info(f"    Accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
        logger.info(f"    F1: {eval_results.get('eval_f1', 'N/A')}")
        logger.info(f"    Loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        
        return {
            "task": task,
            "seed": seed,
            "method": "full_finetune",
            "eval_accuracy": eval_results.get("eval_accuracy"),
            "eval_f1": eval_results.get("eval_f1") or eval_results.get("eval_f1_binary"),
            "eval_loss": eval_results.get("eval_loss"),
            "model_path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"  ❌ Failed to evaluate {task} (seed={seed}): {e}")
        return None


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("Re-evaluating Saved Full Fine-Tuning Models")
    logger.info("="*80)
    
    results_dir = Path("results")
    output_dir = Path("results/drift_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all models
    models = find_full_finetune_models(results_dir)
    
    # Filter for classification tasks only
    tasks_to_eval = ["mrpc", "sst2", "rte"]
    models = [m for m in models if m["task"] in tasks_to_eval]
    
    logger.info(f"\nFound {len(models)} models to evaluate:")
    for m in models:
        logger.info(f"  - {m['task']} (seed={m['seed']})")
    
    if not models:
        logger.error("❌ No models found!")
        return
    
    # Initialize experiment
    experiment = FullFinetuneExperiment(config_path="shared/config.yaml")
    
    # Evaluate each model
    all_results = []
    for model_info in models:
        result = evaluate_model(model_info, experiment)
        if result:
            all_results.append(result)
    
    if not all_results:
        logger.error("❌ No results collected!")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_file = output_dir / "full_finetune_performance_metrics.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"\n✓ Saved results to: {csv_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FULL FINE-TUNING PERFORMANCE SUMMARY")
    logger.info("="*80)
    for task in tasks_to_eval:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        
        # Get primary metric
        if task == "mrpc":
            metric_col = "eval_f1"
            metric_name = "F1"
        else:
            metric_col = "eval_accuracy"
            metric_name = "Accuracy"
        
        mean_val = task_df[metric_col].mean()
        std_val = task_df[metric_col].std()
        
        logger.info(f"\n{task.upper()}:")
        logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(task_df)})")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Evaluation complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()


