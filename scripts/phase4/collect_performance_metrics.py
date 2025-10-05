#!/usr/bin/env python3
"""
Collect final performance metrics from WandB for Phase 2/3 experiments.
This script aggregates accuracy, F1, and other metrics for all tasks, methods, and seeds.
For full_finetune models, evaluates saved models since metrics aren't in WandB.
"""

import wandb
import pandas as pd
import json
import sys
import torch
from pathlib import Path
import logging
from typing import Dict, List
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_fullfinetune_models(
    tasks: List[str] = ["mrpc", "sst2", "rte"],
    seeds: List[int] = [42, 1337, 2024]
) -> List[Dict]:
    """Evaluate saved full_finetune models to extract REAL performance metrics."""
    logger.info("\n" + "="*80)
    logger.info("Evaluating Full Fine-Tune Models from Disk")
    logger.info("="*80)
    
    models_dir = Path("results/downloaded_models")
    all_results = []
    
    for task in tasks:
        for seed in seeds:
            model_name = f"full_finetune_model_{task}_seed{seed}"
            model_path = models_dir / model_name
            
            if not model_path.exists():
                logger.warning(f"   ❌ Model not found: {model_name}")
                continue
            
            try:
                logger.info(f"\n📊 Evaluating: {model_name}")
                
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Load validation dataset
                if task == "mrpc":
                    dataset = load_dataset("glue", "mrpc")
                    validation_key = "validation"
                elif task == "sst2":
                    dataset = load_dataset("glue", "sst2")
                    validation_key = "validation"
                elif task == "rte":
                    dataset = load_dataset("glue", "rte")
                    validation_key = "validation"
                
                val_dataset = dataset[validation_key]
                
                # Tokenize
                def tokenize_function(examples):
                    if task in ["mrpc", "rte"]:
                        return tokenizer(
                            examples["sentence1"],
                            examples["sentence2"],
                            truncation=True,
                            padding="max_length",
                            max_length=128
                        )
                    else:  # sst2
                        return tokenizer(
                            examples["sentence"],
                            truncation=True,
                            padding="max_length",
                            max_length=128
                        )
                
                val_dataset = val_dataset.map(tokenize_function, batched=True)
                val_dataset = val_dataset.rename_column("label", "labels")
                val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                
                # Determine primary metric
                primary_metric = "f1" if task == "mrpc" else "accuracy"
                
                # Create trainer for evaluation
                training_args = TrainingArguments(
                    output_dir="tmp_eval",
                    per_device_eval_batch_size=32,
                    dataloader_num_workers=0,
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    compute_metrics=lambda eval_pred: compute_classification_metrics(eval_pred, primary_metric)
                )
                
                # Run evaluation
                eval_results = trainer.evaluate(val_dataset)
                
                # Extract metrics
                eval_accuracy = eval_results.get('eval_accuracy')
                eval_f1 = eval_results.get('eval_f1') or eval_results.get('eval_f1_binary')
                eval_loss = eval_results.get('eval_loss')
                
                result = {
                    "task": task,
                    "method": "full_finetune",
                    "seed": seed,
                    "eval_accuracy": eval_accuracy,
                    "eval_f1": eval_f1,
                    "eval_loss": eval_loss,
                    "primary_metric": eval_f1 if task == "mrpc" else eval_accuracy,
                    "primary_metric_name": "f1" if task == "mrpc" else "accuracy",
                    "train_loss": None,
                    "train_runtime": None,
                    "run_id": None,
                    "run_name": model_name,
                    "project": "local_evaluation"
                }
                
                all_results.append(result)
                logger.info(f"   ✓ {task} | full_finetune | seed={seed} | {result['primary_metric_name']}={result['primary_metric']:.4f}")
                
                # Cleanup
                del model, trainer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"   ❌ Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.info(f"\n✓ Evaluated {len(all_results)} full_finetune models")
    return all_results


def collect_phase2_phase3_metrics(
    entity: str = "galavny-tel-aviv-university",
    projects: List[str] = ["NLP-Phase2", "NLP-Phase2-LoRA-Rerun", "NLP-Phase3-Representations"],
    tasks: List[str] = ["mrpc", "sst2", "rte"],
    seeds: List[int] = [42, 1337, 2024]
) -> Dict:
    """Collect performance metrics from WandB for all experiments."""
    
    logger.info("="*80)
    logger.info("Collecting Performance Metrics from WandB")
    logger.info("="*80)
    
    api = wandb.Api()
    all_results = []
    
    for project in projects:
        logger.info(f"\n📊 Querying project: {project}")
        try:
            runs = api.runs(f"{entity}/{project}")
            logger.info(f"   Found {len(runs)} runs")
            
            for run in runs:
                # Only include completed runs with valid task/method config
                if run.state != "finished":
                    continue
                
                config = run.config
                task_name = config.get("task_name")
                method = config.get("method")
                seed = config.get("seed")
                
                # Filter for our tasks
                if task_name not in tasks:
                    continue
                
                # Accept both manual experiments and missing experiment_type (older runs)
                experiment_type = config.get("experiment_type", "manual")
                # Skip only if explicitly not manual (allow None/missing)
                if experiment_type and experiment_type != "manual":
                    continue
                
                # Extract metrics from summary
                summary = run.summary
                
                # Get eval metrics (try multiple naming conventions)
                # Note: WandB uses 'eval/' prefix (with slash) not 'eval_' (with underscore)
                eval_accuracy = summary.get("eval/accuracy") or summary.get("eval_accuracy") or summary.get("final_eval_accuracy")
                eval_f1 = summary.get("eval/f1") or summary.get("eval/f1_binary") or summary.get("eval_f1") or summary.get("eval_f1_binary") or summary.get("final_eval_f1")
                eval_loss = summary.get("eval/loss") or summary.get("eval_loss") or summary.get("final_eval_loss")
                
                # For MRPC, F1 is the primary metric
                if task_name == "mrpc":
                    primary_metric = eval_f1
                    primary_metric_name = "f1"
                else:  # SST-2, RTE
                    primary_metric = eval_accuracy
                    primary_metric_name = "accuracy"
                
                result = {
                    "task": task_name,
                    "method": method,
                    "seed": seed,
                    "eval_accuracy": eval_accuracy,
                    "eval_f1": eval_f1,
                    "eval_loss": eval_loss,
                    "primary_metric": primary_metric,
                    "primary_metric_name": primary_metric_name,
                    "train_loss": summary.get("train_loss") or summary.get("final_train_loss"),
                    "train_runtime": summary.get("train_runtime") or summary.get("training_time_seconds"),
                    "run_id": run.id,
                    "run_name": run.name,
                    "project": project
                }
                
                # Only add if we have at least one eval metric
                if eval_accuracy is not None or eval_f1 is not None:
                    all_results.append(result)
                    metric_str = f"{primary_metric:.4f}" if primary_metric is not None else "N/A"
                    logger.info(f"   ✓ {task_name} | {method} | seed={seed} | {primary_metric_name}={metric_str}")
        
        except Exception as e:
            logger.error(f"   ❌ Error querying {project}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        logger.warning("⚠️  No metrics collected!")
        return {"dataframe": df, "summary": {}}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Collected {len(df)} experiment results")
    logger.info(f"{'='*80}")
    
    # Compute summary statistics by task/method
    summary_stats = {}
    for task in tasks:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        
        summary_stats[task] = {}
        for method in ["full_finetune", "lora"]:
            method_df = task_df[task_df["method"] == method]
            if method_df.empty:
                continue
            
            # Get primary metric for this task
            primary_metric_name = method_df.iloc[0]["primary_metric_name"]
            primary_values = method_df["primary_metric"].dropna()
            
            summary_stats[task][method] = {
                "primary_metric_name": primary_metric_name,
                "primary_metric_mean": float(primary_values.mean()) if len(primary_values) > 0 else None,
                "primary_metric_std": float(primary_values.std()) if len(primary_values) > 0 else None,
                "accuracy_mean": float(method_df["eval_accuracy"].mean()) if method_df["eval_accuracy"].notna().any() else None,
                "accuracy_std": float(method_df["eval_accuracy"].std()) if method_df["eval_accuracy"].notna().any() else None,
                "f1_mean": float(method_df["eval_f1"].mean()) if method_df["eval_f1"].notna().any() else None,
                "f1_std": float(method_df["eval_f1"].std()) if method_df["eval_f1"].notna().any() else None,
                "eval_loss_mean": float(method_df["eval_loss"].mean()) if method_df["eval_loss"].notna().any() else None,
                "eval_loss_std": float(method_df["eval_loss"].std()) if method_df["eval_loss"].notna().any() else None,
                "num_seeds": len(method_df)
            }
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*80)
    for task, methods in summary_stats.items():
        logger.info(f"\n{task.upper()}:")
        for method, stats in methods.items():
            metric_name = stats["primary_metric_name"]
            metric_mean = stats["primary_metric_mean"]
            metric_std = stats["primary_metric_std"]
            logger.info(f"  {method:15s} | {metric_name}: {metric_mean:.4f} ± {metric_std:.4f} | n={stats['num_seeds']}")
    
    logger.info("="*80)
    
    return {
        "dataframe": df,
        "summary_stats": summary_stats,
        "tasks": tasks,
        "seeds": seeds
    }


def main():
    """Main function - collects LoRA metrics from WandB and evaluates full_finetune models."""
    output_dir = Path("drift_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect LoRA metrics from WandB
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Collecting LoRA Metrics from WandB")
    logger.info("="*80)
    results = collect_phase2_phase3_metrics()
    lora_df = results["dataframe"]
    
    # Step 2: Evaluate full_finetune models from disk
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Evaluating Full Fine-Tune Models from Disk")
    logger.info("="*80)
    fullft_results = evaluate_fullfinetune_models()
    fullft_df = pd.DataFrame(fullft_results)
    
    # Step 3: Combine results
    if lora_df.empty and fullft_df.empty:
        logger.error("❌ No results collected from either WandB or local models!")
        return
    
    combined_df = pd.concat([lora_df, fullft_df], ignore_index=True)
    
    logger.info("\n" + "="*80)
    logger.info("COMBINED RESULTS")
    logger.info("="*80)
    logger.info(f"  LoRA runs: {len(lora_df)}")
    logger.info(f"  Full FT runs: {len(fullft_df)}")
    logger.info(f"  Total runs: {len(combined_df)}")
    
    # Save combined CSV
    csv_file = output_dir / "performance_metrics.csv"
    combined_df.to_csv(csv_file, index=False)
    logger.info(f"\n✓ Saved combined metrics to: {csv_file}")
    
    # Compute summary stats for combined data
    tasks = ["mrpc", "sst2", "rte"]
    summary_stats = {}
    for task in tasks:
        task_df = combined_df[combined_df["task"] == task]
        if task_df.empty:
            continue
        
        summary_stats[task] = {}
        for method in ["full_finetune", "lora"]:
            method_df = task_df[task_df["method"] == method]
            if method_df.empty:
                continue
            
            primary_metric_name = method_df.iloc[0]["primary_metric_name"]
            primary_values = method_df["primary_metric"].dropna()
            
            summary_stats[task][method] = {
                "primary_metric_name": primary_metric_name,
                "primary_metric_mean": float(primary_values.mean()) if len(primary_values) > 0 else None,
                "primary_metric_std": float(primary_values.std()) if len(primary_values) > 0 else None,
                "accuracy_mean": float(method_df["eval_accuracy"].mean()) if method_df["eval_accuracy"].notna().any() else None,
                "accuracy_std": float(method_df["eval_accuracy"].std()) if method_df["eval_accuracy"].notna().any() else None,
                "f1_mean": float(method_df["eval_f1"].mean()) if method_df["eval_f1"].notna().any() else None,
                "f1_std": float(method_df["eval_f1"].std()) if method_df["eval_f1"].notna().any() else None,
                "num_seeds": len(method_df)
            }
    
    # Save summary stats to JSON
    json_file = output_dir / "performance_summary.json"
    with open(json_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"✓ Saved summary to: {json_file}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*80)
    for task, methods in summary_stats.items():
        logger.info(f"\n{task.upper()}:")
        for method, stats in methods.items():
            metric_name = stats["primary_metric_name"]
            metric_mean = stats["primary_metric_mean"]
            metric_std = stats["primary_metric_std"]
            logger.info(f"  {method:15s} | {metric_name}: {metric_mean:.4f} ± {metric_std:.4f} | n={stats['num_seeds']}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Performance metrics collection complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

