#!/usr/bin/env python3
"""
Collect final performance metrics from WandB for Phase 2/3 experiments.
This script aggregates accuracy, F1, and other metrics for all tasks, methods, and seeds.
"""

import wandb
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_phase2_phase3_metrics(
    entity: str = "galavny-tel-aviv-university",
    projects: List[str] = ["NLP-Phase2", "NLP-Phase2-FullFT", "NLP-Phase2-LoRA", "NLP"],
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
                eval_accuracy = summary.get("eval_accuracy") or summary.get("final_eval_accuracy") or summary.get("eval/accuracy")
                eval_f1 = summary.get("eval_f1") or summary.get("eval_f1_binary") or summary.get("final_eval_f1") or summary.get("eval/f1")
                eval_loss = summary.get("eval_loss") or summary.get("final_eval_loss") or summary.get("eval/loss")
                
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
    """Main function."""
    output_dir = Path("results/drift_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics
    results = collect_phase2_phase3_metrics()
    
    if results["dataframe"].empty:
        logger.error("❌ No results collected. Check WandB projects.")
        return
    
    # Save to CSV
    csv_file = output_dir / "performance_metrics.csv"
    results["dataframe"].to_csv(csv_file, index=False)
    logger.info(f"\n✓ Saved metrics to: {csv_file}")
    
    # Save summary stats to JSON
    json_file = output_dir / "performance_summary.json"
    with open(json_file, 'w') as f:
        json.dump(results["summary_stats"], f, indent=2)
    logger.info(f"✓ Saved summary to: {json_file}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Performance metrics collection complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

