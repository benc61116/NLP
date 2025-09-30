#!/usr/bin/env python3
"""Optuna-based hyperparameter optimization for NLP experiments.

This replaces W&B sweeps with more efficient Bayesian optimization using Optuna.
Implements Tree-structured Parzen Estimator (TPE) for academic-grade hyperparameter search.
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.full_finetune import FullFinetuneExperiment
from experiments.lora_finetune import LoRAExperiment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """Bayesian optimization for hyperparameter tuning using Optuna."""
    
    def __init__(self, task: str, method: str, n_trials: int = 30, 
                 pruning: bool = True, wandb_project: str = "NLP-Phase1-Optuna"):
        """Initialize Optuna optimizer.
        
        Args:
            task: Task name (mrpc, sst2, rte, squad_v2)
            method: Method (full_finetune, lora)  
            n_trials: Number of optimization trials (default: 30 for research efficiency)
            pruning: Enable MedianPruner for early stopping of poor trials
            wandb_project: W&B project for logging
        """
        self.task = task
        self.method = method
        self.n_trials = n_trials
        self.wandb_project = wandb_project
        
        # Create study name for reproducibility
        self.study_name = f"{task}_{method}_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure Optuna study with per-task seed for independent exploration
        task_seed = hash(f"{task}_{method}") % 10000  # Reproducible but different per task
        sampler = TPESampler(seed=task_seed, n_startup_trials=10)  # 10 random trials before TPE
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=4, interval_steps=1) if pruning else None
        
        self.study = optuna.create_study(
            direction="maximize",  # Maximize evaluation metric
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )
        
        # Create output directory
        self.output_dir = Path("results/optuna") / task / method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Optuna optimizer: {self.study_name}")
        logger.info(f"Target: {n_trials} trials with TPE sampler + median pruning")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.
        
        Based on literature best practices and empirical ranges for TinyLlama fine-tuning.
        """
        
        # Common hyperparameters for both methods
        hyperparams = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
        }
        
        # Method-specific batch size suggestions (Optuna requires fixed categories)
        if self.method == "full_finetune":
            # EMERGENCY: Ultra-conservative batch sizes for full fine-tuning to avoid OOM
            hyperparams["per_device_train_batch_size"] = trial.suggest_categorical("per_device_train_batch_size", [1])  # FORCE batch size 1
        else:
            # More aggressive batch sizes for LoRA
            hyperparams["per_device_train_batch_size"] = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        
        # Task-specific adjustments
        if self.task == "squad_v2":
            # QA tasks typically need more epochs and lower learning rates
            hyperparams["num_train_epochs"] = trial.suggest_int("num_train_epochs", 3, 8)
            hyperparams["learning_rate"] = trial.suggest_float("learning_rate", 5e-7, 1e-4, log=True)
        
        # Method-specific parameters
        if self.method == "lora":
            hyperparams.update({
                "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32, 64]),
                "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.3),
            })
        
        return hyperparams
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        
        try:
            # Get hyperparameters for this trial
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Initialize W&B run for this trial
            wandb_run = wandb.init(
                project=self.wandb_project,
                group=f"{self.task}_{self.method}",
                job_type="optuna_trial",
                name=f"{self.study_name}_trial_{trial.number}",
                config=hyperparams,
                reinit="finish_previous"
            )
            
            # Log Optuna trial info
            wandb.log({
                "optuna/trial_number": trial.number,
                "optuna/study_name": self.study_name,
                "optuna/method": self.method,
                "optuna/task": self.task
            })
            
            # Create experiment instance
            if self.method == "full_finetune":
                experiment = FullFinetuneExperiment()
            elif self.method == "lora":
                experiment = LoRAExperiment()
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Configure model to use TinyLlama and disable expensive features for speed
            experiment.config['model']['name'] = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
            experiment.config['training']['extract_base_model_representations'] = False
            experiment.config['training']['save_final_representations'] = False
            
            # CRITICAL: Aggressive memory optimization for Optuna trials
            experiment.config['training']['gradient_checkpointing'] = True
            experiment.config['training']['dataloader_pin_memory'] = False  # Reduce memory
            experiment.config['training']['dataloader_num_workers'] = 0     # Reduce memory
            experiment.config['training']['per_device_eval_batch_size'] = 1  # Minimize eval memory
            experiment.config['training']['gradient_accumulation_steps'] = max(8 // hyperparams['per_device_train_batch_size'], 1)  # Maintain effective batch size
            experiment.config['training']['max_grad_norm'] = 1.0  # Prevent gradient explosion in small batches
            
            # CRITICAL: Disable model checkpoint saving for Optuna (we only need metrics, not models)
            experiment.config['training']['save_strategy'] = 'no'  # Don't save checkpoints during training
            experiment.config['training']['save_total_limit'] = 0  # Don't keep any checkpoints
            experiment.config['training']['load_best_model_at_end'] = False  # Don't need to load best model
            experiment.config['training']['eval_strategy'] = 'no'  # Disable evaluation to match save strategy
            
            # CRITICAL: Disable representation extraction to save GPU memory (major OOM source)
            experiment.config['training']['extract_base_model_representations'] = False
            experiment.config['training']['save_final_representations'] = False
            experiment.config['training']['extract_representations_every_steps'] = None  # Disable step-based extraction
            
            # Memory optimization for ALL tasks (QA and Classification)
            if self.task == 'squad_v2':
                experiment.config['model']['max_length'] = 384  # QA needs longer context for Q+A pairs
                experiment.config['tasks']['squad_v2']['max_samples_train'] = 3000  # Research-grade: 2.3% coverage
                experiment.config['tasks']['squad_v2']['max_samples_eval'] = 300   # Proportional eval set
                
                # CRITICAL: Memory optimizations for 22GB GPU
                # The key issue: eval creates massive activation memory with large eval sets
                experiment.config['training']['per_device_eval_batch_size'] = 1  # Reduce eval batch to 1
                experiment.config['training']['eval_accumulation_steps'] = 4  # Process eval in chunks
                
                # Let num_train_epochs (suggested by Optuna) control duration
                # No max_steps limit - proper full epoch training
                if self.method == "full_finetune":
                    experiment.config['training']['logging_steps'] = 100
                    experiment.config['training']['gradient_accumulation_steps'] = 8  # Increased: simulate batch_size=8
                else:
                    # LoRA settings - can use smaller gradient accumulation (less memory intensive)
                    experiment.config['training']['logging_steps'] = 100
                    experiment.config['training']['gradient_accumulation_steps'] = 4
            
            # CRITICAL FIX: Add memory optimizations for classification tasks (MRPC, SST-2, RTE)
            elif self.task == 'sst2':
                # Large dataset: Research-grade coverage (3000 = 4.5% of 67K)
                experiment.config['tasks']['sst2']['max_samples_train'] = 3000
                experiment.config['tasks']['sst2']['max_samples_eval'] = 150  # ~17% of validation set
                experiment.config['model']['max_length'] = 256  # Shorter sequences for classification
                
                # Memory optimizations for classification
                experiment.config['training']['eval_accumulation_steps'] = 1  # Process eval in smaller chunks
                
            elif self.task in ['mrpc', 'rte']:
                # Small datasets: Already good coverage at 500 (13-20%)
                experiment.config['tasks'][self.task]['max_samples_train'] = 500
                experiment.config['tasks'][self.task]['max_samples_eval'] = 50
                experiment.config['model']['max_length'] = 256  # Shorter sequences for classification
                
                # Memory optimizations for classification
                experiment.config['training']['eval_accumulation_steps'] = 1  # Process eval in smaller chunks
                
                # Let num_train_epochs control duration
                if self.method == "full_finetune":
                    experiment.config['training']['logging_steps'] = 50
                    experiment.config['training']['gradient_accumulation_steps'] = 2
                else:
                    # LoRA settings
                    experiment.config['training']['logging_steps'] = 50
            
            # Run single experiment with suggested hyperparameters
            # The run_single_experiment method handles all configuration including LoRA params
            
            logger.info(f"Trial {trial.number}: Starting training with hyperparams: {hyperparams}")
            
            # Set memory optimization environment variable
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Clear GPU cache before each trial
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Run the experiment using the correct method
            results = experiment.run_single_experiment(
                task_name=self.task,
                seed=42,  # Fixed seed for reproducibility
                skip_wandb_init=True,  # W&B already initialized above
                **hyperparams
            )
            
            # Extract the best metric value with comprehensive error handling
            if results and 'eval_metrics' in results and results['eval_metrics']:
                eval_metrics = results['eval_metrics']
                logger.info(f"Trial {trial.number}: Available eval metrics: {list(eval_metrics.keys())}")
                
                if self.task == 'squad_v2':
                    # For SQuAD v2, try multiple possible F1 keys
                    if 'eval_f1' in eval_metrics:
                        objective_value = eval_metrics['eval_f1']
                    elif 'eval_exact_match' in eval_metrics:
                        # Fallback to exact match if F1 not available
                        objective_value = eval_metrics['eval_exact_match'] 
                        logger.warning(f"Trial {trial.number}: Using exact_match instead of F1: {objective_value:.4f}")
                    elif any('f1' in k.lower() for k in eval_metrics.keys()):
                        # Find any F1-like metric
                        f1_key = next(k for k in eval_metrics.keys() if 'f1' in k.lower())
                        objective_value = eval_metrics[f1_key]
                        logger.warning(f"Trial {trial.number}: Using {f1_key} instead of eval_f1: {objective_value:.4f}")
                    else:
                        logger.error(f"Trial {trial.number}: No F1 or exact match metric found in {list(eval_metrics.keys())}")
                        objective_value = 0.0
                else:
                    # For classification tasks, try multiple possible accuracy keys
                    if 'eval_accuracy' in eval_metrics:
                        objective_value = eval_metrics['eval_accuracy']
                    elif any('accuracy' in k.lower() for k in eval_metrics.keys()):
                        # Find any accuracy-like metric
                        acc_key = next(k for k in eval_metrics.keys() if 'accuracy' in k.lower())
                        objective_value = eval_metrics[acc_key]
                        logger.warning(f"Trial {trial.number}: Using {acc_key} instead of eval_accuracy: {objective_value:.4f}")
                    else:
                        logger.error(f"Trial {trial.number}: No accuracy metric found in {list(eval_metrics.keys())}")
                        objective_value = 0.0
            else:
                logger.error(f"Trial {trial.number}: Experiment failed or no eval_metrics returned")
                if results:
                    logger.error(f"Trial {trial.number}: Available result keys: {list(results.keys())}")
                    if 'error' in results:
                        logger.error(f"Trial {trial.number}: Error was: {results['error']}")
                objective_value = 0.0
            
            # Log final result
            wandb.log({
                "optuna/final_objective": objective_value,
                "optuna/completed": True
            })
            
            logger.info(f"Trial {trial.number}: Objective value = {objective_value:.4f}")
            
            # Clean up W&B and local files to save disk space
            import shutil
            from pathlib import Path
            
            # Get current wandb run directory before finishing
            wandb_run_dir = Path(wandb.run.dir).parent if wandb.run else None
            
            # Finish W&B (syncs data)
            wandb.finish()
            
            # CRITICAL: Delete wandb run directory after sync to save disk space
            if wandb_run_dir and wandb_run_dir.exists():
                try:
                    shutil.rmtree(wandb_run_dir)
                    logger.info(f"Cleaned up W&B run directory: {wandb_run_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean W&B directory: {e}")
            
            # CRITICAL: Delete results directory for this trial (we only need metrics in W&B)
            # Find and delete all results directories created in this trial
            results_base = Path("results")
            if results_base.exists():
                # Get all directories created in the last few minutes (this trial)
                import time
                current_time = time.time()
                for results_dir in results_base.glob("full_finetune_*"):
                    # Delete if created recently (within this trial's runtime)
                    if results_dir.is_dir() and (current_time - results_dir.stat().st_mtime) < 3600:  # Last hour
                        try:
                            shutil.rmtree(results_dir)
                            logger.info(f"Cleaned up results directory: {results_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to clean results directory {results_dir}: {e}")
                
                # Also clean lora directories
                for results_dir in results_base.glob("lora_finetune_*"):
                    if results_dir.is_dir() and (current_time - results_dir.stat().st_mtime) < 3600:
                        try:
                            shutil.rmtree(results_dir)
                            logger.info(f"Cleaned up LoRA results directory: {results_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to clean LoRA results directory {results_dir}: {e}")
            
            # Clean GPU cache
            torch.cuda.empty_cache()
            
            # Report disk usage
            total, used, free = shutil.disk_usage('/')
            usage_percent = (used / total) * 100
            logger.info(f"💾 Disk usage after cleanup: {usage_percent:.1f}% ({used//(1024**3)}GB used, {free//(1024**3)}GB free)")
            
            return objective_value
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number}: Pruned")
            wandb.log({"optuna/pruned": True})
            
            # Clean up even for pruned trials
            import shutil
            from pathlib import Path
            
            wandb_run_dir = Path(wandb.run.dir).parent if wandb.run else None
            wandb.finish()
            
            if wandb_run_dir and wandb_run_dir.exists():
                try:
                    shutil.rmtree(wandb_run_dir)
                except Exception:
                    pass
            
            torch.cuda.empty_cache()
            raise
            
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed with error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Log error to W&B and clean up
            import shutil
            from pathlib import Path
            
            if wandb.run is not None:
                wandb.log({"optuna/error": str(e), "optuna/failed": True})
                wandb_run_dir = Path(wandb.run.dir).parent
                wandb.finish()
                
                # Clean up failed trial's wandb directory
                if wandb_run_dir.exists():
                    try:
                        shutil.rmtree(wandb_run_dir)
                    except Exception:
                        pass
                
            torch.cuda.empty_cache()
            return 0.0  # Return worst possible value for failed trials
    
    def optimize(self) -> Dict[str, Any]:
        """Run Optuna optimization."""
        
        logger.info(f"Starting Optuna optimization: {self.study_name}")
        logger.info(f"Target: {self.n_trials} trials")
        
        # Run optimization
        try:
            self.study.optimize(self.objective, n_trials=self.n_trials, timeout=None)
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        # Get best trial
        best_trial = self.study.best_trial
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {best_trial.params}")
        
        # Prepare results
        results = {
            "study_name": self.study_name,
            "task": self.task,
            "method": self.method,
            "n_trials": len(self.study.trials),
            "n_completed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "hyperparameters": best_trial.params
            },
            "optimization_history": [
                {"trial": i, "value": trial.value, "state": trial.state.name}
                for i, trial in enumerate(self.study.trials)
                if trial.value is not None
            ]
        }
        
        # Save results
        results_file = self.output_dir / "optuna_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Final cleanup: Remove any remaining wandb and results directories
        logger.info("Performing final cleanup of trial artifacts...")
        import shutil
        
        # Clean wandb directories
        wandb_dir = Path("wandb")
        if wandb_dir.exists():
            for run_dir in wandb_dir.glob("run-*"):
                try:
                    shutil.rmtree(run_dir)
                    logger.info(f"Cleaned up: {run_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean {run_dir}: {e}")
        
        # Clean wandb cache to free up space
        try:
            import subprocess
            result = subprocess.run(["wandb", "artifact", "cache", "cleanup", "1GB"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info("Cleaned wandb artifact cache")
            else:
                logger.warning(f"Wandb cache cleanup: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not clean wandb cache: {e}")
        
        # Report final disk usage
        total, used, free = shutil.disk_usage('/')
        usage_percent = (used / total) * 100
        logger.info(f"💾 Final disk usage: {usage_percent:.1f}% ({used//(1024**3)}GB used, {free//(1024**3)}GB free)")
        
        return results

def main():
    """Main function for Optuna optimization."""
    
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--task", type=str, required=True,
                      choices=["mrpc", "sst2", "rte", "squad_v2"],
                      help="Task to optimize")
    parser.add_argument("--method", type=str, required=True,
                      choices=["full_finetune", "lora"],
                      help="Method to optimize")
    parser.add_argument("--n-trials", type=int, default=30,
                      help="Number of Optuna trials (default: 30)")
    parser.add_argument("--no-pruning", action="store_true",
                      help="Disable median pruning")
    parser.add_argument("--wandb-project", type=str, default="NLP-Phase1-Optuna",
                      help="W&B project name")
    parser.add_argument("--output-file", type=str,
                      help="Output file for best hyperparameters (YAML format)")
    
    args = parser.parse_args()
    
    # Ensure reproducibility
    torch.manual_seed(42)
    
    # Initialize optimizer
    optimizer = OptunaOptimizer(
        task=args.task,
        method=args.method,
        n_trials=args.n_trials,
        pruning=not args.no_pruning,
        wandb_project=args.wandb_project
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save optimal hyperparameters in format expected by VM scripts
    if args.output_file:
        optimal_config = {
            "task": args.task,
            "method": args.method,
            "best_hyperparameters": results["best_trial"]["hyperparameters"],
            "expected_performance": results["best_trial"]["value"],
            "optimization_summary": {
                "n_trials": results["n_trials"],
                "n_completed": results["n_completed_trials"],
                "n_pruned": results["n_pruned_trials"]
            }
        }
        
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)
        
        logger.info(f"Optimal hyperparameters saved to: {output_path}")
    
    logger.info("Optuna optimization completed successfully!")

if __name__ == "__main__":
    main()
