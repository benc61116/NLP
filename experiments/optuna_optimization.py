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
        
        # Configure Optuna study
        sampler = TPESampler(seed=42, n_startup_trials=10)  # 10 random trials before TPE
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
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
        }
        
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
                reinit=True
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
                experiment = FullFinetuneExperiment(
                    task_name=self.task,
                    model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                    output_dir=str(self.output_dir / f"trial_{trial.number}")
                )
            elif self.method == "lora":
                experiment = LoRAExperiment(
                    task_name=self.task,
                    model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                    output_dir=str(self.output_dir / f"trial_{trial.number}")
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Configure for Optuna trial
            experiment.config['training'].update({
                'extract_base_model_representations': False,  # Disabled for speed
                'save_final_representations': False,  # Disabled for speed
                'num_train_epochs': hyperparams['num_train_epochs'],
                'per_device_train_batch_size': hyperparams['per_device_train_batch_size'],
                'learning_rate': hyperparams['learning_rate'],
                'warmup_ratio': hyperparams['warmup_ratio'],
                'weight_decay': hyperparams['weight_decay'],
                'evaluation_strategy': 'steps',
                'eval_steps': 50,  # Frequent evaluation for pruning
                'save_strategy': 'no',  # Don't save checkpoints during optimization
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_accuracy' if self.task != 'squad_v2' else 'eval_f1',
                'greater_is_better': True,
                'seed': 42,  # Fixed seed for reproducibility
                'dataloader_num_workers': 2,
                'remove_unused_columns': True,
                'report_to': ["wandb"],
            })
            
            # LoRA-specific configuration
            if self.method == "lora":
                experiment.config['lora'].update({
                    'r': hyperparams['lora_r'],
                    'lora_alpha': hyperparams['lora_alpha'],
                    'lora_dropout': hyperparams['lora_dropout'],
                })
            
            # Run training with Optuna pruning callback
            logger.info(f"Trial {trial.number}: Starting training with hyperparams: {hyperparams}")
            
            # Custom callback for Optuna pruning
            class OptunaPruningCallback:
                def __init__(self, trial, eval_metric):
                    self.trial = trial
                    self.eval_metric = eval_metric
                
                def __call__(self, logs):
                    # Report intermediate values for pruning
                    if self.eval_metric in logs:
                        step = logs.get('step', 0)
                        value = logs[self.eval_metric]
                        self.trial.report(value, step)
                        
                        # Check if trial should be pruned
                        if self.trial.should_prune():
                            wandb.log({"optuna/pruned": True, "optuna/pruned_at_step": step})
                            raise optuna.TrialPruned()
            
            # Set up pruning callback
            eval_metric = 'eval_accuracy' if self.task != 'squad_v2' else 'eval_f1'
            pruning_callback = OptunaPruningCallback(trial, eval_metric)
            
            # Run the experiment
            results = experiment.run_experiment(
                seeds=[42],  # Single seed for speed during optimization
                custom_log_callback=pruning_callback
            )
            
            # Extract the best metric value
            if results and len(results) > 0:
                best_result = results[0]  # Single seed
                if self.task == 'squad_v2':
                    objective_value = best_result['metrics']['eval_f1']
                else:
                    objective_value = best_result['metrics']['eval_accuracy']
            else:
                logger.error(f"Trial {trial.number}: No results returned")
                objective_value = 0.0
            
            # Log final result
            wandb.log({
                "optuna/final_objective": objective_value,
                "optuna/completed": True
            })
            
            logger.info(f"Trial {trial.number}: Objective value = {objective_value:.4f}")
            
            # Clean up
            wandb.finish()
            torch.cuda.empty_cache()
            
            return objective_value
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number}: Pruned")
            wandb.log({"optuna/pruned": True})
            wandb.finish()
            torch.cuda.empty_cache()
            raise
            
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed with error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Log error to W&B
            if wandb.run is not None:
                wandb.log({"optuna/error": str(e), "optuna/failed": True})
                wandb.finish()
                
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
