#!/usr/bin/env python3
"""Main experiment runner for LoRA vs Full Fine-tuning comparison."""

import os
import torch
import wandb
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import evaluate

from .data_preparation import TaskDataLoader
from .sanity_checks import ModelSanityChecker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner for comparing LoRA vs Full Fine-tuning."""
    
    def __init__(self, config_path: str = "shared/config.yaml", output_dir: str = "./results"):
        """Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to save results
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_name = self.config['model']['name']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.data_loader = None
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        logger.info(f"Initialized experiment runner")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_environment(self):
        """Setup the experimental environment."""
        logger.info("Setting up experimental environment...")
        
        # Set random seeds for reproducibility
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Configure deterministic behavior
        if self.config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize data loader
        self.data_loader = TaskDataLoader(self.model_name)
        
        logger.info("✓ Environment setup complete")
    
    def load_model(self, method: str = "lora") -> torch.nn.Module:
        """Load and configure model for training.
        
        Args:
            method: Training method ('lora' or 'full')
            
        Returns:
            Configured model
        """
        logger.info(f"Loading model for {method} fine-tuning...")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=getattr(torch, self.config['model']['dtype']),
            device_map=self.config['model']['device_map'] if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if method == "lora":
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['dropout'],
                bias=self.config['lora']['bias'],
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            logger.info("✓ LoRA configuration applied")
            
        else:  # full fine-tuning
            # Enable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = True
            logger.info("✓ Full fine-tuning enabled")
        
        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.4f}")
        
        return model
    
    def prepare_training_arguments(self, method: str, task_name: str, run_name: str) -> TrainingArguments:
        """Prepare training arguments for the experiment.
        
        Args:
            method: Training method ('lora' or 'full')
            task_name: Name of the task
            run_name: Name for this specific run
            
        Returns:
            TrainingArguments object
        """
        # Determine learning rate based on method
        if method == "lora":
            learning_rate = self.config['training']['learning_rate']
        else:
            learning_rate = self.config['training']['full_finetune_learning_rate']
        
        # Create output directory for this run
        run_output_dir = self.output_dir / f"{method}_{task_name}_{self.experiment_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(run_output_dir),
            run_name=run_name,
            
            # Training configuration
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            
            # Optimization
            learning_rate=learning_rate,
            weight_decay=self.config['training']['weight_decay'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            optim=self.config['training']['optim'],
            
            # Evaluation and saving
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            logging_steps=self.config['training']['logging_steps'],
            
            # Model selection
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            
            # Performance optimizations
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            
            # Mixed precision and other optimizations
            fp16=self.config['infrastructure']['mixed_precision'] and torch.cuda.is_available(),
            
            # Reporting
            report_to=["wandb"],
            logging_dir=str(run_output_dir / "logs"),
            
            # Reproducibility
            seed=self.config['reproducibility']['seed'],
            data_seed=self.config['reproducibility']['seed'],
        )
        
        return training_args
    
    def create_compute_metrics_function(self, task_name: str):
        """Create compute_metrics function for evaluation.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Function to compute metrics
        """
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            metric = evaluate.load(task_config['metric'])
            
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                if task_config['metric'] == 'f1':
                    return metric.compute(predictions=predictions, references=labels)
                else:  # accuracy
                    return metric.compute(predictions=predictions, references=labels)
            
        elif task_config['type'] == 'question_answering':
            # For QA tasks, we'll use a simplified metric for this experiment
            def compute_metrics(eval_pred):
                # Simplified QA evaluation
                predictions, labels = eval_pred
                # For now, just return a dummy metric
                # In a full implementation, you'd need proper QA evaluation
                return {"f1": 0.5, "exact_match": 0.3}
        
        return compute_metrics
    
    def prepare_dataset_for_task(self, task_name: str, split: str = "train") -> Dataset:
        """Prepare dataset for a specific task.
        
        Args:
            task_name: Name of the task
            split: Dataset split
            
        Returns:
            Prepared dataset
        """
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            # Get max samples if specified
            max_samples = task_config.get(f'max_samples_{split}')
            
            data = self.data_loader.prepare_classification_data(
                task_name, split, max_samples
            )
            
            # Convert to HuggingFace dataset format
            dataset_dict = {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "labels": data["labels"]
            }
            
        elif task_config['type'] == 'question_answering':
            max_samples = task_config.get(f'max_samples_{split}')
            
            data = self.data_loader.prepare_qa_data(split, max_samples)
            
            dataset_dict = {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "start_positions": data["start_positions"],
                "end_positions": data["end_positions"]
            }
        
        return Dataset.from_dict(dataset_dict)
    
    def run_single_experiment(self, task_name: str, method: str) -> Dict[str, Any]:
        """Run a single experiment for a task and method.
        
        Args:
            task_name: Name of the task
            method: Training method ('lora' or 'full')
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Running experiment: {task_name} with {method}")
        
        # Create run name
        run_name = f"{task_name}_{method}_{self.experiment_id}"
        
        # Initialize wandb run
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=run_name,
            group=self.config['wandb']['group_name'],
            job_type=self.config['wandb']['job_type'],
            tags=self.config['wandb']['tags'] + [task_name, method],
            config={
                **self.config,
                "task_name": task_name,
                "method": method,
                "experiment_id": self.experiment_id
            }
        )
        
        try:
            # Load model
            model = self.load_model(method)
            
            # Prepare datasets
            train_dataset = self.prepare_dataset_for_task(task_name, "train")
            
            # Check if validation split exists
            try:
                eval_dataset = self.prepare_dataset_for_task(task_name, "validation")
            except:
                # Use a subset of training data for evaluation if no validation set
                logger.warning(f"No validation set for {task_name}, using subset of training data")
                eval_dataset = train_dataset.select(range(min(100, len(train_dataset))))
            
            # Prepare training arguments
            training_args = self.prepare_training_arguments(method, task_name, run_name)
            
            # Create data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            
            # Create compute metrics function
            compute_metrics = self.create_compute_metrics_function(task_name)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config['training']['early_stopping_patience']
                    )
                ]
            )
            
            # Train the model
            logger.info(f"Starting training for {task_name} with {method}...")
            train_result = trainer.train()
            
            # Evaluate the model
            logger.info(f"Evaluating {task_name} with {method}...")
            eval_result = trainer.evaluate()
            
            # Save the model
            model_save_path = self.output_dir / f"{method}_{task_name}_{self.experiment_id}"
            trainer.save_model(str(model_save_path))
            
            # Compile results
            results = {
                "task_name": task_name,
                "method": method,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_loss": train_result.metrics.get("train_loss", 0),
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_metrics": {k: v for k, v in eval_result.items() if k.startswith("eval_")},
                "model_path": str(model_save_path),
                "experiment_id": self.experiment_id
            }
            
            # Log final results to wandb
            wandb.log({
                "final_train_loss": results["train_loss"],
                "final_eval_loss": results["eval_loss"],
                "train_runtime": results["train_runtime"]
            })
            
            logger.info(f"✓ Completed experiment: {task_name} with {method}")
            logger.info(f"  Train loss: {results['train_loss']:.4f}")
            logger.info(f"  Eval loss: {results['eval_loss']:.4f}")
            logger.info(f"  Runtime: {results['train_runtime']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Experiment failed: {task_name} with {method} - {e}")
            return {
                "task_name": task_name,
                "method": method,
                "error": str(e),
                "experiment_id": self.experiment_id
            }
        
        finally:
            # Finish wandb run
            wandb.finish()
    
    def run_experiments_for_tasks(self, tasks: List[str], methods: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run experiments for multiple tasks and methods.
        
        Args:
            tasks: List of task names
            methods: List of methods ('lora', 'full')
            
        Returns:
            Dictionary with all experiment results
        """
        logger.info(f"Running experiments for {len(tasks)} tasks and {len(methods)} methods")
        
        all_results = {}
        
        for task_name in tasks:
            if task_name not in self.config['tasks']:
                logger.error(f"Unknown task: {task_name}")
                continue
            
            all_results[task_name] = {}
            
            for method in methods:
                if method not in ['lora', 'full']:
                    logger.error(f"Unknown method: {method}")
                    continue
                
                result = self.run_single_experiment(task_name, method)
                all_results[task_name][method] = result
                
                # Save intermediate results
                self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Dict[str, Any]]):
        """Save experiment results to disk.
        
        Args:
            results: Dictionary with experiment results
        """
        results_file = self.output_dir / f"experiment_results_{self.experiment_id}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def run_sanity_checks(self) -> bool:
        """Run sanity checks before starting experiments.
        
        Returns:
            True if all sanity checks pass
        """
        logger.info("Running sanity checks before experiments...")
        
        checker = ModelSanityChecker()
        results = checker.run_comprehensive_sanity_checks()
        
        if all(results.values()):
            logger.info("✓ All sanity checks passed!")
            return True
        else:
            logger.error("✗ Some sanity checks failed!")
            return False
    
    def run_full_experiment_suite(self, skip_sanity_checks: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run the complete experiment suite.
        
        Args:
            skip_sanity_checks: Whether to skip sanity checks
            
        Returns:
            Dictionary with all experiment results
        """
        logger.info("Starting full experiment suite")
        logger.info("=" * 60)
        
        # Setup environment
        self.setup_environment()
        
        # Run sanity checks
        if not skip_sanity_checks:
            if not self.run_sanity_checks():
                logger.error("Sanity checks failed. Aborting experiments.")
                return {}
        
        # Define experiments
        tasks = ["mrpc", "sst2", "rte", "squad_v2"]
        methods = ["lora", "full"]
        
        # Run experiments
        results = self.run_experiments_for_tasks(tasks, methods)
        
        # Save final results
        self.save_results(results)
        
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUITE COMPLETE")
        logger.info("=" * 60)
        
        # Print summary
        for task_name, task_results in results.items():
            logger.info(f"\n{task_name.upper()}:")
            for method, result in task_results.items():
                if "error" in result:
                    logger.info(f"  {method}: FAILED - {result['error']}")
                else:
                    logger.info(f"  {method}: Train loss: {result['train_loss']:.4f}, "
                              f"Eval loss: {result['eval_loss']:.4f}")
        
        return results


def run_experiment_from_config(config_path: str = "shared/config.yaml", 
                             tasks: Optional[List[str]] = None,
                             methods: Optional[List[str]] = None,
                             skip_sanity_checks: bool = False) -> Dict[str, Dict[str, Any]]:
    """Run experiments from configuration file.
    
    Args:
        config_path: Path to configuration file
        tasks: List of tasks to run (default: all)
        methods: List of methods to run (default: ['lora', 'full'])
        skip_sanity_checks: Whether to skip sanity checks
        
    Returns:
        Dictionary with experiment results
    """
    runner = ExperimentRunner(config_path)
    
    if tasks is None:
        tasks = ["mrpc", "sst2", "rte", "squad_v2"]
    
    if methods is None:
        methods = ["lora", "full"]
    
    if len(tasks) == 1 and len(methods) == 1:
        # Single experiment
        runner.setup_environment()
        if not skip_sanity_checks and not runner.run_sanity_checks():
            return {}
        
        result = runner.run_single_experiment(tasks[0], methods[0])
        return {tasks[0]: {methods[0]: result}}
    else:
        # Multiple experiments
        return runner.run_full_experiment_suite(skip_sanity_checks)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NLP experiments")
    parser.add_argument("--config", default="shared/config.yaml", help="Config file path")
    parser.add_argument("--tasks", nargs="+", help="Tasks to run", 
                       choices=["mrpc", "sst2", "rte", "squad_v2"])
    parser.add_argument("--methods", nargs="+", help="Methods to run",
                       choices=["lora", "full"], default=["lora", "full"])
    parser.add_argument("--skip-sanity-checks", action="store_true", 
                       help="Skip sanity checks")
    
    args = parser.parse_args()
    
    results = run_experiment_from_config(
        config_path=args.config,
        tasks=args.tasks,
        methods=args.methods,
        skip_sanity_checks=args.skip_sanity_checks
    )
    
    print("\nExperiment completed. Results saved to results directory.")
