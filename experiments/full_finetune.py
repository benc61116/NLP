#!/usr/bin/env python3
"""Full fine-tuning experiments for Llama-2-1.3B with comprehensive tracking and representation extraction."""

import os
import sys
import warnings

# Suppress common warnings for cleaner output
import transformers
transformers.logging.set_verbosity_error()  # Only show errors
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*were not initialized.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*reinit.*deprecated.*")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

import torch
import wandb
import yaml
import json
import logging
import numpy as np
import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset
import evaluate

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_preparation import TaskDataLoader
from shared.metrics import compute_classification_metrics, compute_qa_metrics
from shared.checkpoint_utils import CheckpointManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QADataCollator:
    """Custom data collator for QA tasks that handles start/end positions."""
    
    def __init__(self, tokenizer, padding=True, device=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.device = device
    
    def __call__(self, features):
        # Extract each field
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        start_positions = [f["start_positions"] for f in features]
        end_positions = [f["end_positions"] for f in features]
        
        # Pad input_ids and attention_mask
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                # Pad with tokenizer.pad_token_id
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids.append(ids + [pad_id] * padding_length)
                padded_attention_mask.append(mask + [0] * padding_length)
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
        
        # Create batch tensors (device will be handled by Trainer automatically)
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }
        
        return batch


@dataclass
class RepresentationConfig:
    """Configuration for representation extraction."""
    extract_every_steps: int = 100
    save_layers: List[int] = field(default_factory=lambda: list(range(24)))  # All layers for 1.3B
    max_validation_samples: int = 1000
    save_attention: bool = True
    save_mlp: bool = True
    memory_map: bool = True


class RepresentationExtractor:
    """Extracts and saves model representations during training."""
    
    def __init__(self, config: RepresentationConfig, output_dir: Path, task_name: str, method: str):
        self.config = config
        self.output_dir = output_dir / "representations" / f"{method}_{task_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.method = method
        self.step_counter = 0
        
        # Storage for representations
        self.representations = {}
        self.validation_examples = None
        
        logger.info(f"Initialized representation extractor: {self.output_dir}")
    
    def set_validation_examples(self, examples: Dict[str, torch.Tensor]):
        """Set validation examples for consistent representation extraction."""
        # Limit to max samples for memory efficiency
        max_samples = min(self.config.max_validation_samples, len(examples['input_ids']))
        
        self.validation_examples = {
            'input_ids': examples['input_ids'][:max_samples],
            'attention_mask': examples['attention_mask'][:max_samples],
        }
        if 'labels' in examples:
            self.validation_examples['labels'] = examples['labels'][:max_samples]
        
        logger.info(f"Set {max_samples} validation examples for representation extraction")
    
    def extract_representations(self, model: torch.nn.Module, step: int) -> Dict[str, torch.Tensor]:
        """Extract representations from the model."""
        if self.validation_examples is None:
            logger.warning("No validation examples set for representation extraction")
            return {}
        
        model.eval()
        representations = {}
        
        # Hook to capture layer outputs
        layer_outputs = {}
        hooks = []
        
        def create_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # For transformer layers, output[0] is hidden states
                    layer_outputs[layer_name] = output[0].detach().cpu()
                else:
                    layer_outputs[layer_name] = output.detach().cpu()
            return hook
        
        try:
            # Register hooks for specified layers
            if hasattr(model, 'model'):  # For models with wrapper
                base_model = model.model
            else:
                base_model = model
            
            # Hook transformer layers
            if hasattr(base_model, 'layers'):
                for i in self.config.save_layers:
                    if i < len(base_model.layers):
                        layer = base_model.layers[i]
                        hook = layer.register_forward_hook(create_hook(f'layer_{i}'))
                        hooks.append(hook)
            
            # Forward pass with validation examples
            with torch.no_grad():
                input_ids = self.validation_examples['input_ids'].to(model.device)
                attention_mask = self.validation_examples['attention_mask'].to(model.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Store extracted representations
                for layer_name, layer_output in layer_outputs.items():
                    representations[layer_name] = layer_output
                
                # Also store final hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    representations['final_hidden_states'] = outputs.hidden_states[-1].detach().cpu()
                elif hasattr(outputs, 'last_hidden_state'):
                    representations['final_hidden_states'] = outputs.last_hidden_state.detach().cpu()
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
            model.train()
        
        return representations
    
    def save_representations(self, representations: Dict[str, torch.Tensor], step: int):
        """Save representations to disk."""
        if not representations:
            return
        
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)
        
        for layer_name, tensor in representations.items():
            file_path = step_dir / f"{layer_name}.pt"
            torch.save(tensor, file_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'task_name': self.task_name,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.validation_examples['input_ids']),
            'layer_names': list(representations.keys()),
            'tensor_shapes': {name: list(tensor.shape) for name, tensor in representations.items()}
        }
        
        with open(step_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved representations for step {step} to {step_dir}")


class GradientStatsMonitor:
    """Monitors gradient statistics and norms during training."""
    
    def __init__(self, log_every_steps: int = 50):
        self.log_every_steps = log_every_steps
        self.gradient_stats = []
    
    def compute_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute gradient statistics for the model."""
        total_norm = 0.0
        param_count = 0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                total_norm += grad_norm ** 2
                param_count += param.numel()
        
        total_norm = total_norm ** 0.5
        
        if grad_norms:
            stats = {
                'gradient_norm_total': total_norm,
                'gradient_norm_mean': np.mean(grad_norms),
                'gradient_norm_std': np.std(grad_norms),
                'gradient_norm_max': np.max(grad_norms),
                'gradient_norm_min': np.min(grad_norms),
                'num_parameters_with_gradients': len(grad_norms),
                'total_parameters': param_count,
            }
        else:
            stats = {
                'gradient_norm_total': 0.0,
                'gradient_norm_mean': 0.0,
                'gradient_norm_std': 0.0,
                'gradient_norm_max': 0.0,
                'gradient_norm_min': 0.0,
                'num_parameters_with_gradients': 0,
                'total_parameters': param_count,
            }
        
        return stats


class MemoryProfiler:
    """Profiles GPU and CPU memory usage during training."""
    
    def __init__(self):
        self.memory_stats = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info()
        stats['cpu_memory_rss_mb'] = cpu_memory.rss / 1024 / 1024
        stats['cpu_memory_vms_mb'] = cpu_memory.vms / 1024 / 1024
        stats['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info = torch.cuda.memory_stats(i)
                stats[f'gpu_{i}_memory_allocated_mb'] = memory_info.get('allocated_bytes.all.current', 0) / 1024 / 1024
                stats[f'gpu_{i}_memory_reserved_mb'] = memory_info.get('reserved_bytes.all.current', 0) / 1024 / 1024
                stats[f'gpu_{i}_memory_max_allocated_mb'] = memory_info.get('allocated_bytes.all.peak', 0) / 1024 / 1024
        
        return stats


class FullFinetuneCallback(TrainerCallback):
    """Custom callback for full fine-tuning with comprehensive tracking."""
    
    def __init__(self, representation_extractor: RepresentationExtractor, 
                 gradient_monitor: GradientStatsMonitor,
                 memory_profiler: MemoryProfiler,
                 eval_dataset: Dataset,
                 extract_every_steps: int = 100):
        self.representation_extractor = representation_extractor
        self.gradient_monitor = gradient_monitor
        self.memory_profiler = memory_profiler
        self.eval_dataset = eval_dataset
        self.extract_every_steps = extract_every_steps
        
        # Set validation examples for representation extraction
        if eval_dataset is not None:
            examples = {
                'input_ids': torch.tensor([ex['input_ids'] for ex in eval_dataset]),
                'attention_mask': torch.tensor([ex['attention_mask'] for ex in eval_dataset])
            }
            # Handle both classification and QA tasks
            if 'labels' in eval_dataset[0]:
                examples['labels'] = torch.tensor([ex['labels'] for ex in eval_dataset])
            elif 'start_positions' in eval_dataset[0] and 'end_positions' in eval_dataset[0]:
                examples['start_positions'] = torch.tensor([ex['start_positions'] for ex in eval_dataset])
                examples['end_positions'] = torch.tensor([ex['end_positions'] for ex in eval_dataset])
            
            self.representation_extractor.set_validation_examples(examples)
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        model = kwargs.get('model')
        step = state.global_step
        
        # Log gradient statistics
        if step % self.gradient_monitor.log_every_steps == 0:
            grad_stats = self.gradient_monitor.compute_gradient_stats(model)
            if wandb.run is not None:
                wandb.log({f"gradients/{k}": v for k, v in grad_stats.items()}, step=step)
        
        # Log memory statistics
        memory_stats = self.memory_profiler.get_memory_stats()
        if wandb.run is not None:
            wandb.log({f"memory/{k}": v for k, v in memory_stats.items()}, step=step)
        
        # Extract representations
        if step % self.extract_every_steps == 0:
            logger.info(f"Extracting representations at step {step}")
            representations = self.representation_extractor.extract_representations(model, step)
            self.representation_extractor.save_representations(representations, step)
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called during evaluation."""
        step = state.global_step
        logger.info(f"Evaluation at step {step}")
        
        # Extract representations during evaluation
        model = kwargs.get('model')
        representations = self.representation_extractor.extract_representations(model, step)
        self.representation_extractor.save_representations(representations, step)


class FullFinetuneExperiment:
    """Main experiment runner for full fine-tuning with comprehensive tracking."""
    
    def __init__(self, config_path: str = "shared/config.yaml"):
        """Initialize the experiment."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.data_loader = None
        self.representation_config = RepresentationConfig()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/full_finetune_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(str(self.output_dir))
        
        logger.info(f"Initialized full fine-tuning experiment")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_environment(self):
        """Setup the experimental environment."""
        logger.info("Setting up experimental environment...")
        
        # Set random seeds
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic behavior
            if self.config['reproducibility']['deterministic']:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Initialize tokenizer
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize data loader
        self.data_loader = TaskDataLoader(model_name)
        
        logger.info("✓ Environment setup complete")
    
    def load_model(self, task_name: str = None) -> torch.nn.Module:
        """Load model for full fine-tuning."""
        logger.info("Loading model for full fine-tuning...")
        
        model_name = self.config['model']['name']
        
        # Calculate max memory based on our config
        max_memory = None
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory_percent = self.config['model'].get('max_memory_percent', 95)
            max_memory_bytes = int(total_memory * (max_memory_percent / 100))
            max_memory = {0: max_memory_bytes}
            logger.info(f"Using {max_memory_percent}% of GPU memory: {max_memory_bytes / (1024**3):.1f}GB")
        
        # Choose model type based on task
        if task_name and task_name in self.config['tasks']:
            task_type = self.config['tasks'][task_name].get('type', 'classification')
            if task_type in ['qa', 'question_answering']:
                # For QA tasks, use QuestionAnswering model with QA head
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    dtype=getattr(torch, self.config['model']['dtype']),
                    device_map="auto" if torch.cuda.is_available() else None,
                    max_memory=max_memory,
                    trust_remote_code=True
                )
            else:  # classification tasks
                num_labels = self.config['tasks'][task_name].get('num_labels', 2)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    dtype=getattr(torch, self.config['model']['dtype']),
                    device_map="auto" if torch.cuda.is_available() else None,
                    max_memory=max_memory,
                    trust_remote_code=True
                )
        else:
            # Default to classification with 2 labels
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                dtype=getattr(torch, self.config['model']['dtype']),
                device_map="auto" if torch.cuda.is_available() else None,
                max_memory=max_memory,
                trust_remote_code=True
            )
        
        # Enable gradients for all parameters (full fine-tuning)
        for param in model.parameters():
            param.requires_grad = True
        
        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.4f}")
        
        return model
    
    def prepare_datasets(self, task_name: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        logger.info(f"Preparing datasets for {task_name}")
        
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            # Training data
            train_data = self.data_loader.prepare_classification_data(
                task_name, 'train', task_config.get('max_samples_train')
            )
            train_dataset = Dataset.from_dict({
                "input_ids": train_data["input_ids"],
                "attention_mask": train_data["attention_mask"],
                "labels": train_data["labels"]
            })
            
            # Validation data
            try:
                eval_data = self.data_loader.prepare_classification_data(
                    task_name, 'validation', task_config.get('max_samples_eval')
                )
                eval_dataset = Dataset.from_dict({
                    "input_ids": eval_data["input_ids"],
                    "attention_mask": eval_data["attention_mask"],
                    "labels": eval_data["labels"]
                })
            except Exception as e:
                logger.warning(f"No validation set for {task_name}, using subset of training data")
                eval_dataset = train_dataset.select(range(min(500, len(train_dataset))))
        
        elif task_config['type'] == 'question_answering':
            # For SQuAD v2
            train_data = self.data_loader.prepare_qa_data('train', task_config.get('max_samples_train'))
            train_dataset = Dataset.from_dict({
                "input_ids": train_data["input_ids"],
                "attention_mask": train_data["attention_mask"],
                "start_positions": train_data["start_positions"],
                "end_positions": train_data["end_positions"]
            })
            
            try:
                eval_data = self.data_loader.prepare_qa_data('validation', task_config.get('max_samples_eval'))
                eval_dataset = Dataset.from_dict({
                    "input_ids": eval_data["input_ids"],
                    "attention_mask": eval_data["attention_mask"],
                    "start_positions": eval_data["start_positions"],
                    "end_positions": eval_data["end_positions"]
                })
            except:
                eval_dataset = train_dataset.select(range(min(500, len(train_dataset))))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def create_compute_metrics_function(self, task_name: str):
        """Create compute_metrics function for evaluation."""
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            return lambda eval_pred: compute_classification_metrics(eval_pred, task_config['metric'])
        elif task_config['type'] == 'question_answering':
            return lambda eval_pred: compute_qa_metrics(eval_pred)
        else:
            return None
    
    def extract_base_model_representations(self, eval_dataset: Dataset, task_name: str):
        """Extract representations from the base (pre-trained) model."""
        logger.info(f"Extracting base model representations for {task_name}")
        
        # Load clean pre-trained model
        model_name = self.config['model']['name']
        
        # Use same model type as training
        task_type = self.config['tasks'][task_name].get('type', 'classification')
        if task_type == 'qa':
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=getattr(torch, self.config['model']['dtype']),
                device_map=self.config['model']['device_map'] if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        else:  # classification tasks
            num_labels = self.config['tasks'][task_name].get('num_labels', 2)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                dtype=getattr(torch, self.config['model']['dtype']),
                device_map=self.config['model']['device_map'] if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        # Create base model representation extractor
        base_extractor = RepresentationExtractor(
            self.representation_config,
            self.output_dir,
            task_name,
            "base_pretrained"
        )
        
        # Set validation examples
        examples = {
            'input_ids': torch.tensor([ex['input_ids'] for ex in eval_dataset]),
            'attention_mask': torch.tensor([ex['attention_mask'] for ex in eval_dataset])
        }
        if 'labels' in eval_dataset[0]:
            examples['labels'] = torch.tensor([ex['labels'] for ex in eval_dataset])
        
        base_extractor.set_validation_examples(examples)
        
        # Extract and save base representations
        representations = base_extractor.extract_representations(base_model, step=0)
        base_extractor.save_representations(representations, step=0)
        
        # Clean up
        del base_model
        torch.cuda.empty_cache()
        
        logger.info("✓ Base model representations extracted")
    
    def create_hyperparameter_sweep_config(self, task_name: str) -> Dict[str, Any]:
        """Create W&B sweep configuration for hyperparameter search."""
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            learning_rates = [1e-5, 2e-5]
            sequence_length = 512
        else:  # question_answering
            learning_rates = [5e-6, 1e-5]
            sequence_length = 768
        
        sweep_config = {
            'method': 'grid',
            'name': f'full_finetune_{task_name}_sweep',
            'metric': {
                'name': 'eval_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'learning_rate': {
                    'values': learning_rates
                },
                'per_device_train_batch_size': {
                    'values': [8, 16]
                },
                'warmup_ratio': {
                    'value': 0.1
                },
                'num_train_epochs': {
                    'value': 3
                },
                'seed': {
                    'values': [42, 1337, 2024]  # Multiple seeds for reproducibility
                }
            }
        }
        
        return sweep_config
    
    def run_single_experiment(self, task_name: str, seed: int = 42, **hyperparams) -> Dict[str, Any]:
        """Run a single full fine-tuning experiment."""
        logger.info(f"Running full fine-tuning experiment: {task_name} (seed: {seed})")
        
        # Check resume status
        resume_info = self.checkpoint_manager.get_resume_info(task_name, "full_finetune", seed)
        
        if resume_info["should_skip"]:
            logger.info(f"✅ Skipping {task_name} (seed {seed}) - already completed")
            return {"status": "skipped", "reason": "already_completed"}
        
        if resume_info["should_resume"]:
            logger.info(f"🔄 Resuming {task_name} (seed {seed}) from checkpoint: {resume_info['checkpoint_path']}")
        else:
            logger.info(f"🆕 Starting fresh {task_name} (seed {seed})")
        
        # Override seed in config
        self.config['reproducibility']['seed'] = seed
        
        # Create run name
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"full_ft_{task_name}_seed{seed}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=os.getenv('WANDB_PROJECT', self.config['wandb']['project']),
            entity=self.config['wandb']['entity'],
            name=run_name,
            group=f"full_finetune_{task_name}",
            job_type="full_finetune",
            tags=["full_finetune", task_name, f"seed_{seed}"],
            config={
                **self.config,
                "task_name": task_name,
                "method": "full_finetune",
                "seed": seed,
                **hyperparams
            }
        )
        
        try:
            # Setup environment with new seed
            self.setup_environment()
            
            # Load model
            model = self.load_model(task_name)
            
            # Prepare datasets
            train_dataset, eval_dataset = self.prepare_datasets(task_name)
            
            # Extract base model representations first (only if enabled)
            if self.config['training'].get('save_final_representations', True):
                self.extract_base_model_representations(eval_dataset, task_name)
            else:
                logger.info("Base model representation extraction disabled to save memory")
            
            # Create representation extractor (disabled to save memory)
            representation_extractor = None
            if self.config['training'].get('save_final_representations', True):
                representation_extractor = RepresentationExtractor(
                    self.representation_config,
                    self.output_dir,
                    task_name,
                    "full_finetune"
                )
            
            # Create monitoring components
            gradient_monitor = GradientStatsMonitor()
            memory_profiler = MemoryProfiler()
            
            # Create training arguments
            output_dir = self.output_dir / f"full_ft_{task_name}_seed{seed}"
            output_dir.mkdir(exist_ok=True)
            
            # Apply hyperparameters
            learning_rate = hyperparams.get('learning_rate', 
                                          self.config['training']['full_finetune_learning_rate'])
            batch_size = hyperparams.get('per_device_train_batch_size',
                                       self.config['training']['per_device_train_batch_size'])
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                run_name=run_name,
                
                # Training configuration
                num_train_epochs=hyperparams.get('num_train_epochs', 3),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                
                # Optimization
                learning_rate=learning_rate,
                weight_decay=self.config['training']['weight_decay'],
                warmup_ratio=hyperparams.get('warmup_ratio', 0.1),
                lr_scheduler_type=self.config['training']['lr_scheduler_type'],
                
                # Evaluation and saving
                eval_strategy="steps",  # Updated parameter name
                eval_steps=100,
                save_strategy="steps",
                save_steps=500,
                logging_steps=50,
                
                # Model selection
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                
                # Performance optimizations
                gradient_checkpointing=True,
                dataloader_pin_memory=True,
                fp16=False,  # Disabled when using bfloat16 dtype to avoid conflicts
                bf16=False,  # Let model dtype handle precision
                
                # Reporting
                report_to=["wandb"],
                logging_dir=str(output_dir / "logs"),
                
                # Reproducibility
                seed=seed,
                data_seed=seed,
            )
            
            # Create compute metrics function
            compute_metrics = self.create_compute_metrics_function(task_name)
            
            # Create task-appropriate data collator
            task_config = self.config['tasks'][task_name]
            if task_config['type'] in ['qa', 'question_answering']:
                # For QA tasks, use custom QA data collator to handle start/end positions
                data_collator = QADataCollator(tokenizer=self.tokenizer, padding=True)
            else:
                # For classification tasks, use padding collator
                data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            
            # Create custom callback (conditional based on representation extraction)
            # Skip callbacks for QA tasks temporarily to debug
            if representation_extractor is not None and task_config['type'] not in ['qa', 'question_answering']:
                custom_callback = FullFinetuneCallback(
                    representation_extractor=representation_extractor,
                    gradient_monitor=gradient_monitor,
                    memory_profiler=memory_profiler,
                    eval_dataset=eval_dataset,
                    extract_every_steps=100
                )
            else:
                # Minimal callback without representation extraction
                custom_callback = None
            
            # Create trainer
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
            if custom_callback is not None:
                callbacks.append(custom_callback)
                
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )
            
            # Log initial memory stats
            initial_memory = memory_profiler.get_memory_stats()
            wandb.log({f"initial_memory/{k}": v for k, v in initial_memory.items()})
            
            # Train the model (with resume support)
            logger.info(f"Starting training for {task_name}...")
            start_time = time.time()
            
            # Mark experiment as started
            self.checkpoint_manager.save_experiment_progress(
                task_name, "full_finetune", seed, "started", str(output_dir)
            )
            
            # Train with checkpoint resume if available
            resume_from_checkpoint = resume_info.get("checkpoint_path") if resume_info["should_resume"] else None
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            training_time = time.time() - start_time
            
            # Evaluate the model
            logger.info(f"Final evaluation for {task_name}...")
            eval_result = trainer.evaluate()
            
            # Save the model
            model_save_path = output_dir / "final_model"
            trainer.save_model(str(model_save_path))
            
            # Mark experiment as completed
            self.checkpoint_manager.save_experiment_progress(
                task_name, "full_finetune", seed, "completed", str(model_save_path)
            )
            
            # Final memory profiling
            final_memory = memory_profiler.get_memory_stats()
            wandb.log({f"final_memory/{k}": v for k, v in final_memory.items()})
            
            # Extract final representations (if enabled)
            if representation_extractor is not None:
                final_representations = representation_extractor.extract_representations(
                    model, trainer.state.global_step
                )
                representation_extractor.save_representations(
                    final_representations, trainer.state.global_step
                )
            
            # Compile results
            results = {
                "task_name": task_name,
                "method": "full_finetune",
                "seed": seed,
                "hyperparameters": hyperparams,
                "train_runtime": training_time,
                "train_loss": train_result.metrics.get("train_loss", 0),
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_metrics": {k: v for k, v in eval_result.items() if k.startswith("eval_")},
                "model_path": str(model_save_path),
                "representation_path": str(representation_extractor.output_dir) if representation_extractor else None,
                "total_steps": trainer.state.global_step,
                "final_learning_rate": trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else learning_rate,
            }
            
            # Log final results
            wandb.log({
                "final_train_loss": results["train_loss"],
                "final_eval_loss": results["eval_loss"],
                "training_time_seconds": training_time,
                "total_steps": trainer.state.global_step
            })
            
            logger.info(f"✓ Completed full fine-tuning: {task_name}")
            logger.info(f"  Train loss: {results['train_loss']:.4f}")
            logger.info(f"  Eval loss: {results['eval_loss']:.4f}")
            logger.info(f"  Training time: {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Full fine-tuning failed: {task_name} - {e}")
            
            # Mark experiment as failed
            self.checkpoint_manager.save_experiment_progress(
                task_name, "full_finetune", seed, "failed"
            )
            
            return {
                "task_name": task_name,
                "method": "full_finetune",
                "seed": seed,
                "error": str(e)
            }
        
        finally:
            # Clean up
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            wandb.finish()
    
    def run_hyperparameter_sweep(self, task_name: str) -> List[Dict[str, Any]]:
        """Run hyperparameter sweep for a task."""
        logger.info(f"Running hyperparameter sweep for {task_name}")
        
        sweep_config = self.create_hyperparameter_sweep_config(task_name)
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=os.getenv('WANDB_PROJECT', self.config['wandb']['project']),
            entity=self.config['wandb']['entity']
        )
        
        results = []
        
        def sweep_function():
            # Get hyperparameters from sweep
            hyperparams = dict(wandb.config)
            seed = hyperparams.pop('seed', 42)
            
            result = self.run_single_experiment(task_name, seed, **hyperparams)
            results.append(result)
        
        # Run sweep agent
        wandb.agent(sweep_id, sweep_function, count=12)  # Grid search: 2x2x3 = 12 runs
        
        return results
    
    def run_validation_demo(self, task_name: str = "sst2", num_samples: int = 100) -> Dict[str, Any]:
        """Run a short validation demo as specified in requirements."""
        logger.info(f"Running validation demo: {task_name} with {num_samples} samples")
        
        # Override config for demo
        original_config = self.config['tasks'][task_name].copy()
        self.config['tasks'][task_name]['max_samples_train'] = num_samples
        self.config['tasks'][task_name]['max_samples_eval'] = min(20, num_samples // 5)
        self.config['training']['num_train_epochs'] = 1
        self.config['training']['eval_steps'] = 50
        self.config['training']['save_steps'] = 50
        self.config['training']['logging_steps'] = 10
        
        try:
            result = self.run_single_experiment(task_name, seed=42)
            logger.info("✓ Validation demo completed successfully")
            return result
        
        finally:
            # Restore original config
            self.config['tasks'][task_name] = original_config


def main():
    """Main function for running full fine-tuning experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full fine-tuning experiments for Llama-2-1.3B")
    parser.add_argument("--task", choices=["mrpc", "squad_v2", "sst2", "rte"], 
                       help="Task to run", required=True)
    parser.add_argument("--mode", choices=["single", "sweep", "demo"], 
                       default="single", help="Experiment mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="shared/config.yaml", help="Config file path")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = FullFinetuneExperiment(args.config)
    
    # Override model to use TinyLlama for actual experiments
    if args.mode != "demo":
        experiment.config['model']['name'] = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    if args.mode == "demo":
        # Run validation demo
        result = experiment.run_validation_demo(args.task)
        print(f"Demo completed: {result}")
    
    elif args.mode == "sweep":
        # Run hyperparameter sweep
        results = experiment.run_hyperparameter_sweep(args.task)
        print(f"Sweep completed with {len(results)} runs")
    
    else:
        # Run single experiment
        hyperparams = {}
        if args.learning_rate:
            hyperparams['learning_rate'] = args.learning_rate
        if args.batch_size:
            hyperparams['per_device_train_batch_size'] = args.batch_size
        
        result = experiment.run_single_experiment(args.task, args.seed, **hyperparams)
        print(f"Experiment completed: {result}")


if __name__ == "__main__":
    main()
