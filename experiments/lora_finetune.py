#!/usr/bin/env python3
"""LoRA (Low-Rank Adaptation) fine-tuning experiments for Llama-2-1.3B with comprehensive tracking."""

import os
import sys
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
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
from datasets import Dataset
import evaluate

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.data_preparation import TaskDataLoader
from shared.metrics import compute_classification_metrics, compute_qa_metrics
from models.trainer_utils import (
    ParameterEfficiencyTracker, 
    LoRAAnalyzer, 
    ModelMerger, 
    TrainingEfficiencyMonitor
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoRAExperimentConfig:
    """Configuration for LoRA experiments."""
    # Core LoRA settings (fixed as per requirements)
    rank: int = 8
    alpha: int = 16  # Scaling factor alpha/r = 2
    dropout: float = 0.05
    
    # Target module variations
    target_modules_standard: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    target_modules_extended: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    target_modules_all_linear: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Hyperparameter search ranges
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 3e-4])
    warmup_ratio: float = 0.06  # 6% warmup as specified
    
    # Ablation study settings
    rank_ablation: List[int] = field(default_factory=lambda: [4, 8, 16])
    alpha_ablation: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    # Representation tracking
    extract_every_steps: int = 100
    save_adapter_weights: bool = True
    analyze_rank_utilization: bool = True
    
    # Validation settings
    merge_test_enabled: bool = True
    equivalence_threshold: float = 1e-5


class LoRARepresentationExtractor:
    """Extended representation extractor for LoRA-specific analysis."""
    
    def __init__(self, config: LoRAExperimentConfig, output_dir: Path, task_name: str, method: str):
        self.config = config
        self.output_dir = output_dir / "representations" / f"{method}_{task_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.method = method
        self.step_counter = 0
        
        # Storage for representations
        self.representations = {}
        self.validation_examples = None
        
        logger.info(f"Initialized LoRA representation extractor: {self.output_dir}")
    
    def set_validation_examples(self, examples: Dict[str, torch.Tensor]):
        """Set validation examples for consistent representation extraction."""
        max_samples = min(1000, len(examples['input_ids']))  # Limit for memory
        
        self.validation_examples = {
            'input_ids': examples['input_ids'][:max_samples],
            'attention_mask': examples['attention_mask'][:max_samples],
        }
        if 'labels' in examples:
            self.validation_examples['labels'] = examples['labels'][:max_samples]
        
        logger.info(f"Set {max_samples} validation examples for LoRA representation extraction")
    
    def extract_lora_representations(self, model: torch.nn.Module, step: int) -> Dict[str, torch.Tensor]:
        """Extract representations from LoRA model including adapter-specific info."""
        if self.validation_examples is None:
            logger.warning("No validation examples set for representation extraction")
            return {}
        
        model.eval()
        representations = {}
        
        # Extract standard representations
        layer_outputs = {}
        hooks = []
        
        def create_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_name] = output[0].detach().cpu()
                else:
                    layer_outputs[layer_name] = output.detach().cpu()
            return hook
        
        try:
            # Get base model
            if hasattr(model, 'base_model'):
                base_model = model.base_model.model  # PEFT model structure
            elif hasattr(model, 'model'):
                base_model = model.model
            else:
                base_model = model
            
            # Hook transformer layers
            if hasattr(base_model, 'layers'):
                for i in range(min(len(base_model.layers), 24)):  # Llama-2-1.3B has 24 layers
                    layer = base_model.layers[i]
                    hook = layer.register_forward_hook(create_hook(f'layer_{i}'))
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                input_ids = self.validation_examples['input_ids'].to(model.device)
                attention_mask = self.validation_examples['attention_mask'].to(model.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Store layer representations
                for layer_name, layer_output in layer_outputs.items():
                    representations[layer_name] = layer_output
                
                # Store final hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    representations['final_hidden_states'] = outputs.hidden_states[-1].detach().cpu()
                elif hasattr(outputs, 'last_hidden_state'):
                    representations['final_hidden_states'] = outputs.last_hidden_state.detach().cpu()
            
            # Extract LoRA-specific representations
            if self.config.save_adapter_weights and hasattr(model, 'peft_config'):
                lora_analyzer = LoRAAnalyzer(model)
                
                # Save adapter statistics
                adapter_stats = lora_analyzer.compute_adapter_statistics()
                representations['adapter_statistics'] = torch.tensor(list(adapter_stats.values()))
                
                # Save rank utilization
                if self.config.analyze_rank_utilization:
                    rank_stats = lora_analyzer.analyze_rank_utilization()
                    representations['rank_utilization'] = torch.tensor(list(rank_stats.values()))
                
                # Save adapter weights themselves
                adapter_weights = {}
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        adapter_weights[f"{name}_lora_A"] = module.lora_A.weight.detach().cpu()
                        adapter_weights[f"{name}_lora_B"] = module.lora_B.weight.detach().cpu()
                
                representations.update(adapter_weights)
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
            model.train()
        
        return representations
    
    def save_representations(self, representations: Dict[str, torch.Tensor], step: int):
        """Save representations to disk with LoRA-specific metadata."""
        if not representations:
            return
        
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)
        
        # Separate adapter weights from layer representations
        adapter_weights = {}
        layer_representations = {}
        
        for name, tensor in representations.items():
            if 'lora_A' in name or 'lora_B' in name:
                adapter_weights[name] = tensor
            else:
                layer_representations[name] = tensor
        
        # Save layer representations
        for layer_name, tensor in layer_representations.items():
            file_path = step_dir / f"{layer_name}.pt"
            torch.save(tensor, file_path)
        
        # Save adapter weights separately
        if adapter_weights:
            adapter_dir = step_dir / "adapter_weights"
            adapter_dir.mkdir(exist_ok=True)
            for name, tensor in adapter_weights.items():
                file_path = adapter_dir / f"{name}.pt"
                torch.save(tensor, file_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'task_name': self.task_name,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.validation_examples['input_ids']),
            'layer_names': list(layer_representations.keys()),
            'adapter_names': list(adapter_weights.keys()),
            'tensor_shapes': {name: list(tensor.shape) for name, tensor in representations.items()},
            'lora_config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        with open(step_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved LoRA representations for step {step} to {step_dir}")


class LoRACallback(TrainerCallback):
    """Custom callback for LoRA training with comprehensive tracking."""
    
    def __init__(self, 
                 representation_extractor: LoRARepresentationExtractor,
                 parameter_tracker: ParameterEfficiencyTracker,
                 eval_dataset: Dataset,
                 config: LoRAExperimentConfig,
                 extract_every_steps: int = 100):
        self.representation_extractor = representation_extractor
        self.parameter_tracker = parameter_tracker
        self.eval_dataset = eval_dataset
        self.config = config
        self.extract_every_steps = extract_every_steps
        self.efficiency_monitor = TrainingEfficiencyMonitor()
        
        # Set validation examples for representation extraction
        if eval_dataset is not None:
            examples = {
                'input_ids': torch.tensor([ex['input_ids'] for ex in eval_dataset]),
                'attention_mask': torch.tensor([ex['attention_mask'] for ex in eval_dataset])
            }
            if 'labels' in eval_dataset[0]:
                examples['labels'] = torch.tensor([ex['labels'] for ex in eval_dataset])
            
            self.representation_extractor.set_validation_examples(examples)
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each training step."""
        self.efficiency_monitor.start_timing()
        self.efficiency_monitor.record_memory_usage()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        model = kwargs.get('model')
        step = state.global_step
        
        # End timing
        self.efficiency_monitor.end_timing()
        
        # Log LoRA-specific metrics
        if hasattr(model, 'peft_config'):
            lora_analyzer = LoRAAnalyzer(model)
            
            # Adapter statistics
            adapter_stats = lora_analyzer.compute_adapter_statistics()
            if adapter_stats and wandb.run is not None:
                wandb.log({f"lora_adapters/{k}": v for k, v in adapter_stats.items()}, step=step)
            
            # Rank utilization analysis
            if step % (self.extract_every_steps // 2) == 0:  # Less frequent due to computational cost
                rank_stats = lora_analyzer.analyze_rank_utilization()
                if rank_stats and wandb.run is not None:
                    wandb.log({f"lora_rank/{k}": v for k, v in rank_stats.items()}, step=step)
        
        # Parameter efficiency metrics
        if step % 50 == 0:
            efficiency_metrics = self.parameter_tracker.get_efficiency_metrics()
            if wandb.run is not None:
                wandb.log({f"efficiency/{k}": v for k, v in efficiency_metrics.items()}, step=step)
        
        # Training efficiency metrics
        if step % 100 == 0:
            training_metrics = self.efficiency_monitor.get_efficiency_metrics()
            if training_metrics and wandb.run is not None:
                wandb.log({f"training_efficiency/{k}": v for k, v in training_metrics.items()}, step=step)
        
        # Extract representations
        if step % self.extract_every_steps == 0:
            logger.info(f"Extracting LoRA representations at step {step}")
            representations = self.representation_extractor.extract_lora_representations(model, step)
            self.representation_extractor.save_representations(representations, step)
        
        # Verify base model parameters are frozen
        if step % 500 == 0:
            self._verify_base_model_frozen(model, step)
    
    def _verify_base_model_frozen(self, model: torch.nn.Module, step: int):
        """Verify that base model parameters are frozen (critical validation)."""
        base_params_with_grad = 0
        total_base_params = 0
        lora_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if 'lora_' in name:
                if param.requires_grad:
                    lora_params_with_grad += 1
            else:
                total_base_params += 1
                if param.requires_grad:
                    base_params_with_grad += 1
        
        # Log verification results
        verification_metrics = {
            'base_params_frozen': base_params_with_grad == 0,
            'base_params_with_grad': base_params_with_grad,
            'total_base_params': total_base_params,
            'lora_params_with_grad': lora_params_with_grad
        }
        
        if wandb.run is not None:
            wandb.log({f"verification/{k}": v for k, v in verification_metrics.items()}, step=step)
        
        if base_params_with_grad > 0:
            logger.warning(f"⚠️  Step {step}: {base_params_with_grad} base parameters have gradients! Base model may not be properly frozen.")
        else:
            logger.debug(f"✓ Step {step}: Base model properly frozen, {lora_params_with_grad} LoRA parameters trainable")


class LoRAExperiment:
    """Main experiment runner for LoRA fine-tuning with comprehensive analysis."""
    
    def __init__(self, config_path: str = "shared/config.yaml"):
        """Initialize the LoRA experiment."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.data_loader = None
        self.lora_config = LoRAExperimentConfig()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"results/lora_finetune_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized LoRA experiment")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_environment(self):
        """Setup the experimental environment."""
        logger.info("Setting up LoRA experimental environment...")
        
        # Set random seeds
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
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
        
        logger.info("✓ LoRA environment setup complete")
    
    def create_lora_config(self, target_modules: List[str], rank: int = None, alpha: int = None) -> LoraConfig:
        """Create LoRA configuration."""
        rank = rank or self.lora_config.rank
        alpha = alpha or self.lora_config.alpha
        
        # Determine task type
        task_type = TaskType.CAUSAL_LM
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_config.dropout,
            bias="none",
            task_type=task_type,
            inference_mode=False,
        )
        
        logger.info(f"Created LoRA config: rank={rank}, alpha={alpha}, modules={target_modules}")
        return lora_config
    
    def load_lora_model(self, target_modules: List[str], rank: int = None, alpha: int = None) -> torch.nn.Module:
        """Load model with LoRA adaptation."""
        logger.info("Loading model with LoRA adaptation...")
        
        model_name = self.config['model']['name']
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=getattr(torch, self.config['model']['dtype']),
            device_map=self.config['model']['device_map'] if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Create LoRA config
        lora_config = self.create_lora_config(target_modules, rank, alpha)
        
        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        
        # Verify LoRA setup
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.6f}")
        
        # Verify we're hitting the ~0.3% target
        expected_ratio = 0.003
        if abs(trainable_params/total_params - expected_ratio) > 0.002:
            logger.warning(f"⚠️  Trainable ratio {trainable_params/total_params:.6f} differs significantly from expected ~{expected_ratio:.3f}")
        
        return model
    
    def prepare_datasets(self, task_name: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets (same as full fine-tuning)."""
        logger.info(f"Preparing datasets for LoRA: {task_name}")
        
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
    
    def test_lora_merge_equivalence(self, model: torch.nn.Module, eval_dataset: Dataset) -> Dict[str, float]:
        """Test LoRA merge equivalence as required in validation."""
        logger.info("Testing LoRA merge equivalence...")
        
        # Create test input
        test_examples = eval_dataset.select(range(min(10, len(eval_dataset))))
        test_input = {
            'input_ids': torch.tensor([ex['input_ids'] for ex in test_examples]).to(model.device),
            'attention_mask': torch.tensor([ex['attention_mask'] for ex in test_examples]).to(model.device)
        }
        
        # Test merge equivalence
        try:
            # Create merged model
            merged_model = ModelMerger.merge_lora_weights(model)
            
            # Test equivalence
            equivalence_results = ModelMerger.test_merge_equivalence(
                base_model=None,  # We don't need base model for this test
                adapter_model=model,
                merged_model=merged_model,
                test_input=test_input['input_ids']
            )
            
            logger.info(f"LoRA merge equivalence test results: {equivalence_results}")
            return equivalence_results
            
        except Exception as e:
            logger.error(f"LoRA merge equivalence test failed: {e}")
            return {'error': str(e)}
    
    def run_ablation_study(self, task_name: str, study_type: str, seed: int = 42) -> List[Dict[str, Any]]:
        """Run ablation studies for different LoRA configurations."""
        logger.info(f"Running LoRA ablation study: {study_type} on {task_name}")
        
        results = []
        
        if study_type == "rank":
            # Test different rank values
            for rank in self.lora_config.rank_ablation:
                logger.info(f"Testing rank={rank}")
                result = self.run_single_experiment(
                    task_name=task_name,
                    seed=seed,
                    target_modules=self.lora_config.target_modules_standard,
                    rank=rank,
                    alpha=rank * 2,  # Keep alpha/rank ratio = 2
                    experiment_type=f"ablation_rank_{rank}"
                )
                results.append(result)
        
        elif study_type == "alpha":
            # Test different alpha values (fixed rank=8)
            for alpha in self.lora_config.alpha_ablation:
                logger.info(f"Testing alpha={alpha}")
                result = self.run_single_experiment(
                    task_name=task_name,
                    seed=seed,
                    target_modules=self.lora_config.target_modules_standard,
                    rank=8,
                    alpha=alpha,
                    experiment_type=f"ablation_alpha_{alpha}"
                )
                results.append(result)
        
        elif study_type == "modules":
            # Test different target module configurations
            module_configs = [
                ("standard", self.lora_config.target_modules_standard),
                ("extended", self.lora_config.target_modules_extended),
                ("all_linear", self.lora_config.target_modules_all_linear)
            ]
            
            for config_name, modules in module_configs:
                logger.info(f"Testing modules={config_name}")
                result = self.run_single_experiment(
                    task_name=task_name,
                    seed=seed,
                    target_modules=modules,
                    rank=8,
                    alpha=16,
                    experiment_type=f"ablation_modules_{config_name}"
                )
                results.append(result)
        
        return results
    
    def run_single_experiment(self, 
                            task_name: str, 
                            seed: int = 42,
                            target_modules: List[str] = None,
                            rank: int = None,
                            alpha: int = None,
                            experiment_type: str = "standard",
                            **hyperparams) -> Dict[str, Any]:
        """Run a single LoRA experiment."""
        target_modules = target_modules or self.lora_config.target_modules_standard
        rank = rank or self.lora_config.rank
        alpha = alpha or self.lora_config.alpha
        
        logger.info(f"Running LoRA experiment: {task_name} (seed: {seed}, type: {experiment_type})")
        
        # Override seed in config
        self.config['reproducibility']['seed'] = seed
        
        # Create run name
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"lora_{task_name}_{experiment_type}_seed{seed}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=run_name,
            group=f"lora_{task_name}",
            job_type="lora_finetune",
            tags=["lora", task_name, f"seed_{seed}", experiment_type],
            config={
                **self.config,
                "task_name": task_name,
                "method": "lora",
                "seed": seed,
                "lora_rank": rank,
                "lora_alpha": alpha,
                "lora_target_modules": target_modules,
                "experiment_type": experiment_type,
                **hyperparams
            }
        )
        
        try:
            # Setup environment with new seed
            self.setup_environment()
            
            # Load LoRA model
            model = self.load_lora_model(target_modules, rank, alpha)
            
            # Create parameter tracker
            parameter_tracker = ParameterEfficiencyTracker(model, "lora")
            parameter_tracker.log_parameter_info()
            
            # Prepare datasets
            train_dataset, eval_dataset = self.prepare_datasets(task_name)
            
            # Create representation extractor
            representation_extractor = LoRARepresentationExtractor(
                self.lora_config,
                self.output_dir,
                task_name,
                f"lora_{experiment_type}"
            )
            
            # Create training arguments
            output_dir = self.output_dir / f"lora_{task_name}_{experiment_type}_seed{seed}"
            output_dir.mkdir(exist_ok=True)
            
            # Apply hyperparameters (LoRA-specific learning rates)
            learning_rate = hyperparams.get('learning_rate', self.lora_config.learning_rates[0])
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
                
                # Optimization (LoRA-specific)
                learning_rate=learning_rate,
                weight_decay=self.config['training']['weight_decay'],
                warmup_ratio=self.lora_config.warmup_ratio,  # 6% as specified
                lr_scheduler_type=self.config['training']['lr_scheduler_type'],
                
                # Evaluation and saving
                evaluation_strategy="steps",
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
                fp16=torch.cuda.is_available(),
                
                # Reporting
                report_to=["wandb"],
                logging_dir=str(output_dir / "logs"),
                
                # Reproducibility
                seed=seed,
                data_seed=seed,
            )
            
            # Create compute metrics function
            compute_metrics = self.create_compute_metrics_function(task_name)
            
            # Create data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            
            # Create custom callback
            custom_callback = LoRACallback(
                representation_extractor=representation_extractor,
                parameter_tracker=parameter_tracker,
                eval_dataset=eval_dataset,
                config=self.lora_config,
                extract_every_steps=100
            )
            
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
                    EarlyStoppingCallback(early_stopping_patience=3),
                    custom_callback
                ]
            )
            
            # Log initial efficiency metrics
            efficiency_metrics = parameter_tracker.get_efficiency_metrics()
            wandb.log({f"initial_efficiency/{k}": v for k, v in efficiency_metrics.items()})
            
            # Train the model
            logger.info(f"Starting LoRA training for {task_name}...")
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate the model
            logger.info(f"Final evaluation for LoRA {task_name}...")
            eval_result = trainer.evaluate()
            
            # Test LoRA merge equivalence
            merge_results = {}
            if self.lora_config.merge_test_enabled:
                merge_results = self.test_lora_merge_equivalence(model, eval_dataset)
            
            # Save the LoRA adapter
            adapter_save_path = output_dir / "final_adapter"
            model.save_pretrained(str(adapter_save_path))
            
            # Extract final representations
            final_representations = representation_extractor.extract_lora_representations(
                model, trainer.state.global_step
            )
            representation_extractor.save_representations(
                final_representations, trainer.state.global_step
            )
            
            # Final efficiency analysis
            final_efficiency = parameter_tracker.get_efficiency_metrics()
            
            # Compile results
            results = {
                "task_name": task_name,
                "method": "lora",
                "experiment_type": experiment_type,
                "seed": seed,
                "lora_config": {
                    "rank": rank,
                    "alpha": alpha,
                    "target_modules": target_modules,
                    "dropout": self.lora_config.dropout
                },
                "hyperparameters": hyperparams,
                "train_runtime": training_time,
                "train_loss": train_result.metrics.get("train_loss", 0),
                "eval_loss": eval_result.get("eval_loss", 0),
                "eval_metrics": {k: v for k, v in eval_result.items() if k.startswith("eval_")},
                "efficiency_metrics": final_efficiency,
                "merge_test_results": merge_results,
                "adapter_path": str(adapter_save_path),
                "representation_path": str(representation_extractor.output_dir),
                "total_steps": trainer.state.global_step,
                "final_learning_rate": trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else learning_rate,
            }
            
            # Log final results
            wandb.log({
                "final_train_loss": results["train_loss"],
                "final_eval_loss": results["eval_loss"],
                "training_time_seconds": training_time,
                "total_steps": trainer.state.global_step,
                **{f"final_efficiency/{k}": v for k, v in final_efficiency.items()},
                **{f"merge_test/{k}": v for k, v in merge_results.items() if isinstance(v, (int, float, bool))}
            })
            
            logger.info(f"✓ Completed LoRA experiment: {task_name} ({experiment_type})")
            logger.info(f"  Train loss: {results['train_loss']:.4f}")
            logger.info(f"  Eval loss: {results['eval_loss']:.4f}")
            logger.info(f"  Training time: {training_time:.2f}s")
            logger.info(f"  Trainable params: {final_efficiency['trainable_parameters']:,} ({final_efficiency['trainable_parameter_ratio']:.6f})")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ LoRA experiment failed: {task_name} ({experiment_type}) - {e}")
            return {
                "task_name": task_name,
                "method": "lora",
                "experiment_type": experiment_type,
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
        """Run hyperparameter sweep for LoRA as specified in requirements."""
        logger.info(f"Running LoRA hyperparameter sweep for {task_name}")
        
        results = []
        
        # Grid search over learning rates and seeds
        for learning_rate in self.lora_config.learning_rates:
            for seed in [42, 1337, 2024]:  # Multiple seeds for reproducibility
                logger.info(f"Running LoRA sweep: LR={learning_rate}, seed={seed}")
                
                result = self.run_single_experiment(
                    task_name=task_name,
                    seed=seed,
                    target_modules=self.lora_config.target_modules_standard,
                    learning_rate=learning_rate,
                    experiment_type=f"sweep_lr{learning_rate}_seed{seed}"
                )
                results.append(result)
        
        return results
    


def main():
    """Main function for running LoRA experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA fine-tuning experiments for Llama-2-1.3B")
    parser.add_argument("--task", choices=["mrpc", "squad_v2", "sst2", "rte"], 
                       help="Task to run", required=True)
    parser.add_argument("--mode", choices=["single", "sweep", "ablation"], 
                       default="single", help="Experiment mode")
    parser.add_argument("--ablation-type", choices=["rank", "alpha", "modules"],
                       help="Type of ablation study")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="shared/config.yaml", help="Config file path")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--rank", type=int, help="Override LoRA rank")
    parser.add_argument("--alpha", type=int, help="Override LoRA alpha")
    parser.add_argument("--target-modules", nargs="+", help="Override target modules")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = LoRAExperiment(args.config)
    
    # Ensure model is set to TinyLlama for all experiments  
    experiment.config['model']['name'] = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    if args.mode == "ablation":
        if not args.ablation_type:
            print("Error: --ablation-type required for ablation mode")
            return
        # Run ablation study
        results = experiment.run_ablation_study(args.task, args.ablation_type, args.seed)
        print(f"LoRA ablation study completed with {len(results)} runs")
    
    elif args.mode == "sweep":
        # Run hyperparameter sweep
        results = experiment.run_hyperparameter_sweep(args.task)
        print(f"LoRA sweep completed with {len(results)} runs")
    
    else:
        # Run single experiment
        hyperparams = {}
        if args.learning_rate:
            hyperparams['learning_rate'] = args.learning_rate
        
        # LoRA-specific overrides
        target_modules = args.target_modules or experiment.lora_config.target_modules_standard
        rank = args.rank or experiment.lora_config.rank
        alpha = args.alpha or experiment.lora_config.alpha
        
        result = experiment.run_single_experiment(
            args.task, args.seed, target_modules, rank, alpha, "manual", **hyperparams
        )
        print(f"LoRA experiment completed: {result}")


if __name__ == "__main__":
    main()
