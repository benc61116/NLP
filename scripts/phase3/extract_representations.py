#!/usr/bin/env python3
"""
Phase 3: Post-Training Representation Extraction

This script loads saved models from Phase 2 and extracts representations
for drift analysis. This approach enables:
1. Higher batch sizes during training (Phases 1-2)
2. Memory-efficient layer-by-layer extraction
3. Methodologically sound (representations are model properties, not training artifacts)
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List
import yaml
import argparse
from datasets import Dataset

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.full_finetune import RepresentationExtractor, RepresentationConfig, FullFinetuneExperiment
from experiments.lora_finetune import LoRARepresentationExtractor, LoRAExperiment
from shared.data_preparation import TaskDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_optimal_config(task: str, method: str) -> Dict:
    """Load optimal hyperparameters from Phase 1."""
    config_path = Path(f"analysis/{task}_{method}_optimal.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Optimal config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_saved_model(task: str, method: str, seed: int) -> Path:
    """Find the saved model from Phase 2 - with WandB fallback."""
    results_dir = Path("results")
    
    logger.info(f"Searching for model: {task}/{method}/seed{seed}")
    
    # FIXED: Multiple search patterns for robust model detection
    search_patterns = [
        # Pattern 1: Downloaded models from WandB
        f"downloaded_models/*{task}*seed{seed}*",
        # Pattern 2: Direct phase2 results
        f"phase2/*{task}*{method}*seed{seed}*",
        # Pattern 3: Experiment timestamp directories  
        f"{method}_*/*{task}*seed{seed}*/final_model",
        f"{method}_*/*{task}*seed{seed}*",
        # Pattern 4: LoRA adapter directories
        f"{method}_*/*{task}*seed{seed}*/final_adapter",
        # Pattern 5: Alternative naming patterns
        f"*{method}*{task}*{seed}*/final_model",
        f"*{method}*{task}*{seed}*/final_adapter",
    ]
    
    found_models = []
    
    for pattern in search_patterns:
        matches = list(results_dir.glob(pattern))
        for match in matches:
            if match.exists():
                # Verify this is actually a model directory
                if (match.is_dir() and 
                    any((match / f).exists() for f in ["config.json", "pytorch_model.bin", "model.safetensors", "adapter_config.json"])):
                    found_models.append((pattern, match))
                    logger.info(f"Found model with pattern '{pattern}': {match}")
    
    if not found_models:
        # Try downloading from WandB
        logger.warning(f"Model not found locally for {task}/{method}/seed{seed}")
        logger.info("Attempting to download from WandB...")
        
        try:
            import subprocess
            
            artifact_name = f"full_finetune_model_{task}_seed{seed}" if method == "full_finetune" else f"lora_adapter_{task}_seed{seed}"
            
            download_cmd = [
                'python', 'scripts/download_wandb_models.py',
                '--entity', 'galavny-tel-aviv-university',
                '--project', 'NLP-Phase2',
                '--artifact', artifact_name,
                '--output-dir', 'results/downloaded_models'
            ]
            
            result = subprocess.run(download_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Model downloaded from WandB, retrying search...")
                # Retry search in downloaded_models
                downloaded_path = results_dir / "downloaded_models" / artifact_name
                if downloaded_path.exists():
                    return downloaded_path
            else:
                logger.error(f"Failed to download from WandB: {result.stderr}")
        
        except Exception as e:
            logger.error(f"WandB download attempt failed: {e}")
        
        # List available directories for debugging
        available_dirs = [d.name for d in results_dir.glob("*") if d.is_dir()]
        logger.error(f"Available directories: {available_dirs}")
        raise FileNotFoundError(f"No saved model found for {task}/{method}/seed{seed} (local or WandB)")
    
    # Return the first valid match
    best_match = found_models[0][1]
    logger.info(f"Selected model path: {best_match}")
    return best_match


def extract_representations_from_model(
    task: str,
    method: str,
    seed: int,
    model_path: Path,
    output_dir: Path
):
    """Extract representations from a saved model."""
    logger.info(f"=" * 80)
    logger.info(f"Extracting representations: {task}/{method}/seed{seed}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"=" * 80)
    
    # Initialize WandB for this extraction
    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "NLP-Phase3-Representations"),
        entity=os.environ.get("WANDB_ENTITY", "galavny-tel-aviv-university"),
        name=f"extract_{method}_{task}_seed{seed}",
        job_type="representation_extraction",
        config={
            "task": task,
            "method": method,
            "seed": seed,
            "phase": "phase3"
        },
        settings=wandb.Settings(start_method="thread")  # Fix deprecation warning
    )
    
    # Load optimal config
    optimal_config = load_optimal_config(task, method)
    hyperparams = optimal_config['best_hyperparameters']
    
    # Initialize experiment (to get config and data loaders)
    if method == "full_finetune":
        experiment = FullFinetuneExperiment(config_path="shared/config.yaml")
    else:  # lora
        experiment = LoRAExperiment(config_path="shared/config.yaml")
    
    # FIXED: Correct data loader initialization with Phase 2 consistency
    model_name = experiment.config['model']['name']
    max_length = experiment.config['model']['max_length']
    
    # CRITICAL: Ensure max_length matches Phase 2 training configuration
    if max_length != 384:
        logger.warning(f"max_length={max_length} differs from Phase 2 standard (384)")
        logger.info("Using Phase 2 consistent max_length=384 for methodological rigor")
        max_length = 384
    
    data_loader = TaskDataLoader(model_name=model_name, max_length=max_length)
    logger.info(f"Using max_length={max_length} for representation extraction (Phase 2 consistent)")
    
    # FIXED: Load task-specific data using correct methods
    if task == "squad_v2":
        eval_data = data_loader.prepare_qa_data('validation', 750)  # 750 samples for consistency
        eval_dataset = Dataset.from_dict({
            "input_ids": eval_data["input_ids"],
            "attention_mask": eval_data["attention_mask"],
            "start_positions": eval_data["start_positions"],
            "end_positions": eval_data["end_positions"], 
            "answerability_labels": eval_data["answerability_labels"]
        })
    else:
        eval_data = data_loader.prepare_classification_data(task, 'validation', 750)
        eval_dataset = Dataset.from_dict({
            "input_ids": eval_data["input_ids"],
            "attention_mask": eval_data["attention_mask"],
            "labels": eval_data["labels"]
        })
    
    logger.info(f"Loaded {len(eval_dataset)} validation samples for {task}")
    
    # Initialize representation extractor with correct config parameters
    rep_config = RepresentationConfig(
        extract_every_steps=100,          # Not used in post-training extraction
        save_layers=list(range(24)),      # All transformer layers
        max_validation_samples=750,       # Standard for analysis
        save_attention=False,             # Optional: can enable if needed
        save_mlp=True,                    # Save MLP representations
        memory_map=True                   # Use memory mapping for efficiency
    )
    
    if method == "full_finetune":
        extractor = RepresentationExtractor(
            config=rep_config,
            output_dir=output_dir,
            task_name=task,
            method=f"{method}_seed{seed}"
        )
    else:
        # LoRARepresentationExtractor needs the full experiment config, not RepresentationConfig
        # Add required attributes to rep_config for LoRA extraction
        rep_config.save_adapter_weights = False  # Don't save adapter weights during extraction
        rep_config.analyze_rank_utilization = False  # Don't analyze rank during extraction
        
        extractor = LoRARepresentationExtractor(
            config=rep_config,
            output_dir=output_dir,
            task_name=task,
            method=f"{method}_seed{seed}"
        )
    
    # Set validation examples (convert to proper format)
    num_samples = min(len(eval_dataset), 750)
    eval_examples = {
        'input_ids': torch.tensor(eval_dataset['input_ids'][:num_samples]),
        'attention_mask': torch.tensor(eval_dataset['attention_mask'][:num_samples])
    }
    
    # Add task-specific labels
    if task == "squad_v2":
        eval_examples['start_positions'] = torch.tensor(eval_dataset['start_positions'][:num_samples])
        eval_examples['end_positions'] = torch.tensor(eval_dataset['end_positions'][:num_samples])
        eval_examples['answerability_labels'] = torch.tensor(eval_dataset['answerability_labels'][:num_samples])
    else:
        eval_examples['labels'] = torch.tensor(eval_dataset['labels'][:num_samples])
    
    extractor.set_validation_examples(eval_examples)
    
    # FIXED: Complete model loading implementation
    logger.info(f"Loading model from {model_path}...")
    
    # Determine model loading strategy based on task and method
    target_dtype = getattr(torch, experiment.config['model']['dtype'])
    
    try:
        if task == "squad_v2":
            # SQuAD v2 with answerability head
            from models.squad_v2_qa_model import SquadV2QuestionAnsweringModel
            if method == "lora":
                # LoRA model: Load adapter on top of base model
                from peft import PeftModel
                from transformers import AutoTokenizer
                
                # Load tokenizer and setup padding (same as training)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                base_model = SquadV2QuestionAnsweringModel(
                    model_name=model_name,
                    dtype=experiment.config['model']['dtype']
                )
                # Set pad_token_id in model config
                base_model.config.pad_token_id = tokenizer.pad_token_id
                
                model = PeftModel.from_pretrained(base_model, str(model_path))
                # Ensure pad_token_id persists after PEFT loading
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                # Full fine-tuned model
                model = SquadV2QuestionAnsweringModel.from_pretrained(str(model_path))
        else:
            # Classification tasks
            from transformers import AutoModelForSequenceClassification
            task_config = experiment.config['tasks'][task]
            num_labels = task_config.get('num_labels', 2)
            
            if method == "lora":
                # LoRA model: Load adapter on top of base model  
                from peft import PeftModel
                from transformers import AutoTokenizer
                
                # Load tokenizer and setup padding (same as training)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    torch_dtype=target_dtype
                )
                # Set pad_token_id before loading adapter
                base_model.config.pad_token_id = tokenizer.pad_token_id
                
                model = PeftModel.from_pretrained(base_model, str(model_path))
                # Ensure pad_token_id persists after PEFT loading
                model.config.pad_token_id = tokenizer.pad_token_id
            else:
                # Full fine-tuned model
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path),
                    torch_dtype=target_dtype
                )
        
        # Move to GPU and set to eval mode
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        logger.info(f"✅ Model loaded successfully: {type(model).__name__}")
        logger.info(f"   Method: {method}, Task: {task}, Dtype: {target_dtype}")
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.error(f"Method: {method}, Task: {task}")
        raise
    
    # Extract representations
    logger.info("Extracting representations...")
    with torch.no_grad():
        if method == "lora":
            representations = extractor.extract_lora_representations(model, step=0)
        else:
            representations = extractor.extract_representations(model, step=0)
    
    # Save representations
    extractor.save_representations(representations, step=0)
    
    logger.info(f"✅ Representations extracted and saved to {output_dir}")
    
    # Upload to WandB as artifact for safe storage
    try:
        import wandb
        if wandb.run is not None:
            logger.info(f"📦 Uploading representations to WandB as artifact...")
            artifact = wandb.Artifact(
                name=f"representations_{method}_{task}_seed{seed}",
                type="representations",
                description=f"Extracted representations for {task}/{method}/seed{seed}",
                metadata={
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "phase": "phase3"
                }
            )
            artifact.add_dir(str(output_dir))
            wandb.log_artifact(artifact)
            logger.info(f"✅ Representations uploaded to WandB: {artifact.name}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to upload representations to WandB: {e}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    # Finish WandB run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Extract representations from saved Phase 2 models")
    parser.add_argument("--task", required=True, choices=["mrpc", "sst2", "rte", "squad_v2"])
    parser.add_argument("--method", required=True, choices=["full_finetune", "lora"])
    parser.add_argument("--seed", type=int, required=True, help="Training seed (42, 1337, or 2024)")
    parser.add_argument("--model-path", type=str, help="Path to saved model (auto-detected if not provided)")
    parser.add_argument("--output-dir", type=str, default="results/phase3_representations")
    
    args = parser.parse_args()
    
    # Find model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        logger.info("Auto-detecting saved model path...")
        model_path = find_saved_model(args.task, args.method, args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract representations
    extract_representations_from_model(
        task=args.task,
        method=args.method,
        seed=args.seed,
        model_path=model_path,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()

