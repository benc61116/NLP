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
    """Find the saved model from Phase 2."""
    # Pattern: results/{method}_{timestamp}/final_model
    results_dir = Path("results")
    
    # Search for matching model directory
    pattern = f"{method}_*_{task}_seed{seed}"
    matching_dirs = list(results_dir.glob(f"{method}_*"))
    
    for model_dir in matching_dirs:
        final_model_path = model_dir / "final_model"
        if final_model_path.exists():
            # Check if this is the right task/seed (from metadata or naming)
            logger.info(f"Found candidate model: {final_model_path}")
            return final_model_path
    
    raise FileNotFoundError(f"No saved model found for {task}/{method}/seed{seed}")


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
    
    # Load optimal config
    optimal_config = load_optimal_config(task, method)
    hyperparams = optimal_config['best_hyperparameters']
    
    # Initialize experiment (to get config and data loaders)
    if method == "full_finetune":
        experiment = FullFinetuneExperiment(config_path="shared/config.yaml")
    else:  # lora
        experiment = LoRAExperiment(config_path="shared/config.yaml")
    
    # Load datasets
    data_loader = TaskDataLoader(task, experiment.config)
    train_dataset, eval_dataset = data_loader.load_data()
    
    logger.info(f"Loaded {len(eval_dataset)} validation samples for {task}")
    
    # Initialize representation extractor
    rep_config = RepresentationConfig(
        extract_layers=list(range(24)),  # All transformer layers
        max_validation_samples=750,      # Standard for analysis
        save_attention_weights=False     # Optional: can enable if needed
    )
    
    if method == "full_finetune":
        extractor = RepresentationExtractor(
            config=rep_config,
            output_dir=output_dir,
            task_name=task,
            method=f"{method}_seed{seed}"
        )
    else:
        extractor = LoRARepresentationExtractor(
            config=rep_config,
            output_dir=output_dir,
            task_name=task,
            method=f"{method}_seed{seed}"
        )
    
    # Set validation examples
    eval_examples = {
        'input_ids': torch.stack([torch.tensor(ex['input_ids']) for ex in eval_dataset[:750]]),
        'attention_mask': torch.stack([torch.tensor(ex['attention_mask']) for ex in eval_dataset[:750]])
    }
    if 'labels' in eval_dataset[0]:
        eval_examples['labels'] = torch.stack([torch.tensor(ex['labels']) for ex in eval_dataset[:750]])
    
    extractor.set_validation_examples(eval_examples)
    
    # Load the saved model
    logger.info(f"Loading model from {model_path}...")
    
    # TODO: Load model properly based on task and method
    # This will be task-specific (SQuAD vs classification)
    # For now, placeholder:
    
    if task == "squad_v2":
        from models.squad_v2_qa_model import SquadV2QuestionAnsweringModel
        model = SquadV2QuestionAnsweringModel.from_pretrained(str(model_path))
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    
    logger.info("Model loaded successfully")
    
    # Extract representations
    logger.info("Extracting representations...")
    with torch.no_grad():
        representations = extractor.extract_representations(model, step=0)
    
    # Save representations
    extractor.save_representations(representations, step=0)
    
    logger.info(f"✅ Representations extracted and saved to {output_dir}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


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

