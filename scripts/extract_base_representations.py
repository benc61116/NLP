#!/usr/bin/env python3
"""Extract base model representations for drift analysis.

This script extracts representations from the original pre-trained model
(without any task-specific fine-tuning) to serve as baseline for 
representational drift analysis in Phase 2a.
"""

import os
import sys
import torch
import logging
import argparse
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from shared.data_preparation import TaskDataLoader
from experiments.full_finetune import RepresentationExtractor, RepresentationConfig

def load_shared_config():
    """Load the shared configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shared', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_base_representations_for_task(model, tokenizer, data_loader, task_name: str, num_samples: int = 750, model_name: str = None):
    """Extract base model representations for a specific task."""
    logger.info(f"Extracting base representations for {task_name} ({num_samples} samples)")
    
    # Force GPU memory cleanup before starting
    torch.cuda.empty_cache()
    
    try:
        # Prepare validation data
        if task_name == 'squad_v2':
            eval_dataset = data_loader.prepare_qa_data('validation', num_samples=num_samples)
        else:
            eval_dataset = data_loader.prepare_classification_data(task_name, 'validation', num_samples=num_samples)
        
        # Create output directory (NOT in results/ - this is persistent!)
        output_dir = Path('base_representations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create memory-optimized representation extractor
        repr_config = RepresentationConfig()
        # Keep all layers for drift analysis - this is required for the research
        repr_config.save_layers = list(range(24))  # All 24 layers needed for CKA analysis
        repr_config.memory_map = True
        
        extractor = RepresentationExtractor(
            repr_config,
            output_dir,
            task_name,
            "base_pretrained"
        )
        
        # Prepare validation examples for representation extraction
        # FIXED: prepare_qa_data returns a dict with lists, not a Dataset to iterate over
        input_ids_list = eval_dataset['input_ids']  # List of lists
        attention_mask_list = eval_dataset['attention_mask']  # List of lists
        
        # Convert lists to tensors
        input_ids_tensors = []
        attention_mask_tensors = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            input_ids_tensors.append(torch.tensor(ids, dtype=torch.long))
            attention_mask_tensors.append(torch.tensor(mask, dtype=torch.long))
        
        # Handle variable-length sequences with manual padding (simpler approach)
        try:
            input_ids = torch.stack(input_ids_tensors)
            attention_mask = torch.stack(attention_mask_tensors)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                # Pad to max length for variable-length sequences
                max_len = max(len(ids) for ids in input_ids_tensors)
                padded_input_ids = []
                padded_attention_mask = []
                for ids, mask in zip(input_ids_tensors, attention_mask_tensors):
                    pad_len = max_len - len(ids)
                    if pad_len > 0:
                        pad_id = 0  # Use 0 as pad token ID
                        padded_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                        padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                    else:
                        padded_ids = ids
                        padded_mask = mask
                    padded_input_ids.append(padded_ids)
                    padded_attention_mask.append(padded_mask)
                input_ids = torch.stack(padded_input_ids)
                attention_mask = torch.stack(padded_attention_mask)
            else:
                raise
        
        examples = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Handle labels if present for QA tasks
        if 'start_positions' in eval_dataset and 'end_positions' in eval_dataset:
            examples['start_positions'] = eval_dataset['start_positions']
            examples['end_positions'] = eval_dataset['end_positions']
        
        extractor.set_validation_examples(examples)
        
        # Extract and save representations with aggressive memory management
        logger.info(f"Starting representation extraction with memory optimization...")
        
        # Additional memory cleanup before extraction
        if task_name == 'squad_v2':
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Pre-extraction memory cleanup complete for {task_name}")
        
        # For SQuAD v2, representations are saved directly during extraction
        if task_name == 'squad_v2':
            result = extractor.extract_representations(model, step=0)
            logger.info(f"Representations saved during extraction for {task_name}")
        else:
            representations = extractor.extract_representations(model, step=0)
            
            # Force memory cleanup after extraction
            torch.cuda.empty_cache()
            
            # Save representations immediately
            logger.info(f"Saving representations for {task_name}...")
            extractor.save_representations(representations, step=0)
            
            # Clean up representations from memory immediately after saving
            del representations
            torch.cuda.empty_cache()
        
        # Additional cleanup for SQuAD v2
        if task_name == 'squad_v2':
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Post-save memory cleanup complete for {task_name}")
        
        logger.info(f"✅ Base representations extracted for {task_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to extract base representations for {task_name}: {e}")
        return False

def main():
    """Main function to extract base model representations for all tasks."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract base model representations')
    parser.add_argument('--task', type=str, help='Specific task to process (default: all tasks)')
    args = parser.parse_args()
    
    # Initialize wandb for base representation extraction
    import wandb
    wandb.init(
        project="NLP-Phase0",
        name=f"base_representations_extraction",
        tags=["base_representations", "phase0"],
        notes="Extracting base model representations for drift analysis"
    )
    
    logger.info("Starting base model representation extraction")
    
    # Load model configuration from shared config
    try:
        config = load_shared_config()
        model_name = config['model']['name']
        logger.info(f"Using model from config: {model_name}")
    except Exception as e:
        logger.warning(f"Could not load shared config: {e}, using fallback model")
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    all_tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
    
    # Select tasks to process
    if args.task:
        if args.task not in all_tasks:
            logger.error(f"Unknown task: {args.task}. Available tasks: {all_tasks}")
            return 1
        tasks = [args.task]
        logger.info(f"Processing single task: {args.task}")
    else:
        tasks = all_tasks
        logger.info("Processing all tasks")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Initialize data loader with methodological consistency (max_length=384)
    # CRITICAL: Must match Phase 2 training for valid drift analysis
    data_loader = TaskDataLoader(model_name, max_length=384)
    
    logger.info("Using max_length=384 for methodological consistency with Phase 2 training")
    
    # Extract representations for each task
    successful_extractions = 0
    for task_name in tasks:
        # Adaptive 750 samples: uses all samples for small tasks (MRPC/RTE), optimized for large tasks  
        samples = 750
        success = extract_base_representations_for_task(
            model, tokenizer, data_loader, task_name, num_samples=samples, model_name=model_name
        )
        if success:
            successful_extractions += 1
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Summary
    logger.info(f"✅ Base model representation extraction complete!")
    logger.info(f"Successful extractions: {successful_extractions}/{len(tasks)}")
    
    # Upload base representations to wandb as artifact
    if successful_extractions == len(tasks):
        logger.info("☁️  Uploading base representations to WandB...")
        try:
            artifact = wandb.Artifact(
                name="base_representations",
                type="base_representations",
                description="Base model representations for all tasks (for drift analysis)",
                metadata={
                    "model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                    "tasks": tasks,
                    "num_samples_per_task": 750,
                    "num_layers": 24
                }
            )
            artifact.add_dir("base_representations")
            wandb.log_artifact(artifact)
            logger.info("✅ Base representations uploaded to WandB!")
        except Exception as e:
            logger.warning(f"⚠️  Failed to upload to WandB: {e}")
            logger.info("Local copy still saved in base_representations/")
        
        wandb.finish()
        logger.info("🎉 All base model representations extracted successfully!")
        return 0
    else:
        wandb.finish()
        logger.error(f"❌ {len(tasks) - successful_extractions} extractions failed")
        return 1

if __name__ == "__main__":
    exit(main())
