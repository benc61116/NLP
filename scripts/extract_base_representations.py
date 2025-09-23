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
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from shared.data_preparation import TaskDataLoader
from experiments.full_finetune import RepresentationExtractor, RepresentationConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_base_representations_for_task(model, tokenizer, data_loader, task_name: str, num_samples: int = 1000):
    """Extract base model representations for a specific task."""
    logger.info(f"Extracting base representations for {task_name} ({num_samples} samples)")
    
    try:
        # Prepare validation data
        if task_name == 'squad_v2':
            eval_dataset = data_loader.prepare_qa_data('validation', num_samples=num_samples)
        else:
            eval_dataset = data_loader.prepare_classification_data(task_name, 'validation', num_samples=num_samples)
        
        # Create output directory
        output_dir = Path('results/base_model_representations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create representation extractor
        repr_config = RepresentationConfig()
        extractor = RepresentationExtractor(
            repr_config,
            output_dir,
            task_name,
            "base_pretrained"
        )
        
        # Prepare validation examples for representation extraction
        examples = {
            'input_ids': eval_dataset['input_ids'],
            'attention_mask': eval_dataset['attention_mask']
        }
        if 'labels' in eval_dataset:
            examples['labels'] = eval_dataset['labels']
        
        extractor.set_validation_examples(examples)
        
        # Extract and save representations
        representations = extractor.extract_representations(model, step=0)
        extractor.save_representations(representations, step=0)
        
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
    
    logger.info("Starting base model representation extraction")
    
    # Model configuration
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
    
    # Initialize data loader
    data_loader = TaskDataLoader(model_name)
    
    # Extract representations for each task
    successful_extractions = 0
    for task_name in tasks:
        success = extract_base_representations_for_task(
            model, tokenizer, data_loader, task_name, num_samples=1000
        )
        if success:
            successful_extractions += 1
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Summary
    logger.info(f"✅ Base model representation extraction complete!")
    logger.info(f"Successful extractions: {successful_extractions}/{len(tasks)}")
    
    if successful_extractions == len(tasks):
        logger.info("🎉 All base model representations extracted successfully!")
        return 0
    else:
        logger.error(f"❌ {len(tasks) - successful_extractions} extractions failed")
        return 1

if __name__ == "__main__":
    exit(main())
