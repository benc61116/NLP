#!/usr/bin/env python3
"""Data loading and preprocessing utilities for all NLP tasks."""

import os
import random
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import json


class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""
    pass


class TaskDataLoader:
    """Unified data loader for all NLP tasks in the experiment."""
    
    TASK_CONFIGS = {
        "mrpc": {
            "path": "data/mrpc",
            "task_type": "classification",
            "input_keys": ["sentence1", "sentence2"],
            "label_key": "label",
            "num_labels": 2,
            "description": "Microsoft Research Paraphrase Corpus - paraphrase detection"
        },
        "sst2": {
            "path": "data/sst2", 
            "task_type": "classification",
            "input_keys": ["sentence"],
            "label_key": "label",
            "num_labels": 2,
            "description": "Stanford Sentiment Treebank - binary sentiment classification"
        },
        "rte": {
            "path": "data/rte",
            "task_type": "classification", 
            "input_keys": ["sentence1", "sentence2"],
            "label_key": "label",
            "num_labels": 2,
            "description": "Recognizing Textual Entailment"
        },
        "squad_v2": {
            "path": "data/squad_v2",
            "task_type": "qa",
            "input_keys": ["context", "question"],
            "label_key": "answers",
            "num_labels": None,  # Variable for QA
            "description": "Stanford Question Answering Dataset v2.0"
        }
    }
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-1.3b-hf", max_length: int = 512):
        """Initialize the data loader with model tokenizer.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.datasets = {}
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load all datasets from disk."""
        for task_name, config in self.TASK_CONFIGS.items():
            dataset_path = config["path"]
            if os.path.exists(dataset_path):
                try:
                    self.datasets[task_name] = load_from_disk(dataset_path)
                    print(f"✓ Loaded {task_name} dataset from {dataset_path}")
                except Exception as e:
                    print(f"✗ Failed to load {task_name}: {e}")
                    raise DataPreparationError(f"Could not load {task_name} dataset: {e}")
            else:
                raise DataPreparationError(f"Dataset path {dataset_path} does not exist. Run scripts/download_datasets.py first.")
    
    def get_task_info(self, task_name: str) -> Dict:
        """Get task configuration information."""
        if task_name not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(self.TASK_CONFIGS.keys())}")
        return self.TASK_CONFIGS[task_name].copy()
    
    def get_sample_data(self, task_name: str, split: str = "train", num_samples: int = 10, seed: int = 42) -> Dataset:
        """Get a small sample of data for sanity checks.
        
        Args:
            task_name: Name of the task
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Number of samples to return
            seed: Random seed for reproducibility
            
        Returns:
            Dataset with sampled examples
        """
        if task_name not in self.datasets:
            raise ValueError(f"Task {task_name} not loaded")
            
        dataset = self.datasets[task_name]
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{split}' not available for {task_name}. Available: {available_splits}")
        
        full_dataset = dataset[split]
        total_samples = len(full_dataset)
        
        if num_samples >= total_samples:
            print(f"Warning: Requested {num_samples} samples but only {total_samples} available. Returning all.")
            return full_dataset
        
        # Set seed for reproducible sampling
        random.seed(seed)
        indices = random.sample(range(total_samples), num_samples)
        return full_dataset.select(indices)
    
    def prepare_classification_data(self, task_name: str, split: str = "train", num_samples: Optional[int] = None) -> Dict:
        """Prepare data for classification tasks (MRPC, SST-2, RTE).
        
        Args:
            task_name: Name of the classification task
            split: Dataset split
            num_samples: If provided, sample this many examples
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        config = self.get_task_info(task_name)
        if config["task_type"] != "classification":
            raise ValueError(f"Task {task_name} is not a classification task")
        
        if num_samples:
            dataset = self.get_sample_data(task_name, split, num_samples)
        else:
            dataset = self.datasets[task_name][split]
        
        # Prepare text inputs based on task configuration
        texts = []
        labels = []
        
        for example in dataset:
            if len(config["input_keys"]) == 1:
                # Single sentence tasks (SST-2)
                text = example[config["input_keys"][0]]
            else:
                # Sentence pair tasks (MRPC, RTE)
                text = f"{example[config['input_keys'][0]]} [SEP] {example[config['input_keys'][1]]}"
            
            texts.append(text)
            labels.append(example[config["label_key"]])
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_name": task_name,
            "num_samples": len(texts)
        }
    
    def prepare_qa_data(self, split: str = "train", num_samples: Optional[int] = None) -> Dict:
        """Prepare data for question answering task (SQuAD v2).
        
        Args:
            split: Dataset split
            num_samples: If provided, sample this many examples
            
        Returns:
            Dictionary with tokenized inputs and answer information
        """
        task_name = "squad_v2"
        
        if num_samples:
            dataset = self.get_sample_data(task_name, split, num_samples)
        else:
            dataset = self.datasets[task_name][split]
        
        input_ids_list = []
        attention_mask_list = []
        start_positions = []
        end_positions = []
        
        for example in dataset:
            context = example["context"]
            question = example["question"]
            answers = example["answers"]
            
            # Tokenize question and context separately to get offsets
            tokenized = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                padding=False  # We'll handle padding in the collator
            )
            
            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
            
            # Handle answer positions with proper token alignment
            if answers["text"] and len(answers["text"]) > 0:
                # Use first answer for simplicity
                answer_start_char = answers["answer_start"][0]
                answer_text = answers["text"][0]
                answer_end_char = answer_start_char + len(answer_text)
                
                # Find token positions that correspond to character positions
                offset_mapping = tokenized["offset_mapping"]
                start_token = None
                end_token = None
                
                # Find start token
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char <= answer_start_char < end_char:
                        start_token = i
                        break
                
                # Find end token (exclusive, so we want the token after the answer)
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char < answer_end_char <= end_char:
                        end_token = i
                        break
                
                # Validation and fallback
                if start_token is not None and end_token is not None and start_token <= end_token:
                    start_positions.append(start_token)
                    end_positions.append(end_token)
                else:
                    # Fallback: mark as unanswerable
                    start_positions.append(0)  # CLS token
                    end_positions.append(0)
            else:
                # No answer (SQuAD v2 has unanswerable questions)
                start_positions.append(0)  # CLS token for unanswerable
                end_positions.append(0)
        
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
            "task_name": task_name,
            "num_samples": len(input_ids_list)
        }
    
    def get_all_task_samples(self, num_samples_per_task: int = 10, split: str = "train", seed: int = 42) -> Dict:
        """Get samples from all tasks for comprehensive testing.
        
        Args:
            num_samples_per_task: Number of samples per task
            split: Dataset split to use
            seed: Random seed
            
        Returns:
            Dictionary mapping task names to prepared data
        """
        all_samples = {}
        
        # Classification tasks
        for task_name in ["mrpc", "sst2", "rte"]:
            try:
                all_samples[task_name] = self.prepare_classification_data(
                    task_name, split, num_samples_per_task
                )
                print(f"✓ Prepared {num_samples_per_task} samples for {task_name}")
            except Exception as e:
                print(f"✗ Failed to prepare {task_name}: {e}")
        
        # QA task
        try:
            all_samples["squad_v2"] = self.prepare_qa_data(split, num_samples_per_task)
            print(f"✓ Prepared {num_samples_per_task} samples for squad_v2")
        except Exception as e:
            print(f"✗ Failed to prepare squad_v2: {e}")
        
        return all_samples
    
    def print_dataset_summary(self):
        """Print a summary of all loaded datasets."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        for task_name, config in self.TASK_CONFIGS.items():
            if task_name in self.datasets:
                dataset = self.datasets[task_name]
                print(f"\n{task_name.upper()}: {config['description']}")
                print(f"  Task type: {config['task_type']}")
                print(f"  Input keys: {config['input_keys']}")
                print(f"  Label key: {config['label_key']}")
                if config['num_labels']:
                    print(f"  Number of labels: {config['num_labels']}")
                
                print("  Splits:")
                for split_name, split_data in dataset.items():
                    print(f"    {split_name}: {len(split_data):,} examples")
            else:
                print(f"\n{task_name.upper()}: NOT LOADED")
        
        print("\n" + "="*60)
    
    def validate_data_integrity(self) -> bool:
        """Validate that all datasets are properly loaded and formatted.
        
        Returns:
            True if all datasets pass validation
        """
        print("\nValidating data integrity...")
        all_valid = True
        
        for task_name in self.TASK_CONFIGS.keys():
            try:
                # Try to load a small sample from each task
                if self.TASK_CONFIGS[task_name]["task_type"] == "classification":
                    sample = self.prepare_classification_data(task_name, "train", 5)
                    assert "input_ids" in sample
                    assert "labels" in sample
                    assert sample["input_ids"].shape[0] == 5
                else:  # QA task
                    sample = self.prepare_qa_data("train", 5)
                    assert "input_ids" in sample
                    assert "start_positions" in sample
                    assert sample["input_ids"].shape[0] == 5
                
                print(f"✓ {task_name}: Data integrity validated")
                
            except Exception as e:
                print(f"✗ {task_name}: Validation failed - {e}")
                all_valid = False
        
        return all_valid


def create_data_loader(model_name: str = "meta-llama/Llama-2-1.3b-hf", max_length: int = 512) -> TaskDataLoader:
    """Factory function to create a data loader.
    
    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        
    Returns:
        Configured TaskDataLoader instance
    """
    return TaskDataLoader(model_name, max_length)


if __name__ == "__main__":
    # Demo and validation
    print("Initializing data loader...")
    loader = create_data_loader()
    
    # Print summary
    loader.print_dataset_summary()
    
    # Validate data integrity
    if loader.validate_data_integrity():
        print("\n✓ All datasets validated successfully!")
    else:
        print("\n✗ Some datasets failed validation!")
        exit(1)
    
    # Test sample loading
    print("\nTesting sample data loading...")
    samples = loader.get_all_task_samples(num_samples_per_task=5)
    
    print(f"\nLoaded samples from {len(samples)} tasks:")
    for task_name, data in samples.items():
        print(f"  {task_name}: {data['num_samples']} samples")
        print(f"    Input shape: {data['input_ids'].shape}")
        if 'labels' in data:
            print(f"    Labels shape: {data['labels'].shape}")
    
    print("\n✓ Data preparation module validated!")
