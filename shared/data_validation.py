#!/usr/bin/env python3
"""Data validation utilities to ensure no data leakage and proper splits."""

import os
import sys
import logging
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_preparation import TaskDataLoader

logger = logging.getLogger(__name__)

def hash_text(text: str) -> str:
    """Create a hash of text for duplicate detection."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def validate_no_overlap(train_texts: List[str], val_texts: List[str], test_texts: List[str] = None) -> Dict[str, Any]:
    """Validate that there's no overlap between train/validation/test splits."""
    results = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Convert to sets of hashes for efficient comparison
    train_hashes = {hash_text(text) for text in train_texts}
    val_hashes = {hash_text(text) for text in val_texts}
    
    # Check train/validation overlap
    train_val_overlap = train_hashes.intersection(val_hashes)
    if train_val_overlap:
        results['valid'] = False
        results['issues'].append(f"Found {len(train_val_overlap)} overlapping samples between train and validation")
    
    results['statistics']['train_samples'] = len(train_texts)
    results['statistics']['val_samples'] = len(val_texts)
    results['statistics']['train_val_overlap'] = len(train_val_overlap)
    
    # Check test overlap if test set provided
    if test_texts is not None:
        test_hashes = {hash_text(text) for text in test_texts}
        
        train_test_overlap = train_hashes.intersection(test_hashes)
        val_test_overlap = val_hashes.intersection(test_hashes)
        
        if train_test_overlap:
            results['valid'] = False
            results['issues'].append(f"Found {len(train_test_overlap)} overlapping samples between train and test")
            
        if val_test_overlap:
            results['valid'] = False
            results['issues'].append(f"Found {len(val_test_overlap)} overlapping samples between validation and test")
        
        results['statistics']['test_samples'] = len(test_texts)
        results['statistics']['train_test_overlap'] = len(train_test_overlap)
        results['statistics']['val_test_overlap'] = len(val_test_overlap)
    
    return results

def validate_data_splits_for_task(task_name: str, data_loader: TaskDataLoader) -> Dict[str, Any]:
    """Validate data splits for a specific task."""
    logger.info(f"Validating data splits for {task_name}")
    
    results = {
        'task': task_name,
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    try:
        # Load train and validation data
        if task_name == 'squad_v2':
            train_data = data_loader.prepare_qa_data('train', num_samples=1000)  # Sample for efficiency
            val_data = data_loader.prepare_qa_data('validation', num_samples=500)
        else:
            # Classification tasks
            train_data = data_loader.prepare_classification_data(task_name, 'train', num_samples=1000)  # Sample for efficiency
            val_data = data_loader.prepare_classification_data(task_name, 'validation', num_samples=500)
        
        # Decode tokenized texts back to strings for comparison
        train_texts = []
        val_texts = []
        
        # Handle different data formats (QA data uses lists, classification data uses tensors)
        if isinstance(train_data['input_ids'], list):
            # QA data format (lists)
            for input_ids in train_data['input_ids']:
                decoded_text = data_loader.tokenizer.decode(input_ids, skip_special_tokens=True)
                train_texts.append(decoded_text.strip())
            
            for input_ids in val_data['input_ids']:
                decoded_text = data_loader.tokenizer.decode(input_ids, skip_special_tokens=True)
                val_texts.append(decoded_text.strip())
        else:
            # Classification data format (tensors)
            for i in range(train_data['input_ids'].shape[0]):
                decoded_text = data_loader.tokenizer.decode(train_data['input_ids'][i], skip_special_tokens=True)
                train_texts.append(decoded_text.strip())
            
            for i in range(val_data['input_ids'].shape[0]):
                decoded_text = data_loader.tokenizer.decode(val_data['input_ids'][i], skip_special_tokens=True)
                val_texts.append(decoded_text.strip())
        
        # Validate no overlap
        overlap_results = validate_no_overlap(train_texts, val_texts)
        
        results['valid'] = overlap_results['valid']
        results['issues'] = overlap_results['issues']
        results['statistics'] = overlap_results['statistics']
        
        if results['valid']:
            logger.info(f"✅ Data split validation PASSED for {task_name}")
            logger.info(f"   Train: {results['statistics']['train_samples']} samples")
            logger.info(f"   Val: {results['statistics']['val_samples']} samples")
            logger.info(f"   Overlap: {results['statistics']['train_val_overlap']} samples")
        else:
            logger.error(f"❌ Data split validation FAILED for {task_name}")
            for issue in results['issues']:
                logger.error(f"   - {issue}")
        
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Data validation failed: {e}")
        logger.error(f"❌ Data validation error for {task_name}: {e}")
    
    return results

def validate_data_quality_for_task(task_name: str, data_loader: TaskDataLoader) -> Dict[str, Any]:
    """Validate data quality for a specific task."""
    logger.info(f"Validating data quality for {task_name}")
    
    results = {
        'task': task_name,
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    try:
        # Load train data for quality checks
        if task_name == 'squad_v2':
            train_data = data_loader.prepare_qa_data('train', num_samples=500)
        else:
            train_data = data_loader.prepare_classification_data(task_name, 'train', num_samples=500)
        
        # Decode tokenized texts back to strings for quality checks
        texts = []
        
        # Handle different data formats (QA data uses lists, classification data uses tensors)
        if isinstance(train_data['input_ids'], list):
            # QA data format (lists)
            for input_ids in train_data['input_ids']:
                decoded_text = data_loader.tokenizer.decode(input_ids, skip_special_tokens=True)
                texts.append(decoded_text.strip())
        else:
            # Classification data format (tensors)
            for i in range(train_data['input_ids'].shape[0]):
                decoded_text = data_loader.tokenizer.decode(train_data['input_ids'][i], skip_special_tokens=True)
                texts.append(decoded_text.strip())
        
        # Check for empty/null texts
        empty_texts = 0
        very_short_texts = 0
        very_long_texts = 0
        
        for text in texts:
            if not text or text.strip() == '':
                empty_texts += 1
            elif len(text.strip()) < 10:
                very_short_texts += 1
            elif len(text.strip()) > 2000:
                very_long_texts += 1
        
        results['statistics']['total_samples'] = len(texts)
        results['statistics']['empty_texts'] = empty_texts
        results['statistics']['very_short_texts'] = very_short_texts
        results['statistics']['very_long_texts'] = very_long_texts
        
        # Validate data quality
        if empty_texts > 0:
            results['valid'] = False
            results['issues'].append(f"Found {empty_texts} empty text samples")
        
        if very_short_texts > len(texts) * 0.1:  # More than 10% very short
            results['issues'].append(f"Found {very_short_texts} very short text samples (>10% of dataset)")
        
        # Check label distribution for classification tasks
        if task_name != 'squad_v2' and 'labels' in train_data:
            label_counts = defaultdict(int)
            for label in train_data['labels']:
                label_counts[int(label)] += 1  # Convert tensor to int
            
            results['statistics']['label_distribution'] = dict(label_counts)
            
            # Check for severely imbalanced datasets
            total_samples = len(train_data['labels'])
            min_class_ratio = min(label_counts.values()) / total_samples
            
            if min_class_ratio < 0.05:  # Less than 5% for minority class
                results['issues'].append(f"Severely imbalanced dataset: minority class has {min_class_ratio:.2%} of samples")
        
        if results['valid'] and not results['issues']:
            logger.info(f"✅ Data quality validation PASSED for {task_name}")
        else:
            logger.warning(f"⚠️ Data quality issues found for {task_name}")
            for issue in results['issues']:
                logger.warning(f"   - {issue}")
        
    except Exception as e:
        results['valid'] = False
        results['issues'].append(f"Data quality validation failed: {e}")
        logger.error(f"❌ Data quality validation error for {task_name}: {e}")
    
    return results

def validate_all_tasks() -> Dict[str, Any]:
    """Validate data splits and quality for all tasks."""
    from shared.model_validation import load_config
    
    try:
        config = load_config()
        model_name = config['model']['name']
    except:
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    data_loader = TaskDataLoader(model_name)
    all_tasks = ['mrpc', 'sst2', 'rte', 'squad_v2']
    
    results = {
        'overall_valid': True,
        'task_results': {},
        'summary': {
            'total_tasks': len(all_tasks),
            'passed_split_validation': 0,
            'passed_quality_validation': 0,
            'issues_found': []
        }
    }
    
    logger.info("Starting comprehensive data validation for all tasks")
    logger.info("=" * 60)
    
    for task_name in all_tasks:
        logger.info(f"\nValidating {task_name.upper()}...")
        
        # Validate data splits
        split_results = validate_data_splits_for_task(task_name, data_loader)
        
        # Validate data quality
        quality_results = validate_data_quality_for_task(task_name, data_loader)
        
        # Combine results
        task_result = {
            'split_validation': split_results,
            'quality_validation': quality_results,
            'overall_valid': split_results['valid'] and quality_results['valid']
        }
        
        results['task_results'][task_name] = task_result
        
        if split_results['valid']:
            results['summary']['passed_split_validation'] += 1
        
        if quality_results['valid']:
            results['summary']['passed_quality_validation'] += 1
        
        if not task_result['overall_valid']:
            results['overall_valid'] = False
            results['summary']['issues_found'].extend(
                split_results['issues'] + quality_results['issues']
            )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATA VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    if results['overall_valid']:
        logger.info("🎉 ALL DATA VALIDATION CHECKS PASSED")
    else:
        logger.error("❌ DATA VALIDATION ISSUES FOUND")
        for issue in results['summary']['issues_found']:
            logger.error(f"   - {issue}")
    
    logger.info(f"Split validation: {results['summary']['passed_split_validation']}/{results['summary']['total_tasks']} tasks passed")
    logger.info(f"Quality validation: {results['summary']['passed_quality_validation']}/{results['summary']['total_tasks']} tasks passed")
    
    return results

def main():
    """Run all data validation checks."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    results = validate_all_tasks()
    
    if results['overall_valid']:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
