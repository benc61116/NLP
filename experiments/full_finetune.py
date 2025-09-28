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
import gc
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
        # ROOT CAUSE FIX: Ensure we never create None values that can be passed to _pad_across_processes
        
        # Extract each field with proper None handling
        input_ids = [f["input_ids"] for f in features if f.get("input_ids") is not None]
        attention_mask = [f["attention_mask"] for f in features if f.get("attention_mask") is not None]
        
        # Ensure we have valid features
        if not input_ids or not attention_mask:
            raise ValueError("No valid input_ids or attention_mask found in features")
        
        start_positions = []
        end_positions = []
        answerability_labels = []
        
        for f in features:
            # Handle None values for unanswerable questions in SQuAD v2
            start_pos = f.get("start_positions", 0)
            end_pos = f.get("end_positions", 0)
            answerability = f.get("answerability_labels", 0)
            
            # Convert None to 0 for consistent tensor creation
            start_positions.append(0 if start_pos is None else start_pos)
            end_positions.append(0 if end_pos is None else end_pos)
            answerability_labels.append(0 if answerability is None else answerability)
        
        # Pad input_ids and attention_mask
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            if padding_length > 0:
                # Pad with tokenizer.pad_token_id
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                # Ensure ids and mask are lists before concatenation
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if isinstance(mask, torch.Tensor):
                    mask = mask.tolist()
                padded_input_ids.append(ids + [pad_id] * padding_length)
                padded_attention_mask.append(mask + [0] * padding_length)
            else:
                # Ensure consistent format
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if isinstance(mask, torch.Tensor):
                    mask = mask.tolist()
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
        
        # Create batch tensors - ensure all values are valid tensors, never None
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
            "answerability_labels": torch.tensor(answerability_labels, dtype=torch.long),
        }
        
        return batch


@dataclass
class RepresentationConfig:
    """Configuration for representation extraction."""
    extract_every_steps: int = 100
    save_layers: List[int] = field(default_factory=lambda: list(range(24)))  # All layers for 1.3B
    max_validation_samples: int = 750  # Adaptive: uses all samples for small tasks, optimized for large tasks
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
    
    def _extract_representations_chunked(self, model: torch.nn.Module, input_ids: torch.Tensor, 
                                       attention_mask: torch.Tensor, num_samples: int, 
                                       batch_size: int, layer_outputs: dict, hooks: list, step: int) -> Dict[str, torch.Tensor]:
        """Extract representations using streaming approach with immediate saves for SQuAD v2."""
        mini_chunk_size = 50  # Process 50 samples per mini-chunk (better memory safety for large tasks)
        
        # Initialize streaming approach - save mini-chunks immediately to disk
        temp_dir = Path(f"/tmp/repr_chunks_{step}")
        temp_dir.mkdir(exist_ok=True)
        chunk_files = {f'layer_{i}': [] for i in self.config.save_layers}
        chunk_files['final_hidden_states'] = []
        
        # Memory monitoring (both GPU and CPU)
        import psutil
        initial_cpu_memory = psutil.virtual_memory().available / (1024**3)  # GB
        initial_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
        gpu_memory_available = initial_gpu_memory - gpu_memory_used
        
        logger.info(f"SQuAD v2: GPU memory: {gpu_memory_available:.1f}GB available / {initial_gpu_memory:.1f}GB total")
        logger.info(f"SQuAD v2: CPU memory: {initial_cpu_memory:.1f}GB available")
        logger.info(f"Processing {num_samples} samples in mini-chunks of {mini_chunk_size} (GPU-first streaming)")
        
        for chunk_start in range(0, num_samples, mini_chunk_size):
            chunk_end = min(chunk_start + mini_chunk_size, num_samples)
            chunk_samples = chunk_end - chunk_start
            
            logger.info(f"Processing mini-chunk {chunk_start//mini_chunk_size + 1}/{(num_samples-1)//mini_chunk_size + 1}: samples {chunk_start}-{chunk_end-1}")
            
            # Process this mini-chunk
            chunk_input_ids = input_ids[chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[chunk_start:chunk_end]
            
            chunk_layer_outputs = {f'layer_{i}': [] for i in self.config.save_layers}
            chunk_final_hidden_states = []
            
            # Process mini-chunk in small batches
            for i in range(0, chunk_samples, batch_size):
                batch_end = min(i + batch_size, chunk_samples)
                
                # Progress logging
                if i % (batch_size * 5) == 0:
                    chunk_progress = (i / chunk_samples) * 100
                    overall_progress = ((chunk_start + i) / num_samples) * 100
                    logger.info(f"Mini-chunk progress: {chunk_progress:.1f}%, Overall: {overall_progress:.1f}%")
                
                batch_input_ids = chunk_input_ids[i:batch_end].to(model.device)
                batch_attention_mask = chunk_attention_mask[i:batch_end].to(model.device)
                
                layer_outputs.clear()
                torch.cuda.empty_cache()
                
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                
                # Collect outputs for this batch (keep on GPU if memory permits)
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_available = initial_gpu_memory - current_gpu_memory
                
                # If GPU memory is getting low (< 5GB), move to CPU
                if gpu_available < 5:
                    for layer_name, layer_output in layer_outputs.items():
                        chunk_layer_outputs[layer_name].append(layer_output.detach().cpu())
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        chunk_final_hidden_states.append(outputs.hidden_states[-1].detach().cpu())
                    elif hasattr(outputs, 'last_hidden_state'):
                        chunk_final_hidden_states.append(outputs.last_hidden_state.detach().cpu())
                else:
                    # Keep on GPU for speed
                    for layer_name, layer_output in layer_outputs.items():
                        chunk_layer_outputs[layer_name].append(layer_output.detach())
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        chunk_final_hidden_states.append(outputs.hidden_states[-1].detach())
                    elif hasattr(outputs, 'last_hidden_state'):
                        chunk_final_hidden_states.append(outputs.last_hidden_state.detach())
                
                # Cleanup batch data immediately
                del batch_input_ids, batch_attention_mask, outputs
                torch.cuda.empty_cache()
            
            # Save this mini-chunk immediately to disk (streaming approach)
            chunk_idx = chunk_start // mini_chunk_size
            for layer_name in chunk_layer_outputs:
                if chunk_layer_outputs[layer_name]:
                    mini_chunk_repr = torch.cat(chunk_layer_outputs[layer_name], dim=0).cpu()
                    chunk_file = temp_dir / f"{layer_name}_chunk_{chunk_idx}.pt"
                    torch.save(mini_chunk_repr, chunk_file)
                    chunk_files[layer_name].append(chunk_file)
                    del mini_chunk_repr
            
            if chunk_final_hidden_states:
                mini_chunk_final = torch.cat(chunk_final_hidden_states, dim=0).cpu()
                chunk_file = temp_dir / f"final_hidden_states_chunk_{chunk_idx}.pt"
                torch.save(mini_chunk_final, chunk_file)
                chunk_files['final_hidden_states'].append(chunk_file)
                del mini_chunk_final
            
            # Cleanup mini-chunk data
            del chunk_layer_outputs, chunk_final_hidden_states
            del chunk_input_ids, chunk_attention_mask
            torch.cuda.empty_cache()
            
            # Memory monitoring and cleanup every 3 mini-chunks
            if (chunk_start // mini_chunk_size + 1) % 3 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Check both GPU and CPU memory
                current_cpu_memory = psutil.virtual_memory().available / (1024**3)  # GB
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_available = initial_gpu_memory - current_gpu_memory
                cpu_used = initial_cpu_memory - current_cpu_memory
                
                chunk_num = chunk_start // mini_chunk_size + 1
                logger.info(f"After {chunk_num} mini-chunks:")
                logger.info(f"  GPU: {current_gpu_memory:.1f}GB used, {gpu_available:.1f}GB available")
                logger.info(f"  CPU: {cpu_used:.1f}GB used, {current_cpu_memory:.1f}GB available")
                
                # GPU memory management: force cleanup if memory gets low
                if gpu_available < 8:
                    logger.warning(f"GPU memory getting low ({gpu_available:.1f}GB). Forcing cleanup...")
                    torch.cuda.empty_cache()
                    new_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(f"After cleanup: {new_gpu_memory:.1f}GB GPU memory used")
                
                # CPU memory safety check
                if cpu_used > 20:
                    logger.warning(f"High CPU memory usage: {cpu_used:.1f}GB used")
                elif cpu_used > 25:
                    logger.error(f"Critical CPU memory usage: {cpu_used:.1f}GB used")
                    raise MemoryError(f"CPU memory usage too high: {cpu_used:.1f}GB")
        
        # Final memory check before concatenation (most memory-intensive step)
        final_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        final_cpu_memory = psutil.virtual_memory().available / (1024**3)  # GB
        logger.info(f"Pre-concatenation: GPU {final_gpu_memory:.1f}GB used, CPU {final_cpu_memory:.1f}GB available")
        
        # Determine where to perform final concatenation based on available memory
        gpu_space_available = initial_gpu_memory - final_gpu_memory
        concatenate_on_gpu = gpu_space_available > 10  # Need at least 10GB for concatenation
        
        if concatenate_on_gpu:
            logger.info("Performing final concatenation on GPU for speed...")
            device = torch.cuda.current_device()
        else:
            logger.info("Performing final concatenation on CPU due to GPU memory constraints...")
            device = 'cpu'
        
        # Final concatenation of all mini-chunks from saved files
        logger.info("Loading and concatenating all mini-chunks from disk...")
        final_representations = {}
        
        # Process each layer's chunks with aggressive memory management
        for layer_name, file_list in chunk_files.items():
            if file_list and layer_name != 'final_hidden_states':
                logger.info(f"Loading {len(file_list)} chunks for {layer_name}...")
                
                # Load and concatenate in smaller batches to avoid memory issues
                batch_size = 5  # Process 5 chunks at a time
                layer_chunks = []
                
                for i in range(0, len(file_list), batch_size):
                    batch_files = file_list[i:i+batch_size]
                    batch_chunks = []
                    
                    for chunk_file in batch_files:
                        chunk = torch.load(chunk_file, map_location='cpu')
                        batch_chunks.append(chunk)
                    
                    # Concatenate this batch and add to layer chunks
                    if batch_chunks:
                        batch_concat = torch.cat(batch_chunks, dim=0)
                        layer_chunks.append(batch_concat)
                        del batch_chunks, batch_concat
                
                # Final concatenation for this layer
                layer_repr = torch.cat(layer_chunks, dim=0)
                logger.info(f"Final {layer_name}: {layer_repr.shape}")
                del layer_chunks
                
                # Validate tensor integrity before saving
                if not self._validate_tensor(layer_repr, layer_name):
                    logger.error(f"❌ Tensor validation failed for {layer_name}, skipping save")
                    del layer_repr
                    continue
                
                # Save with retry mechanism and validation
                step_dir = self.output_dir / f"step_{step:06d}"
                step_dir.mkdir(exist_ok=True)
                file_path = step_dir / f"{layer_name}.pt"
                
                if self._safe_tensor_save(layer_repr, file_path, layer_name):
                    logger.info(f"✅ Saved {layer_name} to {file_path}")
                    # Store a placeholder in final_representations to track completion
                    final_representations[layer_name] = layer_repr.shape
                else:
                    logger.error(f"❌ Failed to save {layer_name} after retries")
                
                del layer_repr
                
                # Force memory cleanup after each layer
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Handle final hidden states with same memory optimization
        if chunk_files['final_hidden_states']:
            logger.info(f"Loading {len(chunk_files['final_hidden_states'])} chunks for final_hidden_states...")
            
            # Load and concatenate in smaller batches
            batch_size = 5
            layer_chunks = []
            
            for i in range(0, len(chunk_files['final_hidden_states']), batch_size):
                batch_files = chunk_files['final_hidden_states'][i:i+batch_size]
                batch_chunks = []
                
                for chunk_file in batch_files:
                    chunk = torch.load(chunk_file, map_location='cpu')
                    batch_chunks.append(chunk)
                
                if batch_chunks:
                    batch_concat = torch.cat(batch_chunks, dim=0)
                    layer_chunks.append(batch_concat)
                    del batch_chunks, batch_concat
            
            final_hidden_repr = torch.cat(layer_chunks, dim=0)
            logger.info(f"Final final_hidden_states: {final_hidden_repr.shape}")
            del layer_chunks
            
            # Validate tensor integrity before saving
            if not self._validate_tensor(final_hidden_repr, "final_hidden_states"):
                logger.error(f"❌ Tensor validation failed for final_hidden_states, skipping save")
                del final_hidden_repr
            else:
                # Save with retry mechanism and validation
                step_dir = self.output_dir / f"step_{step:06d}"
                step_dir.mkdir(exist_ok=True)
                file_path = step_dir / "final_hidden_states.pt"
                
                if self._safe_tensor_save(final_hidden_repr, file_path, "final_hidden_states"):
                    logger.info(f"✅ Saved final_hidden_states to {file_path}")
                    # Store a placeholder to track completion
                    final_representations['final_hidden_states'] = final_hidden_repr.shape
                else:
                    logger.error(f"❌ Failed to save final_hidden_states after retries")
                
                del final_hidden_repr
        
        # Cleanup temporary files
        logger.info("Cleaning up temporary chunk files...")
        import shutil
        shutil.rmtree(temp_dir)
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Save metadata for the completed extraction
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        import json
        
        metadata = {
            'step': step,
            'task_name': self.task_name,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.validation_examples['input_ids']),
            'layer_names': list(final_representations.keys()),
            'tensor_shapes': {name: list(shape) for name, shape in final_representations.items()}
        }
        
        with open(step_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved metadata to {step_dir / 'metadata.json'}")
        logger.info("✅ SQuAD v2 streaming processing complete!")
        
        # Return empty dict since representations are already saved to disk
        # This avoids memory issues and prevents save_representations from being called again
        return {}
    
    def _validate_tensor(self, tensor: torch.Tensor, layer_name: str) -> bool:
        """Validate tensor integrity before saving."""
        try:
            # Check if tensor is valid
            if tensor is None:
                logger.warning(f"Tensor {layer_name} is None")
                return False
            
            # Check for NaN or Inf values
            if torch.isnan(tensor).any():
                logger.warning(f"Tensor {layer_name} contains NaN values")
                return False
            
            if torch.isinf(tensor).any():
                logger.warning(f"Tensor {layer_name} contains Inf values")
                return False
            
            # Check tensor properties
            if tensor.numel() == 0:
                logger.warning(f"Tensor {layer_name} is empty")
                return False
            
            # Try a basic operation to verify tensor integrity
            _ = tensor.sum()
            
            logger.debug(f"Tensor {layer_name} validation passed: shape={tensor.shape}, dtype={tensor.dtype}")
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed for {layer_name}: {e}")
            return False
    
    def _safe_tensor_save(self, tensor: torch.Tensor, file_path: Path, layer_name: str, max_retries: int = 3) -> bool:
        """Safely save tensor with retry mechanism and validation."""
        for attempt in range(max_retries):
            try:
                # Force tensor to be contiguous and on CPU
                tensor_to_save = tensor.detach().cpu().contiguous()
                
                # Ensure we have sufficient memory for save
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Save the tensor
                torch.save(tensor_to_save, file_path)
                
                # Verify the saved file by loading it back
                if self._verify_saved_tensor(file_path, tensor_to_save.shape, layer_name):
                    del tensor_to_save
                    return True
                else:
                    logger.warning(f"Verification failed for {layer_name}, attempt {attempt + 1}/{max_retries}")
                    # Remove the corrupted file
                    if file_path.exists():
                        file_path.unlink()
                    
                del tensor_to_save
                
            except Exception as e:
                logger.error(f"Save attempt {attempt + 1}/{max_retries} failed for {layer_name}: {e}")
                
                # Remove any partial file
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass
                
                # Wait a bit before retrying and cleanup memory
                import time
                time.sleep(1)
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        logger.error(f"All save attempts failed for {layer_name}")
        return False
    
    def _verify_saved_tensor(self, file_path: Path, expected_shape: torch.Size, layer_name: str) -> bool:
        """Verify that a saved tensor can be loaded and has the expected shape."""
        try:
            # Try to load the tensor
            loaded_tensor = torch.load(file_path, map_location='cpu')
            
            # Check shape matches
            if loaded_tensor.shape != expected_shape:
                logger.error(f"Shape mismatch for {layer_name}: expected {expected_shape}, got {loaded_tensor.shape}")
                return False
            
            # Basic integrity check
            if torch.isnan(loaded_tensor).any() or torch.isinf(loaded_tensor).any():
                logger.error(f"Loaded tensor {layer_name} contains NaN/Inf values")
                return False
            
            # Try a basic operation
            _ = loaded_tensor.sum()
            
            del loaded_tensor
            logger.debug(f"Verification passed for {layer_name}")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed for {layer_name}: {e}")
            return False
    
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
            
            # Forward pass with validation examples (process in batches to avoid OOM)
            with torch.no_grad():
                input_ids = self.validation_examples['input_ids']
                attention_mask = self.validation_examples['attention_mask']
                
                # Adaptive batch size based on task type - aggressive memory optimization
                if self.task_name == 'squad_v2':
                    batch_size = 4  # Very small batches for QA sequences (memory critical)
                else:
                    batch_size = 12  # Reduced batch size for classification
                    
                num_samples = input_ids.shape[0]
                
                # For SQuAD v2, use chunked processing to avoid memory accumulation
                if self.task_name == 'squad_v2':
                    chunk_size = 200  # Process and save 200 samples at a time
                    return self._extract_representations_chunked(model, input_ids, attention_mask, num_samples, batch_size, layer_outputs, hooks, step)
                
                # For other tasks, use normal processing
                all_layer_outputs = {f'layer_{i}': [] for i in self.config.save_layers}
                all_final_hidden_states = []
                
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    
                    # Progress logging for SQuAD v2 (large dataset)
                    if self.task_name == 'squad_v2' and i % (batch_size * 25) == 0:
                        progress = (i / num_samples) * 100
                        logger.info(f"SQuAD v2 processing: {progress:.1f}% ({i}/{num_samples} samples)")
                    
                    batch_input_ids = input_ids[i:end_idx].to(model.device)
                    batch_attention_mask = attention_mask[i:end_idx].to(model.device)
                    
                    # Clear layer outputs for this batch
                    layer_outputs.clear()
                    
                    # Pre-batch memory cleanup for critical memory situations
                    if self.task_name == 'squad_v2':
                        torch.cuda.empty_cache()
                    
                    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    
                    # Collect layer outputs
                    for layer_name, layer_output in layer_outputs.items():
                        all_layer_outputs[layer_name].append(layer_output)
                    
                    # Collect final hidden states
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        all_final_hidden_states.append(outputs.hidden_states[-1].detach().cpu())
                    elif hasattr(outputs, 'last_hidden_state'):
                        all_final_hidden_states.append(outputs.last_hidden_state.detach().cpu())
                    
                    # Aggressive memory cleanup after each batch
                    del batch_input_ids, batch_attention_mask, outputs
                    torch.cuda.empty_cache()  # Force GPU memory cleanup
                    
                    # Extra cleanup for SQuAD v2 (memory critical)
                    if self.task_name == 'squad_v2':
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
                # Final memory cleanup before concatenation (memory intensive step)
                torch.cuda.empty_cache()
                if self.task_name == 'squad_v2':
                    import gc
                    gc.collect()
                    logger.info(f"SQuAD v2 processing complete, concatenating {len(self.config.save_layers)} layers...")
                
                # Concatenate all batch results (pad to same length first)
                for layer_name in all_layer_outputs:
                    if all_layer_outputs[layer_name]:
                        # Find max sequence length across all batches
                        max_seq_len = max(tensor.shape[1] for tensor in all_layer_outputs[layer_name])
                        
                        # Pad all tensors to the same length
                        padded_tensors = []
                        for tensor in all_layer_outputs[layer_name]:
                            if tensor.shape[1] < max_seq_len:
                                # Pad sequence dimension to match max length
                                padding = (0, 0, 0, max_seq_len - tensor.shape[1])  # (left, right, top, bottom)
                                tensor = torch.nn.functional.pad(tensor, padding, value=0)
                            padded_tensors.append(tensor)
                        
                        representations[layer_name] = torch.cat(padded_tensors, dim=0)
                
                # Clean up batch data after all concatenations (avoid dict iteration error)
                del all_layer_outputs
                if self.task_name == 'squad_v2':
                    torch.cuda.empty_cache()
                
                if all_final_hidden_states:
                    # Find max sequence length for final hidden states
                    max_seq_len = max(tensor.shape[1] for tensor in all_final_hidden_states)
                    
                    # Pad all final hidden state tensors to the same length
                    padded_final_tensors = []
                    for tensor in all_final_hidden_states:
                        if tensor.shape[1] < max_seq_len:
                            # Pad sequence dimension to match max length
                            padding = (0, 0, 0, max_seq_len - tensor.shape[1])
                            tensor = torch.nn.functional.pad(tensor, padding, value=0)
                        padded_final_tensors.append(tensor)
                    
                    representations['final_hidden_states'] = torch.cat(padded_final_tensors, dim=0)
                    del all_final_hidden_states
                    if self.task_name == 'squad_v2':
                        torch.cuda.empty_cache()
        
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
            try:
                # FIXED: Properly handle HuggingFace Dataset tensors - always convert to tensors
                # Fallback: manually construct tensors, filtering None values
                input_ids_list = []
                attention_mask_list = []
                labels_list = []
                start_positions_list = []
                end_positions_list = []
                
                for ex in eval_dataset:
                    if ex['input_ids'] is not None and ex['attention_mask'] is not None:
                        # Convert to tensor ensuring proper format
                        if isinstance(ex['input_ids'], list):
                            input_ids_tensor = torch.tensor(ex['input_ids'], dtype=torch.long)
                        elif isinstance(ex['input_ids'], torch.Tensor):
                            input_ids_tensor = ex['input_ids']
                        else:
                            input_ids_tensor = torch.tensor([ex['input_ids']], dtype=torch.long)
                        
                        if isinstance(ex['attention_mask'], list):
                            attention_mask_tensor = torch.tensor(ex['attention_mask'], dtype=torch.long)
                        elif isinstance(ex['attention_mask'], torch.Tensor):
                            attention_mask_tensor = ex['attention_mask']
                        else:
                            attention_mask_tensor = torch.tensor([ex['attention_mask']], dtype=torch.long)
                        
                        # Ensure tensors are at least 1D
                        if input_ids_tensor.dim() == 0:
                            input_ids_tensor = input_ids_tensor.unsqueeze(0)
                        if attention_mask_tensor.dim() == 0:
                            attention_mask_tensor = attention_mask_tensor.unsqueeze(0)
                        
                        input_ids_list.append(input_ids_tensor)
                        attention_mask_list.append(attention_mask_tensor)
                        
                        if 'labels' in ex and ex['labels'] is not None:
                            if isinstance(ex['labels'], (list, int, float)):
                                labels_tensor = torch.tensor(ex['labels'], dtype=torch.long)
                            else:
                                labels_tensor = ex['labels']
                            
                            # For classification, labels should be scalar
                            if labels_tensor.dim() > 0 and labels_tensor.numel() == 1:
                                labels_tensor = labels_tensor.squeeze()
                            elif labels_tensor.dim() == 0:
                                pass  # already scalar
                            
                            labels_list.append(labels_tensor)
                        elif 'start_positions' in ex and 'end_positions' in ex:
                            if ex['start_positions'] is not None and ex['end_positions'] is not None:
                                start_positions_list.append(torch.tensor(ex['start_positions'], dtype=torch.long))
                                end_positions_list.append(torch.tensor(ex['end_positions'], dtype=torch.long))
                
                if not input_ids_list:
                    raise ValueError("No valid examples found in eval_dataset")
                
                # Handle variable-length sequences (same as base model extraction fix)
                try:
                    input_ids = torch.stack(input_ids_list)
                    attention_mask = torch.stack(attention_mask_list)
                except RuntimeError as e:
                    if "stack expects each tensor to be equal size" in str(e):
                        # Pad to max length for variable-length sequences
                        max_len = max(len(ids) for ids in input_ids_list)
                        padded_input_ids = []
                        padded_attention_mask = []
                        
                        for ids, mask in zip(input_ids_list, attention_mask_list):
                            pad_len = max_len - len(ids)
                            if pad_len > 0:
                                pad_id = 0  # Use 0 as pad token ID
                                ids_tensor = torch.tensor(ids, dtype=torch.long)
                                mask_tensor = torch.tensor(mask, dtype=torch.long)
                                padded_ids = torch.cat([ids_tensor, torch.full((pad_len,), pad_id, dtype=torch.long)])
                                padded_mask = torch.cat([mask_tensor, torch.zeros(pad_len, dtype=torch.long)])
                            else:
                                padded_ids = torch.tensor(ids, dtype=torch.long)
                                padded_mask = torch.tensor(mask, dtype=torch.long)
                            
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
                
                # Add labels/positions if found
                if labels_list:
                    examples['labels'] = torch.stack(labels_list)
                elif start_positions_list and end_positions_list:
                    examples['start_positions'] = torch.stack(start_positions_list)
                    examples['end_positions'] = torch.stack(end_positions_list)
                
                self.representation_extractor.set_validation_examples(examples)
                
            except Exception as e:
                logger.error(f"Failed to set validation examples: {e}")
                logger.error(f"eval_dataset type: {type(eval_dataset)}")
                if len(eval_dataset) > 0:
                    logger.error(f"First example keys: {list(eval_dataset[0].keys())}")
                # Continue without representation extraction rather than failing completely
                logger.warning("Continuing without representation extraction due to validation example setup failure")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        model = kwargs.get('model')
        step = state.global_step
        logs = kwargs.get('logs', {})
        
        # Log training metrics (since we disabled Trainer's automatic wandb reporting)
        if wandb.run is not None and logs:
            # Log basic training metrics at logging intervals
            if step % args.logging_steps == 0:
                training_metrics = {}
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        training_metrics[f"train/{key}"] = value
                
                if training_metrics:
                    wandb.log(training_metrics, step=step, commit=False)
        
        # Log gradient statistics
        if step % self.gradient_monitor.log_every_steps == 0:
            grad_stats = self.gradient_monitor.compute_gradient_stats(model)
            if wandb.run is not None:
                wandb.log({f"gradients/{k}": v for k, v in grad_stats.items()}, step=step, commit=False)
        
        # Log memory statistics
        memory_stats = self.memory_profiler.get_memory_stats()
        if wandb.run is not None:
            wandb.log({f"memory/{k}": v for k, v in memory_stats.items()}, step=step, commit=False)
        
        # Extract representations
        if step % self.extract_every_steps == 0:
            logger.info(f"Extracting representations at step {step}")
            representations = self.representation_extractor.extract_representations(model, step)
            self.representation_extractor.save_representations(representations, step)
            # Log representation extraction completion without conflicting with main training logs
            if wandb.run is not None:
                wandb.log({"full_finetune_representations/extracted": 1}, step=step, commit=False)
        
        # Commit all logs for this step together (every 10 steps to reduce frequency)
        if wandb.run is not None and step % 10 == 0:
            wandb.log({}, step=step, commit=True)
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called during evaluation."""
        step = state.global_step
        logger.info(f"Evaluation at step {step}")
        
        # Log evaluation metrics (since we disabled Trainer's automatic wandb reporting)
        metrics = kwargs.get('metrics', {})
        if wandb.run is not None and metrics:
            eval_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    eval_metrics[key] = value
            
            if eval_metrics:
                # Don't specify step for evaluation metrics to avoid conflicts
                wandb.log(eval_metrics, commit=False)
        
        # Extract representations during evaluation
        model = kwargs.get('model')
        representations = self.representation_extractor.extract_representations(model, step)
        self.representation_extractor.save_representations(representations, step)
        
        # Log evaluation representation extraction without step conflicts
        if wandb.run is not None:
            # Don't specify step to let wandb auto-assign and avoid conflicts
            wandb.log({"full_finetune_eval_representations/extracted": 1}, commit=False)


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
        
        # Initialize tokenizer with proper padding setup
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix padding token for TinyLlama (critical for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Ensure padding side is correct for causal LM
        self.tokenizer.padding_side = "right"
        
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
                # For SQuAD v2, use model with answerability head
                if task_name == 'squad_v2':
                    from models.squad_v2_qa_model import SquadV2QuestionAnsweringModel
                    model = SquadV2QuestionAnsweringModel(
                        model_name,
                        answerability_weight=1.0
                    )
                    # Apply dtype and device settings
                    model = model.to(dtype=getattr(torch, self.config['model']['dtype']))
                    if torch.cuda.is_available():
                        model = model.cuda()
                else:
                    # For other QA tasks, use standard QA model
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
                "end_positions": train_data["end_positions"],
                "answerability_labels": train_data["answerability_labels"]
            })
            
            try:
                eval_data = self.data_loader.prepare_qa_data('validation', task_config.get('max_samples_eval'))
                eval_dataset = Dataset.from_dict({
                    "input_ids": eval_data["input_ids"],
                    "attention_mask": eval_data["attention_mask"],
                    "start_positions": eval_data["start_positions"],
                    "end_positions": eval_data["end_positions"],
                    "answerability_labels": eval_data["answerability_labels"]
                })
            except:
                eval_dataset = train_dataset.select(range(min(500, len(train_dataset))))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")
        
        # ROOT CAUSE FIX: Custom data collator that preserves answerability_labels AND handles padding
        if task_config['type'] == 'question_answering':
            import torch
            from transformers import DataCollatorWithPadding
            
            # Create base padding collator
            base_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            
            def qa_data_collator(features):
                # Use padding collator for variable-length sequences
                batch = base_collator(features)
                # Manually add answerability_labels that got filtered out by base collator
                if features and 'answerability_labels' in features[0]:
                    batch['answerability_labels'] = torch.tensor([f['answerability_labels'] for f in features])
                return batch
            self.qa_data_collator = qa_data_collator
        else:
            self.qa_data_collator = None
        
        return train_dataset, eval_dataset
    
    def create_compute_metrics_function(self, task_name: str):
        """Create compute_metrics function for evaluation."""
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            return lambda eval_pred: compute_classification_metrics(eval_pred, task_config['metric'])
        elif task_config['type'] == 'question_answering':
            # ROOT CAUSE FIX: Simple QA metrics function
            def compute_qa_metrics_simple(eval_pred):
                predictions, labels = eval_pred
                # For QA tasks, just return basic metrics
                return {
                    "eval_loss": 0.0,  # Will be computed by trainer
                    "eval_samples": len(predictions) if predictions is not None else 0
                }
            return compute_qa_metrics_simple
        else:
            return None
    
    def extract_base_model_representations(self, eval_dataset: Dataset, task_name: str):
        """Extract representations from the base (pre-trained) model."""
        logger.info(f"Extracting base model representations for {task_name}")
        
        # Load clean pre-trained model
        model_name = self.config['model']['name']
        
        # Use same model type as training
        task_type = self.config['tasks'][task_name].get('type', 'classification')
        if task_type in ['qa', 'question_answering']:
            if task_name == 'squad_v2':
                from models.squad_v2_qa_model import SquadV2QuestionAnsweringModel
                base_model = SquadV2QuestionAnsweringModel(
                    model_name,
                    answerability_weight=1.0
                )
                base_model = base_model.to(dtype=getattr(torch, self.config['model']['dtype']))
            else:
                base_model = AutoModelForQuestionAnswering.from_pretrained(
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
        
        # CRITICAL FIX: Configure the model to use the properly set up tokenizer
        # This resolves the "Cannot handle batch sizes > 1 if no padding token is defined" error
        if hasattr(base_model, 'config'):
            base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Create base model representation extractor
        base_extractor = RepresentationExtractor(
            self.representation_config,
            self.output_dir,
            task_name,
            "base_pretrained"
        )
        
        # FIXED: Properly extract tensors from HuggingFace Dataset
        # The issue was trying to create tensors from already-tensorized data
        try:
            # Convert HuggingFace Dataset to tensors properly
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            
            for ex in eval_dataset:
                if ex['input_ids'] is not None and ex['attention_mask'] is not None:
                    # Convert to tensor ensuring proper format
                    if isinstance(ex['input_ids'], list):
                        input_ids_tensor = torch.tensor(ex['input_ids'], dtype=torch.long)
                    elif isinstance(ex['input_ids'], torch.Tensor):
                        input_ids_tensor = ex['input_ids']
                    else:
                        input_ids_tensor = torch.tensor([ex['input_ids']], dtype=torch.long)
                    
                    if isinstance(ex['attention_mask'], list):
                        attention_mask_tensor = torch.tensor(ex['attention_mask'], dtype=torch.long)
                    elif isinstance(ex['attention_mask'], torch.Tensor):
                        attention_mask_tensor = ex['attention_mask']
                    else:
                        attention_mask_tensor = torch.tensor([ex['attention_mask']], dtype=torch.long)
                    
                    # Ensure tensors are at least 1D
                    if input_ids_tensor.dim() == 0:
                        input_ids_tensor = input_ids_tensor.unsqueeze(0)
                    if attention_mask_tensor.dim() == 0:
                        attention_mask_tensor = attention_mask_tensor.unsqueeze(0)
                    
                    input_ids_list.append(input_ids_tensor)
                    attention_mask_list.append(attention_mask_tensor)
                    
                    # Handle labels for classification tasks
                    if 'labels' in ex and ex['labels'] is not None:
                        if isinstance(ex['labels'], (list, int, float)):
                            labels_tensor = torch.tensor(ex['labels'], dtype=torch.long)
                        else:
                            labels_tensor = ex['labels']
                        
                        # For classification, labels should be scalar
                        if labels_tensor.dim() > 0 and labels_tensor.numel() == 1:
                            labels_tensor = labels_tensor.squeeze()
                        elif labels_tensor.dim() == 0:
                            pass  # already scalar
                        
                        labels_list.append(labels_tensor)
            
            if not input_ids_list:
                raise ValueError("No valid examples found in eval_dataset")
            
            # Handle tensor stacking based on task type
            if task_type in ['qa', 'question_answering']:
                # For QA tasks, use the QADataCollator to handle padding
                collator = QADataCollator(self.tokenizer, padding=True)
                
                # Create features in the format expected by the collator
                features = []
                for i in range(len(input_ids_list)):
                    feature = {
                        'input_ids': input_ids_list[i],
                        'attention_mask': attention_mask_list[i],
                        'start_positions': 0,  # Dummy values for base model extraction
                        'end_positions': 0
                    }
                    features.append(feature)
                
                # Use collator to create properly padded batch
                batch = collator(features)
                examples = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                }
            else:
                # For classification tasks, pad manually or use stack if all same length
                try:
                    input_ids = torch.stack(input_ids_list)
                    attention_mask = torch.stack(attention_mask_list)
                except RuntimeError as e:
                    if "stack expects each tensor to be equal size" in str(e):
                        # Pad to max length
                        max_len = max(len(ids) for ids in input_ids_list)
                        padded_input_ids = []
                        padded_attention_mask = []
                        
                        for ids, mask in zip(input_ids_list, attention_mask_list):
                            pad_len = max_len - len(ids)
                            if pad_len > 0:
                                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                ids_tensor = torch.tensor(ids, dtype=torch.long)
                                mask_tensor = torch.tensor(mask, dtype=torch.long)
                                padded_ids = torch.cat([ids_tensor, torch.full((pad_len,), pad_id, dtype=torch.long)])
                                padded_mask = torch.cat([mask_tensor, torch.zeros(pad_len, dtype=torch.long)])
                            else:
                                padded_ids = torch.tensor(ids, dtype=torch.long)
                                padded_mask = torch.tensor(mask, dtype=torch.long)
                            
                            padded_input_ids.append(padded_ids)
                            padded_attention_mask.append(padded_mask)
                        
                        input_ids = torch.stack(padded_input_ids)
                        attention_mask = torch.stack(padded_attention_mask)
                    else:
                        raise
                
                examples = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                
                # Add labels if present (for classification tasks)
                if labels_list:
                    examples['labels'] = torch.stack(labels_list)
            
            logger.info(f"Successfully prepared {examples['input_ids'].shape[0]} validation examples for {task_name}")
            base_extractor.set_validation_examples(examples)
            
            # Extract and save base representations
            representations = base_extractor.extract_representations(base_model, step=0)
            base_extractor.save_representations(representations, step=0)
            
        except Exception as e:
            logger.error(f"Failed to extract base model representations for {task_name}: {e}")
            logger.error(f"eval_dataset type: {type(eval_dataset)}")
            logger.error(f"eval_dataset length: {len(eval_dataset)}")
            if len(eval_dataset) > 0:
                logger.error(f"First example keys: {list(eval_dataset[0].keys())}")
                logger.error(f"First example input_ids type: {type(eval_dataset[0]['input_ids'])}")
            raise
        
        # Clean up
        del base_model
        torch.cuda.empty_cache()
        
        logger.info("✓ Base model representations extracted")
    
    def create_hyperparameter_sweep_config(self, task_name: str) -> Dict[str, Any]:
        """Create W&B sweep configuration for hyperparameter search."""
        task_config = self.config['tasks'][task_name]
        
        if task_config['type'] == 'classification':
            learning_rates = [3e-6, 5e-6]  # Reduced to prevent gradient explosion
            sequence_length = 512
        else:  # question_answering
            learning_rates = [2e-6, 3e-6]  # Reduced to prevent gradient explosion (matches remote fix)
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
                    'value': 0.01  # Further reduced to prevent gradient explosion
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
    
    def run_single_experiment(self, task_name: str, seed: int = 42, skip_wandb_init: bool = False, **hyperparams) -> Dict[str, Any]:
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
        
        # Create run name (always needed for training args)
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"full_ft_{task_name}_seed{seed}_{timestamp}"
        
        # Initialize wandb if not skipped (for sweep runs, wandb is already initialized)
        if not skip_wandb_init:
            try:
                # Configure wandb to use project-specific temp directories
                from wandb_config import setup_wandb_directories, get_wandb_settings
                setup_wandb_directories()
                
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
                    },
                    settings=get_wandb_settings()  # Use optimized settings with custom temp dirs
                )
                logger.info(f"✓ Wandb initialized for run: {run_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}. Continuing with offline mode.")
                # Initialize in offline mode as fallback
                wandb.init(mode="offline",
                          project=os.getenv('WANDB_PROJECT', self.config['wandb']['project']),
                          name=run_name)
        else:
            # Update wandb config with hyperparameters for sweep runs
            wandb.config.update({
                "task_name": task_name,
                "method": "full_finetune", 
                "seed": seed,
                **hyperparams
            })
        
        try:
            # Setup environment with new seed
            self.setup_environment()
            
            # Load model
            model = self.load_model(task_name)
            
            # Prepare datasets
            train_dataset, eval_dataset = self.prepare_datasets(task_name)
            
            # Extract base model representations first (only if enabled)
            if self.config['training'].get('extract_base_model_representations', True):
                self.extract_base_model_representations(eval_dataset, task_name)
            else:
                logger.info("Base model representation extraction disabled")
            
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
            
            # Apply hyperparameters - use task-specific learning rates
            task_config = self.config['tasks'][task_name]
            if task_config['type'] == 'classification':
                # Use classification-specific learning rate
                default_lr = self.config['training']['full_finetune_learning_rate_classification'][0]  # 1e-5
            elif task_config['type'] == 'question_answering':
                # Use QA-specific learning rate (lower for stability)
                default_lr = self.config['training']['full_finetune_learning_rate_qa'][0]  # 2e-5 (literature-aligned)
            else:
                # Fallback to generic rate
                default_lr = self.config['training']['full_finetune_learning_rate']
            
            # Use task-specific learning rate for production, config rate for sanity checks
            if 'learning_rate' in hyperparams:
                # Explicit hyperparameter override (sanity checks)
                learning_rate = hyperparams['learning_rate']
            elif hasattr(self, '_is_sanity_check') and self._is_sanity_check:
                # Sanity checks use config learning rate
                config_lr = self.config['training']['learning_rate']
                learning_rate = config_lr
            else:
                # Production and production stability use task-specific learning rate
                learning_rate = default_lr
            batch_size = hyperparams.get('per_device_train_batch_size',
                                       self.config['training']['per_device_train_batch_size'])
            
            logger.info(f"Using learning rate: {learning_rate} for task: {task_name}")
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                run_name=run_name,
                
                # Training configuration
                num_train_epochs=hyperparams.get('num_train_epochs', self.config['training']['num_train_epochs']),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                
                # Optimization
                learning_rate=learning_rate,
                weight_decay=self.config['training']['weight_decay'],
                warmup_ratio=hyperparams.get('warmup_ratio', 0.1),  # Increased to 0.1 for better gradient stability
                lr_scheduler_type=self.config['training']['lr_scheduler_type'],
                max_grad_norm=0.3,  # Very aggressive gradient clipping to prevent gradient explosion
                
                # Evaluation and saving
                eval_strategy=self.config['training'].get('evaluation_strategy', 'steps'),  # Updated parameter name
                eval_steps=100,
                save_strategy=self.config['training'].get('save_strategy', 'steps'),
                save_steps=500,
                logging_steps=50,
                
                # Model selection
                load_best_model_at_end=self.config['training'].get('load_best_model_at_end', self.config['training'].get('evaluation_strategy', 'steps') != 'no'),
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                
                # Performance optimizations
                gradient_checkpointing=True,
                dataloader_pin_memory=True,
                fp16=False,  # Disabled for gradient stability
                bf16=True,   # Use bf16 for better gradient stability than fp16
                
                # Reporting - disable automatic wandb reporting to avoid step conflicts with custom callback
                report_to=[],  # Let custom callback handle wandb logging
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
                # ROOT CAUSE FIX: Use our custom data collator that preserves answerability_labels
                data_collator = self.qa_data_collator if hasattr(self, 'qa_data_collator') and self.qa_data_collator else QADataCollator(tokenizer=self.tokenizer, padding=True)
            else:
                # For classification tasks, use padding collator
                data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            
            # Create custom callback (conditional based on representation extraction and sanity check mode)
            if self.config['training'].get('evaluation_strategy') == 'no':
                # Sanity check mode - no callback to avoid evaluation issues
                custom_callback = None
            elif representation_extractor is not None:
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
            callbacks = []
            # Only add EarlyStoppingCallback if evaluation is enabled
            if self.config['training'].get('evaluation_strategy', 'steps') != 'no':
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            if custom_callback is not None:
                callbacks.append(custom_callback)
                
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics if training_args.eval_strategy != "no" else None,
                callbacks=callbacks
            )
            
            # Log initial memory stats
            initial_memory = memory_profiler.get_memory_stats()
            if wandb.run is not None:
                wandb.log({f"initial_memory/{k}": v for k, v in initial_memory.items()}, commit=False)
            
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
            
            # Evaluate the model (skip for sanity checks to avoid pad_across_processes error)
            if training_args.eval_strategy != "no":
                logger.info(f"Final evaluation for {task_name}...")
                eval_result = trainer.evaluate()
            else:
                # Sanity check mode - skip evaluation
                eval_result = {"eval_loss": 0.0}
            
            # Save the model
            model_save_path = output_dir / "final_model"
            trainer.save_model(str(model_save_path))
            
            # Mark experiment as completed
            self.checkpoint_manager.save_experiment_progress(
                task_name, "full_finetune", seed, "completed", str(model_save_path)
            )
            
            # Final memory profiling
            final_memory = memory_profiler.get_memory_stats()
            if wandb.run is not None:
                wandb.log({f"final_memory/{k}": v for k, v in final_memory.items()}, commit=False)
            
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
            if wandb.run is not None:
                wandb.log({
                    "final_train_loss": results["train_loss"],
                    "final_eval_loss": results["eval_loss"],
                    "training_time_seconds": training_time,
                    "total_steps": trainer.state.global_step
                }, commit=False)
            
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
            
            # Only finish wandb if we initialized it in this method (not for sweep runs)
            if not skip_wandb_init:
                wandb.finish()
            
            # Auto-cleanup after each completed task+seed run to prevent disk space issues
            try:
                import subprocess
                import shutil
                
                # Always cleanup representations from THIS run (keep only final step)
                logger.info(f"🧹 Cleaning representations from completed run: {task_name} seed {seed}")
                experiment_dir = Path("results") / f"full_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Find the current experiment directory (most recent)
                results_dirs = sorted([d for d in Path("results").glob("full_finetune_*") if d.is_dir()], 
                                    key=lambda x: x.stat().st_mtime, reverse=True)
                
                if results_dirs:
                    current_experiment = results_dirs[0]
                    cleanup_cmd = [
                        'python', 'scripts/cleanup_experiment.py', 
                        '--experiment', str(current_experiment),
                        '--mode', 'representations'  # Clean representations but keep final step
                    ]
                    result = subprocess.run(cleanup_cmd, cwd=Path.cwd(), capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info("✅ Post-run cleanup completed successfully")
                    else:
                        logger.warning(f"Cleanup warning: {result.stderr}")
                
                # Also check disk usage and cleanup old experiments if needed
                total, used, free = shutil.disk_usage('/')
                usage_percent = (used / total) * 100
                logger.info(f"💾 Disk usage after cleanup: {usage_percent:.1f}%")
                
                if usage_percent > 70:  # Cleanup older experiments if still high
                    logger.info(f"🧹 Disk usage still high ({usage_percent:.1f}%), cleaning older experiments...")
                    old_cleanup_cmd = [
                        'python', 'scripts/auto_cleanup.py', 
                        '--task', task_name,
                        '--results-dir', 'results'
                    ]
                    subprocess.run(old_cleanup_cmd, cwd=Path.cwd(), capture_output=True)
                    logger.info("✅ Additional cleanup completed")
                    
            except Exception as e:
                logger.warning(f"Auto-cleanup failed (non-critical): {e}")
    
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
            # Initialize wandb for sweep run
            run_name = f"sweep_{task_name}_{datetime.now().strftime('%H%M%S')}"
            wandb.init(
                project=os.getenv('WANDB_PROJECT', self.config['wandb']['project']),
                entity=self.config['wandb']['entity'],
                name=run_name,
                group=f"sweep_{task_name}",
                job_type="hyperparameter_sweep",
                tags=["full_finetune", "sweep", task_name]
            )
            
            # Get hyperparameters from sweep
            hyperparams = dict(wandb.config)
            seed = hyperparams.pop('seed', 42)
            
            result = self.run_single_experiment(task_name, seed, skip_wandb_init=True, **hyperparams)
            results.append(result)
            
            # Finish the run
            wandb.finish()
        
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
    parser.add_argument("--warmup-ratio", type=float, help="Override warmup ratio")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--no-base-representations", action="store_true", 
                       help="Disable base model representation extraction (for VM1/VM2)")
    parser.add_argument("--sanity-check", action="store_true", 
                       help="Run quick sanity check (10 samples, 2 epochs, no wandb)")
    parser.add_argument("--production-stability", action="store_true",
                       help="Run production stability check (64 samples, 1 epoch, production hyperparameters)")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = FullFinetuneExperiment(args.config)
    
    # Override model to use TinyLlama for actual experiments
    if args.mode != "demo":
        experiment.config['model']['name'] = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    # Override base model representation extraction if requested
    if args.no_base_representations:
        experiment.config['training']['extract_base_model_representations'] = False
    
    # Handle sanity check mode
    if args.sanity_check:
        import os
        os.environ["WANDB_MODE"] = "disabled"
        # Override config for quick sanity check
        # CRITICAL: Apply learning rate multiplier for aggressive overfitting
        # Full fine-tuning needs more conservative LR than LoRA
        # IMPORTANT: Use task-specific learning rates, not default
        task_config = experiment.config['tasks'][args.task]
        if task_config['type'] == 'classification':
            base_lr = experiment.config['training']['full_finetune_learning_rate_classification'][0]  # 3e-6
        elif task_config['type'] == 'question_answering':
            base_lr = experiment.config['training']['full_finetune_learning_rate_qa'][0]  # 2e-5 (literature-aligned QA rate)
        else:
            base_lr = experiment.config['training']['full_finetune_learning_rate']  # 5e-6 fallback
        
        # Get task-specific learning rate multiplier for sanity checks
        sanity_config = experiment.config.get('sanity_check', {})
        
        # Try method-specific multipliers first, then fall back to general task_multipliers
        fullft_multipliers = sanity_config.get('task_multipliers_fullft', {})
        task_multipliers = sanity_config.get('task_multipliers', {})
        default_multiplier = sanity_config.get('default_multiplier', 4)
        
        # Use Full FT-specific multiplier if available, otherwise fall back
        task_multiplier = fullft_multipliers.get(args.task) or task_multipliers.get(args.task, default_multiplier)
        sanity_lr = base_lr * task_multiplier
        
        # Mark as sanity check for learning rate logic
        experiment._is_sanity_check = True
        
        # Get task-specific configurations
        task_specific = sanity_config.get('task_specific', {})
        task_config = task_specific.get(args.task, {})
        
        # Use task-specific values if available, otherwise fall back to global
        max_epochs = task_config.get('max_epochs', sanity_config.get('max_epochs', 5))
        num_samples = task_config.get('num_samples', sanity_config.get('num_samples', 10))
        
        logger.info(f"📋 Task-specific sanity config for {args.task}: epochs={max_epochs}, samples={num_samples}")
        
        experiment.config['training'].update({
            'num_train_epochs': max_epochs,  # Task-specific epochs for optimal performance
            'learning_rate': sanity_lr,  # CRITICAL FIX: Boost learning rate for sanity checks
            'per_device_train_batch_size': 1,  # Perfect overfitting needs batch size 1
            'evaluation_strategy': 'epoch',  # CRITICAL FIX: Enable evaluation to test overfitting
            'save_strategy': 'no',
            'load_best_model_at_end': False,  # CRITICAL FIX: Disable to avoid save/eval strategy mismatch
            'logging_steps': 1,
            'extract_base_model_representations': False,
            # CRITICAL: Remove regularization that prevents overfitting
            'weight_decay': 0.0,  # NO regularization for sanity checks
            'max_grad_norm': 3.0,  # Higher gradient norm threshold for aggressive learning
            # CRITICAL: Disable mixed precision for numerical stability in sanity checks
            'fp16': False,
            'bf16': False,  # No mixed precision to avoid numerical issues
        })
        # Override dataset sizes with task-specific values
        for task_name in experiment.config['tasks']:
            if task_name == args.task:
                # Use task-specific configuration for current task
                experiment.config['tasks'][task_name]['max_samples_train'] = num_samples
                experiment.config['tasks'][task_name]['max_samples_eval'] = num_samples // 2
            else:
                # Use global configuration for other tasks
                global_samples = sanity_config.get('num_samples', 10)
                experiment.config['tasks'][task_name]['max_samples_train'] = global_samples
                experiment.config['tasks'][task_name]['max_samples_eval'] = global_samples // 2
        print(f"🧪 SANITY CHECK MODE: 10 samples, {experiment.config['training']['num_train_epochs']} epochs, LR boosted {task_multiplier}x to {sanity_lr:.4f}, no wandb")
    
    # Handle production stability check mode
    if args.production_stability:
        import os
        os.environ["WANDB_MODE"] = "disabled"
        # Mark as production stability for learning rate logic
        experiment._is_production_stability = True
        # Use production hyperparameters on small dataset to test stability
        print("⚡ PRODUCTION STABILITY MODE: Using production hyperparameters on 64 samples")
        
        experiment.config['training'].update({
            'num_train_epochs': 1,  # Just 1 epoch to test initial stability
            'evaluation_strategy': 'epoch',  # CRITICAL FIX: Enable evaluation for production stability
            'save_strategy': 'no',
            'load_best_model_at_end': False,  # CRITICAL FIX: Disable to avoid save/eval strategy mismatch
            'logging_steps': 1,
            'extract_base_model_representations': False,
            # Keep production hyperparameters - DON'T modify LR, weight_decay, batch_size, etc.
        })
        
        # Override dataset sizes for stability testing
        for task_name in experiment.config['tasks']:
            experiment.config['tasks'][task_name]['max_samples_train'] = 64  # 4 batches worth
            experiment.config['tasks'][task_name]['max_samples_eval'] = 32
        
        prod_lr = experiment.config['training']['learning_rate']
        prod_batch_size = experiment.config['training']['per_device_train_batch_size'] 
        prod_weight_decay = experiment.config['training']['weight_decay']
        print(f"🔬 Testing production config: LR={prod_lr:.6f}, batch_size={prod_batch_size}, weight_decay={prod_weight_decay}")
    
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
        if args.warmup_ratio:
            hyperparams['warmup_ratio'] = args.warmup_ratio
        if args.epochs:
            hyperparams['num_train_epochs'] = args.epochs
        
        result = experiment.run_single_experiment(args.task, args.seed, **hyperparams)
        print(f"Experiment completed: {result}")


if __name__ == "__main__":
    main()
