#!/usr/bin/env python3
"""Quick debug script to see why LoRA representations aren't being saved."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from experiments.lora_finetune import LoRAExperiment, LoRARepresentationExtractor
from dataclasses import dataclass

@dataclass
class RepresentationConfig:
    extract_every_steps: int
    save_layers: list
    max_validation_samples: int
    save_attention: bool
    save_mlp: bool
    memory_map: bool
    save_adapter_weights: bool = False
    analyze_rank_utilization: bool = False

# Initialize experiment
experiment = LoRAExperiment(config_path="shared/config.yaml")

# Create representation config
rep_config = RepresentationConfig(
    extract_every_steps=100,
    save_layers=list(range(22)),
    max_validation_samples=10,  # Just 10 samples for testing
    save_attention=False,
    save_mlp=True,
    memory_map=True
)
rep_config.save_adapter_weights = False
rep_config.analyze_rank_utilization = False

# Create extractor
output_dir = Path("results/debug_lora_test")
extractor = LoRARepresentationExtractor(
    config=rep_config,
    output_dir=output_dir,
    task_name="mrpc",
    method="lora_seed42"
)

print(f"✓ Extractor created: {extractor.output_dir}")

# Load a LoRA model
model_path = Path("results/lora_finetune_20251003_050941/lora_mrpc_manual_seed42/final_adapter")
print(f"Loading model from: {model_path}")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16
)
base_model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base_model, str(model_path))
model.config.pad_token_id = tokenizer.pad_token_id

print(f"✓ Model loaded: {type(model)}")

# Create dummy validation examples
dummy_input_ids = torch.randint(0, 1000, (10, 128))
dummy_attention_mask = torch.ones((10, 128))
dummy_labels = torch.randint(0, 2, (10,))

extractor.set_validation_examples({
    'input_ids': dummy_input_ids,
    'attention_mask': dummy_attention_mask,
    'labels': dummy_labels
})

print(f"✓ Validation examples set: {len(extractor.validation_examples['input_ids'])} samples")

# Extract representations
print("Extracting representations...")
model = model.cpu()  # Move to CPU for testing
representations = extractor.extract_lora_representations(model, step=0)

print(f"\n✓ Extraction complete!")
print(f"  Number of representation tensors: {len(representations)}")
print(f"  Keys: {list(representations.keys())[:5]}")

if representations:
    print("\nSaving representations...")
    extractor.save_representations(representations, step=0)
    print(f"✓ Saved to: {extractor.output_dir}")
    
    # Check if files exist
    saved_files = list(extractor.output_dir.glob("**/*.pt"))
    print(f"  Files saved: {len(saved_files)}")
else:
    print("\n❌ NO REPRESENTATIONS EXTRACTED!")
    print("This is the bug - extract_lora_representations returned an empty dict")

