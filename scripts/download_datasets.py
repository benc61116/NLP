#!/usr/bin/env python3
"""Download all datasets for the NLP project."""

import os
from datasets import load_dataset
import json

def download_datasets():
    """Download all required datasets and save them locally."""
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    print("Downloading GLUE-MRPC...")
    mrpc = load_dataset("glue", "mrpc")
    mrpc.save_to_disk("data/mrpc")
    
    print("Downloading GLUE-SST2...")
    sst2 = load_dataset("glue", "sst2") 
    sst2.save_to_disk("data/sst2")
    
    print("Downloading GLUE-RTE...")
    rte = load_dataset("glue", "rte")
    rte.save_to_disk("data/rte")
    
    # Create a manifest file with dataset info
    manifest = {
        "datasets": {
            "mrpc": {
                "name": "GLUE-MRPC",
                "task": "paraphrase detection",
                "train_size": len(mrpc["train"]),
                "validation_size": len(mrpc["validation"]),
                "test_size": len(mrpc["test"])
            },
            "sst2": {
                "name": "GLUE-SST2", 
                "task": "sentiment analysis",
                "train_size": len(sst2["train"]),
                "validation_size": len(sst2["validation"]),
                "test_size": len(sst2["test"])
            },
            "rte": {
                "name": "GLUE-RTE",
                "task": "textual entailment", 
                "train_size": len(rte["train"]),
                "validation_size": len(rte["validation"]),
                "test_size": len(rte["test"])
            }
        }
    }
    
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("All datasets downloaded successfully!")
    print("Dataset manifest saved to data/manifest.json")

if __name__ == "__main__":
    download_datasets()
