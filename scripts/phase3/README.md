# Phase 3: Post-Training Representation Extraction

## Overview

Phase 3 extracts layer-wise representations from models trained in Phase 2 for drift analysis in Phase 4.

## Why Separate Extraction from Training?

**Memory Efficiency:**
- Representation extraction during training consumes 12-15GB GPU memory
- This would force `batch_size=1`, making training 4x slower
- Separating extraction allows efficient training (batch_size=2-4) in Phase 2

**Methodological Soundness:**
- Representations are properties of trained models, not training artifacts
- Post-training extraction is standard practice in representation analysis research
- Produces identical results to in-training extraction
- Enables memory-efficient layer-by-layer processing

## Prerequisites

- Phase 2 completed (18 trained models in `results/`)
- Base representations extracted (`base_representations/` from Phase 0)
- Config file: `shared/config.yaml`

## Execution

```bash
cd scripts/phase3

# Extract representations from all 18 models (~8 hours)
bash vm1.sh
```

This will extract representations from:
- 9 Full fine-tuned models (MRPC, SST-2, RTE × 3 seeds each)
- 9 LoRA adapters (MRPC, SST-2, RTE × 3 seeds each)

## What Gets Extracted

For each model:
- 22 transformer layers (layer_0.pt through layer_21.pt)
- Final hidden states
- Metadata (model config, extraction settings)

**Extraction settings:**
- 750 validation samples per task (consistent with training)
- `max_length=384` (matches Phase 2 training configuration)
- Layer-by-layer processing with immediate disk save

## Output Structure

```
results/phase3_representations/representations/
├── full_finetune_mrpc_seed42/
│   └── step_000000/
│       ├── layer_0.pt through layer_21.pt
│       ├── final_hidden_states.pt
│       └── metadata.json
├── lora_mrpc_seed42/
│   └── step_000000/
│       ├── layer_0.pt through layer_21.pt
│       └── ...
└── ... (18 directories total)
```

**Total size:** ~12GB for all representations

## Memory Requirements

**During Extraction:**
- No training overhead (no gradients, no optimizer states)
- Layer-by-layer processing with immediate disk save
- Memory usage: ~8-12GB (much lower than training)

## Next Steps

Proceed to Phase 4 for drift analysis (CKA/cosine similarity computation).
