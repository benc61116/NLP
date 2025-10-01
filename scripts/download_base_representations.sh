#!/bin/bash
# Download base representations from WandB artifacts when needed for Phase 3

echo "📥 Downloading base representations from WandB..."

python -c "
import wandb

wandb.login()
run = wandb.init(project='NLP-Phase0', entity='galavny-tel-aviv-university')

# Download the artifact
artifact = run.use_artifact('galavny-tel-aviv-university/NLP-Phase0/base_representations:latest')
artifact_dir = artifact.download(root='base_representations')

print(f'✅ Base representations downloaded to: {artifact_dir}')
wandb.finish()
"

echo "🎉 Download complete!"
