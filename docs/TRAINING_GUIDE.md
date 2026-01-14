# Madrigal Training Guide

This guide covers training Madrigal from scratch or fine-tuning with pre-trained checkpoints.

## Training Pipeline Overview

```
Stage 1: Modality Pretraining (Optional - checkpoints provided)
    ↓
Stage 2: Contrastive Alignment (SimCLR)
    ↓
Stage 3: DDI Fine-tuning
    ↓
Inference/Integration
```

## Stage 1: Modality-Specific Pretraining

Pre-trained checkpoints are provided in `modality_pretraining/`. Only retrain if:
- You have new/different data sources
- You want to experiment with encoder architectures

### Structure Encoder (GIN)
```bash
cd modality_pretraining/str
python structure_pretraining_muv.py \
    --gin_hidden_dims 256 256 256 256 \
    --gin_num_mlp_layer 3 \
    --gin_batch_norm
```

### Knowledge Graph Encoder (HGT)
```bash
cd modality_pretraining/kg
python kg_pretraining.py
```

### Cell Viability Encoder
See notebook: `modality_pretraining/cv/`

### Transcriptomics Encoder (ChemCPA)
```bash
# First generate RDKit embeddings (see chemcpa.ipynb)
cd modality_pretraining/tx
python sweep.py  # or run chemcpa.ipynb
```

## Stage 2: Contrastive Pretraining (Multimodal Alignment)

This aligns embeddings from different modalities using SimCLR.

```bash
# Using provided config
python pretrain.py --from_yaml configs/cl_pretrain/pretrain_drugbank_shared_basal.yaml

# Key parameters:
# --pretrain_mode: 'str_center' (structure as anchor) or 'double_random'
# --temperature: contrastive loss temperature (default: 0.1)
# --pretrain_epochs: number of epochs (default: 1000)
# --batch_size: batch size for pretraining
```

### Config Example (`configs/cl_pretrain/pretrain_drugbank_shared_basal.yaml`):
```yaml
data_source: DrugBank
split_method: split_by_pairs
pretrain_mode: str_center
temperature: 0.1
pretrain_epochs: 1000
batch_size: 256
lr: 1e-4
```

Output: `checkpoint_*.pt` in `CL_CKPT_DIR/DrugBank/split_by_*/`

## Stage 3: DDI Fine-tuning

### Without Pretraining (Train from Scratch)
```bash
python train_ddi_batch.py \
    --from_yaml configs/ddi_finetune/DrugBank/sweep_config_*.yaml \
    --split_method split_by_pairs \
    --finetune_mode str_str+random_sample \
    --seed 42
```

### With Pretrained Checkpoint (Recommended)
```bash
python train_ddi_batch.py \
    --checkpoint checkpoint_1000.pt \
    --from_yaml configs/ddi_finetune/DrugBank/sweep_config_*.yaml \
    --split_method split_by_pairs \
    --finetune_mode str_str+random_sample \
    --seed 42
```

### Key Fine-tuning Parameters
| Parameter | Description |
|-----------|-------------|
| `--checkpoint` | Pretrained checkpoint from Stage 2 |
| `--finetune_mode` | Modality combination (e.g., `str_str+random_sample`, `full_full`) |
| `--split_method` | Data split strategy |
| `--lr` | Learning rate (default: 1e-4) |
| `--epochs` | Training epochs (default: 100) |
| `--batch_size` | Batch size |

### Finetune Modes
- `str_str`: Structure only
- `str_full`: Structure + all other modalities
- `full_full`: All modalities for both drugs
- `ablation_str_str`: Ablation study mode

## Using Pre-trained Models for Inference

### Quick Inference (Notebook)
See `notebooks/quick_predictions.ipynb` for interactive predictions.

### Programmatic Inference
```python
import torch
from madrigal.evaluate.predict import test
from madrigal.data.data import get_test_data

# Load checkpoint
checkpoint_dir = "path/to/model_output/DrugBank/split_by_pairs/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run inference
results = test(checkpoint_dir, device=device)
```

### Generate Embeddings
```python
from madrigal.evaluate.predict import get_all_drug_embeddings

embeddings = get_all_drug_embeddings(
    checkpoint_dir="path/to/checkpoint/",
    device=device
)
```

## Integration into Your Product

### 1. Minimal Inference API
```python
import torch
from madrigal.models.models import NovelDDIEncoder
from madrigal.data.data import get_test_data

class MadrigalPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model from checkpoint
        self.model = NovelDDIEncoder(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, drug1_idx, drug2_idx):
        """Predict DDI outcomes for a drug pair."""
        with torch.no_grad():
            # Get predictions
            scores = self.model.predict_pair(drug1_idx, drug2_idx)
        return scores

# Usage
predictor = MadrigalPredictor("best_model.pt", device="cuda")
scores = predictor.predict(drug1_idx=0, drug2_idx=1)
```

### 2. Batch Prediction
```python
def batch_predict(self, drug_pairs):
    """
    drug_pairs: List of (drug1_idx, drug2_idx) tuples
    """
    results = []
    for d1, d2 in drug_pairs:
        scores = self.predict(d1, d2)
        results.append(scores)
    return torch.stack(results)
```

### 3. Export to ONNX (Optional)
```python
# For deployment without PyTorch
dummy_input = (torch.zeros(1, ...), torch.zeros(1, ...))
torch.onnx.export(model, dummy_input, "madrigal.onnx")
```

## Hardware Requirements

| Stage | GPU Memory | Time (approx) |
|-------|------------|---------------|
| Stage 1 (each modality) | 8-16 GB | 2-24 hours |
| Stage 2 (pretraining) | 16-40 GB | 1-2 days |
| Stage 3 (fine-tuning) | 16 GB | 4-16 hours |
| Inference | 4-8 GB | seconds |

## Experiment Tracking

The codebase uses Weights & Biases (wandb) for tracking:
```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
python train_ddi_batch.py --from_yaml config.yaml
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use gradient accumulation
- Enable mixed precision: add `--fp16` if supported

### Slow Training
- Ensure GPU is being used: check `torch.cuda.is_available()`
- Use DataLoader with `num_workers > 0`
- Enable `pin_memory=True` for DataLoader

### Import Errors
Remember: Import `torch_geometric` BEFORE `torchdrug`:
```python
import torch_geometric  # First!
import torchdrug        # Second!
```