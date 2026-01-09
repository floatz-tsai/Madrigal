# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Madrigal is a PyTorch-based multimodal AI model for predicting drug combination outcomes from preclinical data. It fuses four modalities: molecular structure, knowledge graphs, cell viability, and transcriptomics to predict drug-drug interactions (DDIs).

**Paper**: arXiv:2503.02781 - "Multimodal AI predicts clinical outcomes of drug combinations from preclinical data"

## Build & Development Commands

### Environment Setup
```bash
# Create conda environment
mamba env create -f env_new.yaml

# Activate environment
mamba activate madrigal_env

# Install package in editable mode
python -m pip install -e .

# Verify installation
python -c "import madrigal; print('Imported')"
```

### Training Commands
```bash
# Second-stage modality alignment (pretraining)
python pretrain.py --from_yaml configs/cl_pretrain/pretrain_drugbank_shared_basal.yaml

# DDI finetuning
python train_ddi_batch.py --from_yaml configs/ddi_finetune/DrugBank/sweep_config_*.yaml

# Finetuning with pretrained checkpoint
python train_ddi_batch.py --checkpoint=checkpoint_1000.pt --finetune_mode=str_str+random_sample --split_method=split_by_pairs --from_yaml=configs/ddi_finetune/DrugBank/sweep_config_*.yaml
```

### Configuration
Requires a `.env` file in project root with:
```
PROJECT_DIR=/path/to/Madrigal/
BASE_DIR=/path/to/Madrigal_Data/
DATA_DIR=/path/to/Madrigal_Data/processed_data/
ENCODER_CKPT_DIR=/path/to/Madrigal/modality_pretraining/
CL_CKPT_DIR=/path/to/Madrigal_Data/model_output/pretrain/
```

## Architecture

### Two-Stage Training Pipeline
1. **Stage 1 (Optional)**: Modality-specific pretraining in `modality_pretraining/`
   - `str/`: GIN for molecular structure
   - `kg/`: HGT for knowledge graphs
   - `cv/`: MLP for cell viability
   - `tx/`: ChemCPA for transcriptomics

2. **Stage 2**: Contrastive learning alignment (`pretrain.py`)
   - SimCLR-based multimodal alignment using InfoNCE loss

3. **Stage 3**: DDI finetuning (`train_ddi_batch.py`)
   - Transformer-based multimodal fusion with bilinear decoder

### Core Package Structure (`madrigal/`)
- `models/models.py` - `NovelDDIEncoder` and modality-specific encoders (GIN, HGT, MLPEncoder)
- `models/simclr.py` - `SimCLR_NovelDDI` for contrastive pretraining
- `data/data.py` - `LongDDIDataset` and data loaders
- `evaluate/evaluate.py` - `evaluate_ft()`, `evaluate_pt()` training evaluation
- `evaluate/predict.py` - Inference and embedding generation
- `parse_args.py` - CLI argument parsing and YAML config loading
- `utils.py` - Constants, utilities, LARS optimizer, LR schedulers
- `chemcpa/` - ChemCPA integration for transcriptomics

### Key Constants (`madrigal/utils.py`)
- `MOL_DIM = 67` - TorchDrug molecular feature dimension
- `CELL_LINES` - 16 cell lines for transcriptomics
- `NON_TX_MODALITIES` - Configurable via `.env` (default: ["str", "kg", "cv"])

### Supported Data Sources
- TWOSIDES, ONSIDES, DrugBank

### Split Methods
- `split_by_triplets`, `split_by_pairs`, `split_by_drugs_random`, `split_by_drugs_atc`, `split_by_drugs_targets`, `split_by_drugs_taxonomy`

### Ablation/Finetune Modes
- `str_str`, `str_full`, `full_full`, `ablation_str_str`, etc.

## Notebooks
- `notebooks/generate_embeddings.ipynb` - Generate embeddings and normalized scores
- `notebooks/quick_predictions.ipynb` - Quick inference for specific drug pairs
- `notebooks/fig*/` - Paper figure reproduction

## Known Issues
1. TorchDrug must be imported AFTER torch_geometric
2. Requires `torchdrug>=0.2.0.post1` (earlier versions have LR scheduler issues)
3. Requires CUDA < 12.0 with PyTorch 1.13.1; for CUDA > 12.0, use PyTorch 2.1.0 with pytorch-geometric < 2.4.0

## External Data
- Datasets: [Harvard Dataverse](https://doi.org/10.7910/DVN/ZFTW3J)
- Checkpoints: [HuggingFace](https://huggingface.co/mims-harvard/Madrigal/tree/main)
