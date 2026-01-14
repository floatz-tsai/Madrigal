# Contrastive Pretraining Process (pretrain.py)

This document details the contrastive pretraining stage of Madrigal, which aligns embeddings from different modalities (structure, knowledge graph, cell viability, transcriptomics) into a shared representation space using SimCLR.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Contrastive Pretraining Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Drug (same drug, two views)                                                 │
│         │                                                                     │
│         ▼                                                                     │
│   ┌─────────────┐     ┌─────────────┐                                        │
│   │   View 1    │     │   View 2    │     (Different modality subsets)       │
│   │  (e.g. STR) │     │(e.g. KG+CV) │                                        │
│   └──────┬──────┘     └──────┬──────┘                                        │
│          │                   │                                                │
│          ▼                   ▼                                                │
│   ┌─────────────────────────────────────┐                                    │
│   │         NovelDDIEncoder              │  (Shared encoder)                  │
│   │  ┌───────┬───────┬───────┬───────┐  │                                    │
│   │  │  GIN  │  HGT  │  MLP  │ChemCPA│  │  (Modality encoders)               │
│   │  │ (STR) │ (KG)  │ (CV)  │ (TX)  │  │                                    │
│   │  └───────┴───────┴───────┴───────┘  │                                    │
│   │              │                       │                                    │
│   │              ▼                       │                                    │
│   │      [Uni-modal Projector]           │  (If raw_encoder_output=True)      │
│   │              │                       │                                    │
│   └──────────────┼───────────────────────┘                                    │
│                  │                                                            │
│          ┌───────┴───────┐                                                    │
│          ▼               ▼                                                    │
│   ┌─────────────┐ ┌─────────────┐                                            │
│   │ Predictor 1 │ │ Predictor 2 │     (MLP projection heads)                  │
│   └──────┬──────┘ └──────┬──────┘                                            │
│          │               │                                                    │
│          ▼               ▼                                                    │
│       aug_1           aug_2            (Projected embeddings)                 │
│          │               │                                                    │
│          └───────┬───────┘                                                    │
│                  ▼                                                            │
│         ┌─────────────────┐                                                  │
│         │  InfoNCE Loss   │            (Contrastive loss)                    │
│         └─────────────────┘                                                  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Input Data Format

### 1.1 Data Loading (`get_pretrain_data`)

**Source**: `madrigal/data/data.py:275`

```python
train_drugs, val_drugs, pretrain_drugs, train_loader, val_loader, pretrain_loader, \
    collator, masks, all_train_subset_masks, hard_negative_mask, all_extra_molecules, \
    extra_mol_str_masks, kg_args = get_pretrain_data(args, hparams)
```

### 1.2 Drug Selection

Only drugs with **at least 2 modalities available** are used for pretraining:

```python
pretrain_drugs = mod_avail_df[mod_avail_df.sum(axis=1) >= 2].index.values
```

### 1.3 Batch Data Structure

Each batch from `train_loader` contains:

| Component | Shape | Description |
|-----------|-------|-------------|
| `batch_drug_indices` | `[batch_size]` | Drug indices in metadata |
| `batch_mols` | `PackedMolecule` | TorchDrug molecular graphs |
| `batch_kg` | `HeteroData` | PyG heterogeneous graph (shared) |
| `batch_cv` | `[batch_size, cv_dim]` | Cell viability features |
| `batch_tx_dict` | `{cell_line: [batch_size, tx_dim]}` | Transcriptomics per cell line |

### 1.4 Modality Masks

**Masks** indicate which modalities are available for each drug:
- `0` = modality **available** (not masked)
- `1` = modality **unavailable** (masked)

```python
# Example mask for a drug with STR, KG, CV available but TX unavailable
mask = [0, 0, 0, 1, 1, 1, ...]  # [str, kg, cv, tx_a375, tx_a549, ...]
       # ↑available    ↑masked
```

**Mask dimensions**: `[num_drugs, num_modalities]` where:
- Indices 0: Structure (STR)
- Index 1: Knowledge Graph (KG)
- Index 2: Cell Viability (CV)
- Indices 3+: Transcriptomics per cell line (16 cell lines)

## 2. Subset Mask Sampling

### 2.1 Pretraining Modes

**Source**: `madrigal/utils.py:360-390`

| Mode | View 1 (aug_1) | View 2 (aug_2) | Purpose |
|------|----------------|----------------|---------|
| `str_center` | STR only | Random subset (excl. STR) | Structure as anchor |
| `double_random` | Random subset | Random subset | Full random |
| `str_kg` | STR only | KG only | STR-KG alignment |

### 2.2 Mask Sampling Process

```python
# For str_center mode:
# View 1: Only structure (index 0)
aug1 = [0, 1, 1, 1, 1, ...]  # Only STR visible
       # ↑STR  ↑all others masked

# View 2: Random subset of available modalities (excluding STR)
aug2 = [1, 0, 0, 1, 1, ...]  # KG + CV visible (example)
       # ↑STR masked  ↑KG,CV visible
```

### 2.3 Balanced Sampling

With `pretrain_unbalanced=False`, modality sampling is weighted by inverse frequency:
```python
mod_probs = (1 / (1 - masks).sum(axis=0))  # Rarer modalities sampled more often
```

## 3. Model Architecture

### 3.1 SimCLR_NovelDDI

**Source**: `madrigal/models/simclr.py:11-141`

```python
class SimCLR_NovelDDI(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=1024, T=1.0,
                 raw_encoder_output=False, shared_predictor=False):
        self.base_encoder = base_encoder      # NovelDDIEncoder
        self.T = T                             # Temperature for contrastive loss
        self.predictor_1 = MLP(dim, mlp_dim, dim)  # 2-layer MLP
        self.predictor_2 = MLP(dim, mlp_dim, dim)  # 2-layer MLP
```

### 3.2 Predictor MLP Structure

```python
# 2-layer MLP with BatchNorm
predictor = nn.Sequential(
    nn.Linear(256, 1024, bias=False),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 256, bias=False),
    nn.BatchNorm1d(256, affine=False),  # No learnable params (SimCLR design)
)
```

### 3.3 Base Encoder (NovelDDIEncoder)

**Source**: `madrigal/models/models.py`

```
NovelDDIEncoder
├── str_encoder (GIN)           # Graph Isomorphism Network for molecules
├── kg_encoder (HGT)            # Heterogeneous Graph Transformer for KG
├── cv_encoder (MLP)            # MLP for cell viability
├── tx_encoder (ChemCPA)        # Transcriptomics encoder
├── uni_projector               # Projects unimodal embeddings to shared space
└── [Optional: Transformer fusion - NOT used in pretraining]
```

## 4. Forward Pass / Tensor Flow

### 4.1 Training Step

**Source**: `pretrain.py:59-106`

```python
for i, batch_data in enumerate(train_loader):
    # 1. Move data to device
    batch_drug_indices, batch_data = to_device(batch_data, device)

    # 2. Sample two different modality subsets for each drug
    batch_mask1, batch_mask2 = pretrain_modality_subset_sampler(...)

    # 3. Forward pass with contrastive loss
    _, _, (logits, labels, loss) = model(
        batch_drug_indices,    # [batch_size]
        batch_mask1,           # [batch_size, num_modalities] - View 1 mask
        batch_mask2,           # [batch_size, num_modalities] - View 2 mask
        batch_hard_neg_mask,   # Optional: mask for hard negatives
        batch_data,            # (mols, kg, cv, tx_dict)
        batch_extra_mols,      # Optional: extra negative molecules
        extra_mol_str_masks    # Optional: masks for extra molecules
    )

    # 4. Backward pass
    loss.backward()
    optimizer.step()
```

### 4.2 SimCLR Forward Pass

**Source**: `madrigal/models/simclr.py:110-140`

```python
def forward(self, drug_indices, batch_mask_1, batch_mask_2, ...):
    # Unpack batch data
    batch_mols, batch_kg, batch_cv, batch_tx_dict = batch_data

    # Encode View 1 (e.g., structure only)
    z1 = self.base_encoder(
        drug_indices,          # [batch_size]
        batch_mask_1,          # [batch_size, num_mods] - e.g., [0,1,1,1,...]
        batch_mols,            # PackedMolecule
        batch_kg,              # HeteroData
        batch_cv,              # [batch_size, cv_dim]
        batch_tx_dict,         # {cell_line: [batch_size, tx_dim]}
        raw_encoder_output=True
    )
    # Shape: [batch_size, embed_dim]

    # Project through predictor 1
    aug_1 = self.predictor_1(z1)  # [batch_size, 256]

    # Encode View 2 (e.g., KG + CV)
    z2 = self.base_encoder(drug_indices, batch_mask_2, ...)
    aug_2 = self.predictor_2(z2)  # [batch_size, 256]

    # Compute contrastive loss
    logits, labels, loss = self.contrastive_loss(aug_1, aug_2, ...)

    return aug_1, aug_2, (logits, labels, loss)
```

### 4.3 Encoder Forward Pass (Unimodal)

When `raw_encoder_output=True`, the encoder returns unimodal embeddings:

```python
# In NovelDDIEncoder.encode() with raw_encoder_output=True:

# 1. Encode each modality
str_out = self.str_encoder(batch_mols)          # [batch, str_dim]
kg_out = self.kg_encoder(batch_kg, drug_ids)    # [batch, kg_dim]
cv_out = self.cv_encoder(batch_cv)              # [batch, cv_dim]
tx_out = self.tx_encoder(batch_tx)              # [batch, tx_dim]

# 2. Stack all embeddings
all_embeds = torch.stack([str_out, kg_out, cv_out, tx_out], dim=1)
# Shape: [batch_size, num_modalities, embed_dim]

# 3. Select only visible modalities based on mask
# mask: [batch_size, num_modalities], 0=visible, 1=masked
visible_embeds = all_embeds[~batch_masks, :]  # Select unmasked

# 4. Project through uni_projector (if single modality visible)
z = self.uni_projector(visible_embeds)  # [batch_size, 256]

return z
```

## 5. Contrastive Loss (InfoNCE)

### 5.1 Loss Computation

**Source**: `madrigal/models/simclr.py:74-108`

```python
def contrastive_loss(self, aug1, aug2, batch_too_hard_neg_mask):
    # 1. Concatenate both views
    features = torch.cat([aug1, aug2], dim=0)  # [2*batch, 256]

    # 2. Create positive pair labels
    # For batch_size=4: labels identify which samples are from same drug
    labels = torch.cat([torch.arange(batch_size)] * 2, dim=0)
    # labels = [0,1,2,3, 0,1,2,3]

    # 3. Create label matrix (positive pairs = 1)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Shape: [2*batch, 2*batch], 1 where i,j are same drug

    # 4. Normalize features
    features = F.normalize(features, dim=1)

    # 5. Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    # Shape: [2*batch, 2*batch]

    # 6. Remove diagonal (self-similarity)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(...)

    # 7. Apply temperature scaling
    logits = similarity_matrix / self.T  # T typically 0.1

    # 8. Cross-entropy loss
    loss = CrossEntropyLoss()(logits, labels)

    return logits, labels, loss
```

### 5.2 Loss Intuition

```
Similarity Matrix (before diagonal removal):
         aug1_0  aug1_1  aug1_2  aug2_0  aug2_1  aug2_2
aug1_0   [1.0    0.2     0.3     0.9*    0.1     0.2  ]
aug1_1   [0.2    1.0     0.4     0.1     0.85*   0.3  ]
aug1_2   [0.3    0.4     1.0     0.2     0.3     0.88*]
aug2_0   [0.9*   0.1     0.2     1.0     0.2     0.3  ]
aug2_1   [0.1    0.85*   0.3     0.2     1.0     0.4  ]
aug2_2   [0.2    0.3     0.88*   0.3     0.4     1.0  ]

* = positive pairs (same drug, different views)
    These should have HIGH similarity

All other pairs = negatives (different drugs)
    These should have LOW similarity
```

## 6. Training Configuration

### 6.1 Key Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `pretrain_batch_size` | 256 | Batch size |
| `pretrain_lr` | 1e-4 | Learning rate (scaled by batch/512) |
| `pretrain_num_epochs` | 1000 | Total epochs |
| `warmup_epochs` | 10 | LR warmup epochs |
| `moco_t` (Temperature) | 0.1 | InfoNCE temperature |
| `feature_dim` | 256 | Embedding dimension |
| `moco_mlp_dim` | 1024 | Predictor hidden dimension |

### 6.2 Optimizer

```python
# LARS optimizer (common for contrastive learning)
optimizer = LARS(
    model.parameters(),
    lr=pretrain_lr * batch_size / 512,  # Linear scaling
    weight_decay=1e-6,
    momentum=0.9
)
```

### 6.3 Learning Rate Schedule

Cosine annealing with warmup:

```python
def adjust_learning_rate(optimizer, epoch, base_lr, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1 + cos(pi * (epoch - warmup) / (total - warmup)))
    return lr
```

## 7. Checkpoint Saving

### 7.1 Saved Components

```python
save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),       # Full SimCLR model weights
    'optimizer': optimizer.state_dict(),
    'encoder_configs': encoder_configs,      # Encoder architecture config
    'kg_args': kg_args,                      # KG encoder arguments
})
```

### 7.2 State Dict Structure

```
checkpoint['state_dict']:
├── base_encoder.str_encoder.*        # GIN weights (transferred to finetune)
├── base_encoder.kg_encoder.*         # HGT weights (transferred to finetune)
├── base_encoder.cv_encoder.*         # CV MLP weights (transferred)
├── base_encoder.tx_encoder.*         # ChemCPA weights (transferred)
├── base_encoder.uni_projector.*      # Unimodal projector (optionally transferred)
├── base_encoder.pos_encoder.*        # NOT transferred (re-initialized)
├── base_encoder.transformer.*        # NOT transferred (re-initialized)
├── predictor_1.*                     # NOT transferred (discarded)
└── predictor_2.*                     # NOT transferred (discarded)
```

## 8. Summary Diagram

```
┌────────────────────── TRAINING LOOP ──────────────────────┐
│                                                            │
│  for epoch in range(1000):                                 │
│      for batch in train_loader:                            │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 1. Sample two modality subsets (masks)              │  │
│  │    mask1 = [0,1,1,1,...] (STR only)                 │  │
│  │    mask2 = [1,0,0,1,...] (KG+CV)                    │  │
│  └─────────────────────────────────────────────────────┘  │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 2. Encode with NovelDDIEncoder                      │  │
│  │    z1 = encoder(drugs, mask1, data)  # [B, 256]     │  │
│  │    z2 = encoder(drugs, mask2, data)  # [B, 256]     │  │
│  └─────────────────────────────────────────────────────┘  │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 3. Project through predictor MLPs                   │  │
│  │    aug1 = predictor_1(z1)  # [B, 256]               │  │
│  │    aug2 = predictor_2(z2)  # [B, 256]               │  │
│  └─────────────────────────────────────────────────────┘  │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 4. Compute InfoNCE loss                             │  │
│  │    - Positive pairs: (aug1[i], aug2[i])             │  │
│  │    - Negative pairs: all other combinations         │  │
│  │    - Loss = CrossEntropy(similarity/T, labels)      │  │
│  └─────────────────────────────────────────────────────┘  │
│          │                                                 │
│          ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 5. Backward + optimizer step                        │  │
│  │    loss.backward()                                  │  │
│  │    optimizer.step()                                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Save checkpoint_1000.pt │
              │ (encoder weights for    │
              │  DDI finetuning)        │
              └─────────────────────────┘
```
