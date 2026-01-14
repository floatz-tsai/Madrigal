# Madrigal Data Collection Guide

This document describes how the data for Madrigal was collected and organized, based on the paper [arXiv:2503.02781](https://arxiv.org/abs/2503.02781).

## Overview

Madrigal integrates four modalities of preclinical drug data to predict drug-drug interaction (DDI) outcomes:
1. **Molecular Structure** - Chemical structure representations
2. **Knowledge Graph** - Biological pathways and targets from PrimeKG
3. **Cell Viability** - PRISM drug screening data
4. **Transcriptomics** - Gene expression changes from Extended CMap

---

## How the Original Paper Collected Data

### Step 1: DDI Outcome Data Collection

#### DrugBank (Expert-Curated)
- **Source**: [DrugBank XML dumps](https://go.drugbank.com/releases) (version 2023-01-04)
- **Collection Method**:
  1. Downloaded full DrugBank XML database dump
  2. Extracted drug interaction statements using regex patterns
  3. Manually grouped semantically similar interaction outcomes into 158 categories
  4. Mapped all drugs to DrugBank IDs
- **Result**: 1,188,371 drug combinations across 3,632 drugs

#### TWOSIDES (Real-World Evidence)
- **Source**: [FDA Adverse Event Reporting System (FAERS)](https://www.fda.gov/drugs/drug-approvals-and-databases/fda-adverse-event-reporting-system-faers)
- **Collection Method**:
  1. Extracted drug pair co-occurrences from FAERS reports
  2. Applied statistical filters:
     - Minimum report count thresholds
     - Proportional reporting ratio (PRR) calculations
     - Chi-square statistical significance tests
  3. Filtered to retain only statistically significant DDI signals
- **Result**: 4,656,138 drug combinations across 1,457 drugs and 795 outcomes

### Step 2: Drug Identity Matching

The paper used a two-tier matching strategy to link drugs across all data sources:

1. **Primary Key**: DrugBank ID
   - All drug interaction data mapped to compounds via DrugBank ID

2. **Canonical SMILES**: RDKit-transformed canonical SMILES
   - Used to match drugs across modality sources (PRISM, CMap, PrimeKG)
   - Ensures chemical identity consistency

```
DrugBank ID → Canonical SMILES → Match to all modalities
```

### Step 3: Modality Data Collection

#### Structure Modality
- **Source**: Drug SMILES strings from DrugBank
- **Processing**:
  1. Extract canonical SMILES from DrugBank
  2. Convert to molecular graphs using TorchDrug
  3. Store as PyTorch geometric objects

#### Knowledge Graph Modality
- **Source**: [PrimeKG](https://github.com/mims-harvard/PrimeKG) (Harvard biomedical knowledge graph)
- **Processing**:
  1. Downloaded PrimeKG (contains 10+ biomedical entity types)
  2. **Critical**: Removed all drug-drug interaction edges to prevent data leakage
  3. **Critical**: Removed all drug-phenotype interaction edges to prevent data leakage
  4. Matched drugs to PrimeKG via DrugBank ID
  5. Converted to PyG HeteroData format for HGT encoder

#### Cell Viability Modality
- **Source**: [PRISM Repurposing Dataset](https://depmap.org/repurposing/) (Broad Institute)
- **Processing**:
  1. Downloaded PRISM drug sensitivity screening data
  2. Matched drugs via canonical SMILES
  3. Extracted cell viability signatures across cancer cell lines

#### Transcriptomics Modality
- **Source**: [Extended Connectivity Map (CMap) 2020](https://clue.io/) (Broad Institute)
- **Processing**:
  1. Downloaded L1000 gene expression data
  2. Selected 16 cell lines with sufficient drug coverage:
     - A375, A549, HA1E, HCC515, HEPG2, HT29
     - MCF7, PC3, VCAP, ASC, NPC, SKB
     - FIBRNPC, SHSY5Y, NEU, NEU.KCL
  3. Averaged signatures across doses per cell line
  4. Matched drugs via canonical SMILES

### Step 4: Creating the Unified Metadata

The `combined_metadata_ddi.pkl` file was created by:

1. Starting with all unique drugs from DrugBank and TWOSIDES
2. For each drug, checking availability in each modality source
3. Recording:
   - Canonical SMILES (from RDKit)
   - Modality availability flags (view_str, view_kg, view_cv, view_tx_*)
   - Signature IDs linking to external data files
   - Dosage information for transcriptomics

---

## Primary Data Sources Summary

| Dataset | Description | Drugs | Outcomes | Drug Combinations |
|---------|-------------|-------|----------|-------------------|
| **DrugBank** (2023-01-04) | Expert-curated DDI database | 3,632 | 158 | 1,188,371 |
| **TWOSIDES** (2019-11-15) | Self-reported DDIs from FDA FAERS | 1,457 | 795 | 4,656,138 |

### Modality Data Sources

| Modality | Source | Description |
|----------|--------|-------------|
| Structure | RDKit + TorchDrug | SMILES converted to molecular graphs |
| Knowledge Graph | [PrimeKG](https://github.com/mims-harvard/PrimeKG) | Biomedical knowledge graph |
| Cell Viability | [PRISM Repurposing](https://depmap.org/repurposing/) | Drug sensitivity screening |
| Transcriptomics | [Extended CMap 2020](https://clue.io/) | Gene expression signatures |

## Drug Matching Strategy

All drugs are matched across databases using a unified identifier system:

1. **Primary identifier**: DrugBank ID
2. **Canonical linking**: RDKit-transformed canonical SMILES

This ensures the same compound across different databases is properly linked to all its available modalities.

## Metadata Structure (`combined_metadata_ddi.pkl`)

The central metadata file links each drug to its available modalities.

### Columns

| Category | Fields | Description |
|----------|--------|-------------|
| Molecular | `canonical_smiles` | RDKit canonical SMILES string |
| Modality Flags | `view_str`, `view_kg`, `view_cv`, `view_tx_{cell_line}` | 1 if modality available, 0 if masked |
| Signature IDs | `kg_sig_id`, `cv_sig_id`, `{cell_line}_max_dose_averaged_sig_id` | Links to external data files |
| Dosage | `{cell_line}_pert_dose` | Perturbation dosage for transcriptomics |

### Transcriptomics Cell Lines (16 total)

The 16 cell lines used for transcriptomics modality:
- A375, A549, HA1E, HCC515, HEPG2, HT29
- MCF7, PC3, VCAP, ASC, NPC, SKB
- FIBRNPC, SHSY5Y, NEU, NEU.KCL

## Modality-Specific Data Files

### Structure (`views_features_new/str/`)
- `all_molecules_torchdrug.pt` - TorchDrug molecular graph objects

### Knowledge Graph (`views_features_new/kg/`)
- `KG_data_hgt.pt` - PyG HeteroData object for HGT encoder
- Source: PrimeKG with **drug-drug and drug-phenotype edges excluded** to prevent information leakage

### Cell Viability (`views_features_new/cv/`)
- `cv_cp_data.csv` - PRISM cell viability signatures

### Transcriptomics (`views_features_new/tx/`)
- `tx_cp_data_averaged_intermediate.csv` - Averaged gene expression signatures

## DDI Outcome Data

Located in `polypharmacy_new/{dataset}/split_by_{method}/`:

### Datasets
- `DrugBank/` - Expert-curated outcomes
- `TWOSIDES/` - FAERS-derived outcomes
- `ONSIDES/` - Clinical outcome data

### Split Methods
| Split | Description |
|-------|-------------|
| `split_by_pairs` | Random split by drug pairs |
| `split_by_triplets` | Random split by (drug1, drug2, outcome) |
| `split_by_drugs_random` | Random split by individual drugs |
| `split_by_drugs_atc` | Split by ATC classification (tests generalization) |
| `split_by_drugs_targets` | Split by therapeutic targets (tests generalization) |
| `split_by_drugs_taxonomy` | Split by drug taxonomy |

## Data Preprocessing Notes

1. **PrimeKG filtering**: All drug-drug interactions and drug-phenotype interactions removed to avoid information leakage during training

2. **Transcriptomics selection**: 16 cell lines selected based on data availability and diversity

3. **Missing modality handling**: Madrigal uses an attention bottleneck module to handle missing modalities during both training and inference

4. **SMILES canonicalization**: All SMILES strings processed through RDKit for consistent representation

## Data Access

### Download
```bash
# Using Python script
python download_data.py

# Using shell script
bash download_data.sh
```

### External Links
- **Datasets**: [Harvard Dataverse](https://doi.org/10.7910/DVN/ZFTW3J)
- **Model Checkpoints**: [HuggingFace](https://huggingface.co/mims-harvard/Madrigal)
- **Paper**: [arXiv:2503.02781](https://arxiv.org/abs/2503.02781)

## References

1. Chandak P, et al. "Building a knowledge graph to enable precision medicine." Scientific Data (2023). [PrimeKG]
2. Corsello SM, et al. "Discovering the anticancer potential of non-oncology drugs." Nature Cancer (2020). [PRISM]
3. Subramanian A, et al. "A next generation connectivity map." Cell (2017). [CMap]
4. Wishart DS, et al. "DrugBank 5.0." Nucleic Acids Research (2018). [DrugBank]
5. Tatonetti NP, et al. "Data-driven prediction of drug effects and interactions." Science Translational Medicine (2012). [TWOSIDES]
