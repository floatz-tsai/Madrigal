"""
Data Retriever module for matching drugs to preprocessed Madrigal data.

This module helps match input drugs to existing data files for:
- Structure (molecular graphs)
- Knowledge Graph (PrimeKG embeddings)
- Cell Viability (PRISM data)
- Transcriptomics (CMap data)
"""

import os
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class DataRetriever:
    """
    Retrieve and match drug data from preprocessed Madrigal files.
    """

    def __init__(self, data_dir: str, base_dir: str = None):
        """
        Initialize DataRetriever.

        Args:
            data_dir: Path to processed_data directory (DATA_DIR)
            base_dir: Path to base Madrigal_Data directory (BASE_DIR)
        """
        self.data_dir = data_dir
        self.base_dir = base_dir or data_dir

        # Paths to data files
        self.metadata_path = os.path.join(data_dir, 'views_features_new/combined_metadata_ddi.pkl')
        self.str_path = os.path.join(data_dir, 'views_features_new/str/all_molecules_torchdrug.pt')
        self.kg_path = os.path.join(data_dir, 'views_features_new/kg/KG_data_hgt.pt')
        self.cv_path = os.path.join(data_dir, 'views_features_new/cv/cv_cp_data.csv')
        self.tx_path = os.path.join(data_dir, 'views_features_new/tx/tx_cp_data_averaged_intermediate.csv')

        # Load metadata
        self._metadata = None
        self._str_data = None
        self._kg_data = None

    @property
    def metadata(self) -> pd.DataFrame:
        """Lazy load metadata."""
        if self._metadata is None:
            if os.path.exists(self.metadata_path):
                self._metadata = pd.read_pickle(self.metadata_path)
                print(f"Loaded metadata: {self._metadata.shape[0]} drugs")
            else:
                raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
        return self._metadata

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Canonicalize SMILES using RDKit."""
        if not HAS_RDKIT:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            return None
        except:
            return None

    def find_drug_by_smiles(self, smiles: str) -> Optional[Dict]:
        """
        Find a drug in metadata by SMILES.

        Args:
            smiles: SMILES string (will be canonicalized)

        Returns:
            Dictionary with drug info and available modalities, or None
        """
        canonical_smiles = self.canonicalize_smiles(smiles)
        if canonical_smiles is None:
            print(f"Warning: Could not canonicalize SMILES: {smiles}")
            return None

        # Search in metadata
        matches = self.metadata[self.metadata['canonical_smiles'] == canonical_smiles]

        if len(matches) == 0:
            # Try original SMILES
            matches = self.metadata[self.metadata['canonical_smiles'] == smiles]

        if len(matches) == 0:
            return None

        idx = matches.index[0]
        row = matches.iloc[0]

        return {
            'index': idx,
            'canonical_smiles': row['canonical_smiles'],
            'has_structure': row.get('view_str', 1) == 1,
            'has_kg': row.get('view_kg', 0) == 1,
            'has_cv': row.get('view_cv', 0) == 1,
            'has_tx': any(row.get(f'view_tx_{cl}', 0) == 1 for cl in self.get_cell_lines()),
            'metadata_row': row.to_dict()
        }

    def find_drugs_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Find multiple drugs by SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with match results
        """
        results = []
        for smiles in smiles_list:
            result = self.find_drug_by_smiles(smiles)
            if result:
                results.append({
                    'input_smiles': smiles,
                    'found': True,
                    'index': result['index'],
                    'canonical_smiles': result['canonical_smiles'],
                    'has_structure': result['has_structure'],
                    'has_kg': result['has_kg'],
                    'has_cv': result['has_cv'],
                    'has_tx': result['has_tx']
                })
            else:
                results.append({
                    'input_smiles': smiles,
                    'found': False,
                    'index': None,
                    'canonical_smiles': None,
                    'has_structure': False,
                    'has_kg': False,
                    'has_cv': False,
                    'has_tx': False
                })

        return pd.DataFrame(results)

    def get_cell_lines(self) -> List[str]:
        """Get list of available cell lines for transcriptomics."""
        return [
            'A375', 'A549', 'HA1E', 'HCC515', 'HEPG2', 'HT29',
            'MCF7', 'PC3', 'VCAP', 'ASC', 'NPC', 'SKB',
            'FIBRNPC', 'SHSY5Y', 'NEU', 'NEU.KCL'
        ]

    def get_modality_availability(self, drug_indices: List[int]) -> pd.DataFrame:
        """
        Get modality availability matrix for given drug indices.

        Args:
            drug_indices: List of drug indices in metadata

        Returns:
            DataFrame with modality availability
        """
        meta = self.metadata.loc[drug_indices]

        availability = {
            'index': drug_indices,
            'view_str': [1] * len(drug_indices),  # Structure always available
            'view_kg': meta['view_kg'].values,
            'view_cv': meta['view_cv'].values,
        }

        # Add transcriptomics per cell line
        for cl in self.get_cell_lines():
            col = f'view_tx_{cl}'
            if col in meta.columns:
                availability[col] = meta[col].values
            else:
                availability[col] = [0] * len(drug_indices)

        return pd.DataFrame(availability)

    def prepare_drug_batch(self, drug_indices: List[int],
                           checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Prepare a batch of drugs for embedding generation.

        Args:
            drug_indices: List of drug indices
            checkpoint_path: Path to model checkpoint (for loading encoder configs)

        Returns:
            Dictionary with all required tensors for model input
        """
        from madrigal.data.data import get_all_drugs_data

        class Args:
            def __init__(self, data_dir, checkpoint, indices):
                self.path_base = data_dir
                self.checkpoint = checkpoint
                self.kg_encoder = 'hgt'
                self.split_method = 'split_by_pairs'
                self.batch_size = None
                self.data_source = 'DrugBank'
                self.repeat = None
                self.kg_sampling_num_neighbors = None
                self.kg_sampling_num_layers = None
                self.num_negative_samples_per_pair = None
                self.negative_sampling_probs_type = None
                self.num_workers = 2
                self.first_num_drugs = len(indices)

        args = Args(self.data_dir, checkpoint_path, drug_indices)

        # This loads the data in the format expected by the model
        _, test_loader, _, _, label_map = get_all_drugs_data(args)
        batch = next(iter(test_loader))

        return batch

    def get_available_drugs_summary(self) -> pd.DataFrame:
        """
        Get summary of all drugs and their modality availability.

        Returns:
            DataFrame with drug count per modality combination
        """
        meta = self.metadata.copy()
        meta['view_str'] = 1

        # Count drugs with each modality
        summary = {
            'Total drugs': len(meta),
            'With structure': (meta['view_str'] == 1).sum(),
            'With KG': (meta['view_kg'] == 1).sum(),
            'With CV': (meta['view_cv'] == 1).sum(),
        }

        # Check any TX
        tx_cols = [c for c in meta.columns if c.startswith('view_tx_')]
        if tx_cols:
            meta['has_any_tx'] = meta[tx_cols].max(axis=1)
            summary['With any TX'] = (meta['has_any_tx'] == 1).sum()

        # Full modality drugs
        meta['full_modality'] = (
            (meta['view_str'] == 1) &
            (meta['view_kg'] == 1) &
            (meta['view_cv'] == 1) &
            (meta.get('has_any_tx', 0) == 1)
        )
        summary['Full modality'] = meta['full_modality'].sum()

        return pd.DataFrame([summary]).T.rename(columns={0: 'Count'})

    def export_drug_list(self, output_path: str, include_modalities: bool = True):
        """
        Export list of all drugs with their SMILES and modality availability.

        Args:
            output_path: Path to save CSV
            include_modalities: Whether to include modality columns
        """
        meta = self.metadata.copy()

        if include_modalities:
            cols = ['canonical_smiles', 'view_kg', 'view_cv']
            cols += [c for c in meta.columns if c.startswith('view_tx_')]
        else:
            cols = ['canonical_smiles']

        meta[cols].to_csv(output_path)
        print(f"Exported {len(meta)} drugs to {output_path}")
