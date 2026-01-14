"""
Index module for drug data retrieval and embedding generation.

This module provides utilities to:
1. Look up drug information from external APIs (PubChem, DrugBank)
2. Match drugs to existing preprocessed data files
3. Prepare data for embedding generation
"""

from .drug_lookup import DrugLookup
from .data_retriever import DataRetriever

__all__ = ['DrugLookup', 'DataRetriever']
