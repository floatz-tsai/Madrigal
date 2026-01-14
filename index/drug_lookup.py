"""
Drug Lookup module for fetching drug information from external APIs.

Supports:
- PubChem API: Get SMILES, molecular properties
- DrugBank ID matching
- RDKit canonicalization
"""

import requests
import time
from typing import Optional, Dict, List, Any, Union
import pandas as pd

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not installed. SMILES canonicalization will be disabled.")


class DrugLookup:
    """
    Lookup drug information from external APIs.

    Supported identifiers:
    - DrugBank ID (e.g., "DB00001")
    - PubChem CID (e.g., 2244)
    - Drug name (e.g., "Aspirin")
    - SMILES string
    """

    PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, cache_results: bool = True):
        """
        Initialize DrugLookup.

        Args:
            cache_results: Whether to cache API results to avoid repeated calls
        """
        self.cache_results = cache_results
        self._cache: Dict[str, Any] = {}

    def get_smiles_from_name(self, drug_name: str) -> Optional[str]:
        """
        Get canonical SMILES from drug name via PubChem API.

        Args:
            drug_name: Drug name (e.g., "Aspirin")

        Returns:
            Canonical SMILES string or None if not found
        """
        cache_key = f"name_{drug_name}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Use IsomericSMILES which is more reliable
            url = f"{self.PUBCHEM_BASE_URL}/compound/name/{drug_name}/property/IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                props = data['PropertyTable']['Properties'][0]
                # Try different SMILES property names (PubChem returns different keys)
                smiles = (props.get('IsomericSMILES') or props.get('CanonicalSMILES') or
                          props.get('ConnectivitySMILES') or props.get('SMILES'))
                if smiles and self.cache_results:
                    self._cache[cache_key] = smiles
                return smiles
            else:
                print(f"Warning: Could not find drug '{drug_name}' in PubChem")
                return None

        except Exception as e:
            print(f"Error fetching SMILES for '{drug_name}': {e}")
            return None

    def get_smiles_from_cid(self, cid: int) -> Optional[str]:
        """
        Get canonical SMILES from PubChem CID.

        Args:
            cid: PubChem Compound ID

        Returns:
            Canonical SMILES string or None if not found
        """
        cache_key = f"cid_{cid}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            url = f"{self.PUBCHEM_BASE_URL}/compound/cid/{cid}/property/IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                props = data['PropertyTable']['Properties'][0]
                smiles = (props.get('IsomericSMILES') or props.get('CanonicalSMILES') or
                          props.get('ConnectivitySMILES') or props.get('SMILES'))
                if smiles and self.cache_results:
                    self._cache[cache_key] = smiles
                return smiles
            else:
                print(f"Warning: Could not find CID {cid} in PubChem")
                return None

        except Exception as e:
            print(f"Error fetching SMILES for CID {cid}: {e}")
            return None

    def get_drugbank_info_from_name(self, drug_name: str) -> Optional[Dict]:
        """
        Search for DrugBank information via PubChem cross-references.

        Args:
            drug_name: Drug name

        Returns:
            Dictionary with DrugBank ID and other info, or None
        """
        try:
            # First get CID from name
            url = f"{self.PUBCHEM_BASE_URL}/compound/name/{drug_name}/cids/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            cid = response.json()['IdentifierList']['CID'][0]

            # Then get synonyms which may contain DrugBank ID
            url = f"{self.PUBCHEM_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                synonyms = response.json()['InformationList']['Information'][0]['Synonym']

                # Look for DrugBank ID pattern
                drugbank_id = None
                for syn in synonyms:
                    if syn.startswith('DB') and len(syn) == 7 and syn[2:].isdigit():
                        drugbank_id = syn
                        break

                return {
                    'cid': cid,
                    'drugbank_id': drugbank_id,
                    'synonyms': synonyms[:10]  # First 10 synonyms
                }
            return None

        except Exception as e:
            print(f"Error fetching DrugBank info for '{drug_name}': {e}")
            return None

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Canonicalize SMILES string using RDKit.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES or None if invalid
        """
        if not HAS_RDKIT:
            print("Warning: RDKit not available, returning original SMILES")
            return smiles

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                print(f"Warning: Invalid SMILES: {smiles}")
                return None
        except Exception as e:
            print(f"Error canonicalizing SMILES: {e}")
            return None

    def lookup_drugs_batch(self, identifiers: List[Union[str, int]],
                           id_type: str = 'name',
                           delay: float = 0.2) -> pd.DataFrame:
        """
        Lookup multiple drugs and return as DataFrame.

        Args:
            identifiers: List of drug identifiers
            id_type: Type of identifier ('name', 'cid', 'smiles')
            delay: Delay between API calls to avoid rate limiting

        Returns:
            DataFrame with columns: [identifier, canonical_smiles, drugbank_id, status]
        """
        results = []

        for i, identifier in enumerate(identifiers):
            print(f"Processing {i+1}/{len(identifiers)}: {identifier}")

            result = {
                'identifier': identifier,
                'canonical_smiles': None,
                'drugbank_id': None,
                'status': 'not_found'
            }

            if id_type == 'name':
                smiles = self.get_smiles_from_name(str(identifier))
                if smiles:
                    result['canonical_smiles'] = self.canonicalize_smiles(smiles)
                    info = self.get_drugbank_info_from_name(str(identifier))
                    if info:
                        result['drugbank_id'] = info.get('drugbank_id')
                    result['status'] = 'found'

            elif id_type == 'cid':
                smiles = self.get_smiles_from_cid(int(identifier))
                if smiles:
                    result['canonical_smiles'] = self.canonicalize_smiles(smiles)
                    result['status'] = 'found'

            elif id_type == 'smiles':
                result['canonical_smiles'] = self.canonicalize_smiles(str(identifier))
                if result['canonical_smiles']:
                    result['status'] = 'found'

            results.append(result)

            # Rate limiting
            if i < len(identifiers) - 1:
                time.sleep(delay)

        return pd.DataFrame(results)

    def get_compound_properties(self, smiles: str) -> Optional[Dict]:
        """
        Get molecular properties from PubChem for a given SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular properties
        """
        try:
            url = f"{self.PUBCHEM_BASE_URL}/compound/smiles/{smiles}/property/MolecularWeight,MolecularFormula,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()['PropertyTable']['Properties'][0]
            return None

        except Exception as e:
            print(f"Error fetching properties: {e}")
            return None
