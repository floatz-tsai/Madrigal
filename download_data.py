#!/usr/bin/env python3
"""
Download Madrigal dataset from Harvard Dataverse.
Run: python download_data.py
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# Base directory for data
BASE_DIR = Path("/Users/ping-hsuntsai/Desktop/git/Madrigal_Data/processed_data")

# Harvard Dataverse API
DATAVERSE_API = "https://dataverse.harvard.edu/api/access/datafile"
DOI = "doi:10.7910/DVN/ZFTW3J"

# Essential files to download (file_id: local_path)
# File IDs can be found by inspecting the Dataverse page
ESSENTIAL_FILES = {
    # views_features_new - core metadata and features
    "combined_metadata_ddi.pkl": "views_features_new/combined_metadata_ddi.pkl",

    # Structure modality
    "all_molecules_torchdrug.pt": "views_features_new/str/all_molecules_torchdrug.pt",

    # Knowledge Graph modality
    "KG_data_hgt.pt": "views_features_new/kg/KG_data_hgt.pt",

    # Cell Viability modality
    "cv_cp_data.csv": "views_features_new/cv/cv_cp_data.csv",

    # Transcriptomics modality
    "tx_cp_data_averaged_intermediate.csv": "views_features_new/tx/tx_cp_data_averaged_intermediate.csv",
    "tx_cp_metadata_for_adapting.pkl": "views_features_new/tx/tx_cp_metadata_for_adapting.pkl",
    "tx_cp_data_averaged_for_adapting.csv": "views_features_new/tx/tx_cp_data_averaged_for_adapting.csv",

    # RDKit embeddings
    "rdkit2D_embeddings_combined_all_normalized.parquet": "views_features_new/tx/embeddings/rdkit2D_embeddings_combined_all_normalized.parquet",
    "rdkit2D_embeddings_all_all_normalized.parquet": "views_features_new/tx/embeddings/rdkit2D_embeddings_all_all_normalized.parquet",

    # DrugBank DDI data - split_by_pairs
    "drugbank_train_df_pairs.csv": "polypharmacy_new/DrugBank/split_by_pairs/train_df.csv",
    "drugbank_val_df_pairs.csv": "polypharmacy_new/DrugBank/split_by_pairs/val_df.csv",
    "drugbank_test_df_pairs.csv": "polypharmacy_new/DrugBank/split_by_pairs/test_df.csv",
    "drugbank_ddi_directed_final_label_map.pkl": "polypharmacy_new/DrugBank/drugbank_ddi_directed_final_label_map.pkl",
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def get_file_list():
    """Get list of files from Dataverse API."""
    url = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={DOI}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        files = data.get('data', {}).get('latestVersion', {}).get('files', [])
        return {f['dataFile']['filename']: f['dataFile']['id'] for f in files}
    return {}


def main():
    print("=" * 60)
    print("Madrigal Data Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {BASE_DIR}")
    print("\nFetching file list from Harvard Dataverse...")

    file_map = get_file_list()

    if not file_map:
        print("\nCould not fetch file list automatically.")
        print("\nPlease download manually from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZFTW3J")
        print("\nRequired files:")
        for filename, local_path in ESSENTIAL_FILES.items():
            print(f"  - {filename} -> {local_path}")
        return

    print(f"\nFound {len(file_map)} files in dataset.")
    print("\nDownloading essential files...")

    downloaded = 0
    skipped = 0
    failed = []

    for filename, local_path in ESSENTIAL_FILES.items():
        dest = BASE_DIR / local_path

        if dest.exists():
            print(f"[SKIP] {filename} (already exists)")
            skipped += 1
            continue

        # Find matching file in dataverse
        file_id = None
        for dv_filename, fid in file_map.items():
            if filename in dv_filename or dv_filename.endswith(filename):
                file_id = fid
                break

        if file_id:
            url = f"{DATAVERSE_API}/{file_id}"
            try:
                print(f"\n[DOWNLOAD] {filename}")
                download_file(url, dest)
                downloaded += 1
            except Exception as e:
                print(f"[ERROR] Failed to download {filename}: {e}")
                failed.append(filename)
        else:
            print(f"[NOT FOUND] {filename} - download manually")
            failed.append(filename)

    print("\n" + "=" * 60)
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed/Not found: {len(failed)}")

    if failed:
        print("\nFiles to download manually:")
        for f in failed:
            print(f"  - {f}")
        print("\nDownload from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZFTW3J")


if __name__ == "__main__":
    main()
