#!/bin/bash
# Download essential Madrigal data from Harvard Dataverse
# Usage: bash download_data.sh

BASE_DIR="/Users/ping-hsuntsai/Desktop/git/Madrigal_Data/processed_data"
API_URL="https://dataverse.harvard.edu/api/access/datafile"

echo "========================================"
echo "Madrigal Data Downloader"
echo "========================================"
echo "Target: $BASE_DIR"
echo ""

# Create directories
mkdir -p "$BASE_DIR/views_features_new/str"
mkdir -p "$BASE_DIR/views_features_new/kg"
mkdir -p "$BASE_DIR/views_features_new/cv"
mkdir -p "$BASE_DIR/views_features_new/tx/embeddings"
mkdir -p "$BASE_DIR/polypharmacy_new/DrugBank/split_by_pairs"

download_file() {
    local file_id=$1
    local dest_path=$2
    local desc=$3

    if [ -f "$dest_path" ]; then
        echo "[SKIP] $desc (exists)"
        return
    fi

    echo "[DOWNLOADING] $desc..."
    curl -L -o "$dest_path" "$API_URL/$file_id"
    echo "[DONE] $desc"
}

echo "Downloading core metadata..."
download_file "10638803" "$BASE_DIR/views_features_new/combined_metadata_ddi.pkl" "combined_metadata_ddi.pkl (22 MB)"

echo ""
echo "Downloading structure modality..."
download_file "10638805" "$BASE_DIR/views_features_new/str/all_molecules_torchdrug.pt" "all_molecules_torchdrug.pt (628 MB)"

echo ""
echo "Downloading knowledge graph modality..."
download_file "10638826" "$BASE_DIR/views_features_new/kg/KG_data_hgt.pt" "KG_data_hgt.pt (112 MB)"

echo ""
echo "Downloading cell viability modality..."
download_file "10638836" "$BASE_DIR/views_features_new/cv/cv_cp_data.csv" "cv_cp_data.csv (43 MB)"

echo ""
echo "Downloading transcriptomics modality..."
download_file "10638798" "$BASE_DIR/views_features_new/tx/tx_cp_data_averaged_intermediate.csv" "tx_cp_data_averaged_intermediate.csv (1 GB)"
download_file "10897711" "$BASE_DIR/views_features_new/tx/tx_cp_metadata_for_adapting.pkl" "tx_cp_metadata_for_adapting.pkl (38 MB)"

echo ""
echo "Downloading RDKit embeddings..."
download_file "10638814" "$BASE_DIR/views_features_new/tx/embeddings/rdkit2D_embeddings_combined_all_normalized.parquet" "rdkit2D_embeddings_combined_all_normalized.parquet (10 MB)"
download_file "10638808" "$BASE_DIR/views_features_new/tx/embeddings/rdkit2D_embeddings_all_all_normalized.parquet" "rdkit2D_embeddings_all_all_normalized.parquet (10 MB)"

echo ""
echo "Downloading DrugBank DDI data..."
download_file "10638577" "$BASE_DIR/polypharmacy_new/DrugBank/drugbank_ddi_directed_final_label_map.pkl" "drugbank_ddi_directed_final_label_map.pkl"

# Note: Files are .tab (tab-separated) on Dataverse, download and convert to .csv
download_file "10638670" "$BASE_DIR/polypharmacy_new/DrugBank/split_by_pairs/train_df.tab" "train_df.tab (17 MB)"
download_file "10638631" "$BASE_DIR/polypharmacy_new/DrugBank/split_by_pairs/val_df.tab" "val_df.tab (2 MB)"
download_file "10638553" "$BASE_DIR/polypharmacy_new/DrugBank/split_by_pairs/test_df.tab" "test_df.tab (5 MB)"

echo ""
echo "Converting .tab to .csv format..."
for f in "$BASE_DIR/polypharmacy_new/DrugBank/split_by_pairs/"*.tab; do
    if [ -f "$f" ]; then
        csv_file="${f%.tab}.csv"
        if [ ! -f "$csv_file" ]; then
            # Convert tab to comma separated
            sed 's/\t/,/g' "$f" > "$csv_file"
            echo "Converted: $(basename $csv_file)"
        fi
    fi
done

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Total expected size: ~1.9 GB"
echo ""
echo "Verify with: tree -h $BASE_DIR"