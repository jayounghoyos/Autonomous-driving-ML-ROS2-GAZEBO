#!/bin/bash
# Download comma2k19 Dataset
# 33 hours of highway driving data from comma.ai

set -e

DATASET_DIR="$HOME/datasets/comma2k19"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "================================================"
echo "comma2k19 Dataset Downloader"
echo "================================================"
echo ""
echo "Dataset: 33 hours, 2019 segments, ~100GB total"
echo "Source: Highway 280, California"
echo ""
echo "Download options:"
echo "1) Full dataset (~100GB) - All 2019 segments"
echo "2) Sample chunk (~10GB) - ~200 segments"
echo "3) Minimal sample - Just example segment for testing"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Downloading full comma2k19 dataset..."
        echo "This will take a while (~100GB)..."

        # Clone the repository with download scripts
        if [ ! -d "comma2k19" ]; then
            git clone https://github.com/commaai/comma2k19.git
        fi

        cd comma2k19

        # Download all chunks
        for i in {0..9}; do
            echo "Downloading chunk $i/10..."
            wget "https://commadataci.blob.core.windows.net/comma2k19/Chunk_$i.tar"
            tar -xf "Chunk_$i.tar"
            rm "Chunk_$i.tar"
        done

        echo "Full dataset downloaded!"
        ;;

    2)
        echo "Downloading sample chunk (~10GB)..."

        if [ ! -d "comma2k19" ]; then
            git clone https://github.com/commaai/comma2k19.git
        fi

        cd comma2k19

        # Download just first chunk
        wget https://commadataci.blob.core.windows.net/comma2k19/Chunk_0.tar
        tar -xf Chunk_0.tar
        rm Chunk_0.tar

        echo "Sample chunk downloaded!"
        ;;

    3)
        echo "Cloning repository with example segment..."

        git clone https://github.com/commaai/comma2k19.git
        cd comma2k19

        echo "Example segment available in repository"
        echo "Check the 'example' directory for sample data"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Dataset location: $DATASET_DIR/comma2k19"
echo ""
echo "Data format:"
echo "  - Camera images (road-facing)"
echo "  - GPS coordinates"
echo "  - IMU data (accelerometer, gyroscope)"
echo "  - CAN bus data (steering, speed, etc.)"
echo ""
echo "Next steps:"
echo "  1. Explore data format (check README)"
echo "  2. Run preprocessing script"
echo "  3. Extract steering angles from CAN data"
