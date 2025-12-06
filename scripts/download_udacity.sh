#!/bin/bash
# Download Udacity Self-Driving Car Dataset
# Multiple sources available - choose based on your preference

set -e

DATASET_DIR="$HOME/datasets/udacity_driving"
mkdir -p "$DATASET_DIR"

echo "================================================"
echo "Udacity Self-Driving Car Dataset Downloader"
echo "================================================"
echo ""
echo "Choose download method:"
echo "1) Kaggle (requires kaggle CLI)"
echo "2) Direct download (original links)"
echo "3) Manual instructions"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Downloading from Kaggle..."
        if ! command -v kaggle &> /dev/null; then
            echo "Installing Kaggle CLI..."
            pip install --break-system-packages kaggle
        fi

        echo "Please ensure ~/.kaggle/kaggle.json exists with your API key"
        echo "Get it from: https://www.kaggle.com/settings -> Create New API Token"
        read -p "Press Enter when ready..."

        cd "$DATASET_DIR"
        kaggle datasets download -d sshikamaru/udacity-self-driving-car-dataset
        unzip -q udacity-self-driving-car-dataset.zip
        echo "Download complete!"
        ;;

    2)
        echo "Attempting direct download..."
        cd "$DATASET_DIR"

        # Try original Udacity link
        echo "Downloading Track 1 data..."
        wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

        if [ -f "data.zip" ]; then
            unzip -q data.zip
            echo "Download complete!"
        else
            echo "Direct download failed. Try Kaggle or manual method."
        fi
        ;;

    3)
        echo ""
        echo "Manual Download Instructions:"
        echo "============================="
        echo ""
        echo "Option A - Kaggle (Easiest):"
        echo "  1. Visit: https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset"
        echo "  2. Click 'Download'"
        echo "  3. Extract to: $DATASET_DIR"
        echo ""
        echo "Option B - Roboflow (Cleaned, smaller):"
        echo "  1. Visit: https://public.roboflow.com/object-detection/self-driving-car"
        echo "  2. Choose format (COCO JSON recommended)"
        echo "  3. Download and extract to: $DATASET_DIR"
        echo ""
        echo "Option C - Academic Torrents:"
        echo "  1. Visit: https://academictorrents.com/details/bcde779f81adbaae45ef69f9dd07f3e76eab3b27"
        echo "  2. Download torrent"
        echo "  3. Save to: $DATASET_DIR"
        echo ""
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Dataset directory: $DATASET_DIR"
echo "Next steps:"
echo "  1. Check data format"
echo "  2. Run preprocessing script"
echo "  3. Start training"
