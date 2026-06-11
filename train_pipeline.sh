#!/bin/bash
# Stop execution if any command fails
set -e

echo "🚀 Starting Full YOLO Training Pipeline..."

# 1. Pull latest code from GitHub
echo "📥 Pulling latest code from GitHub..."
git pull origin main || echo "⚠️  Git pull failed or not in a git repository, continuing anyway..."

# 2. Create and Activate Virtual Environment
if [ ! -d ".venv" ]; then
    echo "🐍 Creating Virtual Environment..."
    python3 -m venv .venv
fi

echo "📦 Activating Virtual Environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install tqdm ultralytics opencv-python Pillow numpy torch torchvision

# 3. Check if Dataset Exists
echo "🔍 Checking for dataset..."
if [ ! -d "dataset_yolo" ]; then
    echo "🏗️  Dataset not found! Generating the massive 5,000-page dataset..."
    python data_gen.py
else
    # Check if the training directory has roughly the expected number of images
    FILE_COUNT=$(find dataset_yolo/images/train -maxdepth 1 -type f -name "*.png" 2>/dev/null | head -n 4000 | wc -l)
    if [ "$FILE_COUNT" -lt 4000 ]; then
        echo "🏗️  Dataset incomplete (found $FILE_COUNT train images). Regenerating dataset..."
        rm -rf dataset_yolo
        python data_gen.py
    else
        echo "✅ Dataset looks complete ($FILE_COUNT train images found). Skipping generation!"
    fi
fi

# 4. Start YOLO Multi-GPU Training
echo "🧠 Starting YOLO Training Script..."
python train_yolo.py

echo "🎉 Pipeline finished successfully!"
