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

# 3. Check if Dataset Exists and Compile PDFs
echo "🔍 Verifying Dataset and PDFs..."
# data_gen.py is smart and will instantly skip images it has already generated,
# and will compile any missing PDFs!
python data_gen.py

# 4. Start YOLO Multi-GPU Training
echo "🧠 Starting YOLO Training Script..."
python train_yolo.py

echo "🎉 Pipeline finished successfully!"
