#!/bin/bash
# Stop execution if any command fails
set -e

echo "================================================================="
echo "🚀 Starting Full Linux YOLO OBB Dataset Gen & Training Pipeline..."
echo "================================================================="

# 1. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "📦 Activating virtual environment..."
source .venv/bin/activate

echo "📥 Upgrading pip..."
pip install --upgrade pip

echo "📥 Installing dependencies..."
# Use opencv-python-headless to prevent libGL/GUI issues on headless remote servers
pip install -r requirements.txt
pip install tqdm shapely

# 2. Generate Dataset (50 PDFs, 5 pages per PDF, 30-50 expressions per page)
echo "📂 Generating YOLO Dataset (50 PDFs)..."
python3 data_gen.py --num-pdfs 50 --pages-per-pdf 5 --min-expr 30 --max-expr 50

# 3. Train YOLO OBB model for 30 epochs
echo "🧠 Starting YOLO OBB model training for 30 epochs..."
python3 train_yolo.py --epochs 30 --name yolo_50_pdfs_30_epochs

echo "================================================================="
echo "🎉 Pipeline finished successfully!"
echo "Model weights are saved in: runs/obb/yolo_50_pdfs_30_epochs/weights/best.pt"
echo "================================================================="
