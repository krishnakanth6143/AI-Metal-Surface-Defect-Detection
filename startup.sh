#!/bin/bash
# Start script for the Metal Surface Defect Detection system

echo "======================================"
echo "Metal Surface Defect Detection System"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔓 Activating virtual environment..."
source venv/Scripts/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup completed!"
echo ""
echo "📂 Available scripts:"
echo "   1. python download_dataset.py    - Download NEU dataset"
echo "   2. python train.py               - Train CNN model"
echo "   3. python predict.py             - Test inference"
echo "   4. python web/app.py             - Start web UI"
echo ""
echo "🚀 To start training:"
echo "   python train.py"
echo ""
echo "🌐 To start web interface:"
echo "   python web/app.py"
echo ""
