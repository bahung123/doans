#!/bin/bash
set -e

echo "HUST RAG - Setup Script"
echo ""

echo "[1/5] Checking Python..."
python3 --version || { echo "Error: Python 3.10+ required"; exit 1; }

echo "[2/5] Creating virtual environment..."
[ -d "venv" ] || python3 -m venv venv

echo "[3/5] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[4/5] Downloading data..."
[ -d "data/chroma" ] || python scripts/download_data.py

echo "[5/5] Creating .env..."
if [ ! -f ".env" ]; then
    echo "SILICONFLOW_API_KEY=your_key_here" > .env
    echo "GROQ_API_KEY=your_key_here" >> .env
    echo "Please edit .env with your API keys"
fi

echo ""
echo "Setup complete!"
echo "Run: source venv/bin/activate && python scripts/run_app.py"
