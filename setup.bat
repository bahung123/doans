@echo off
echo HUST RAG - Setup Script
echo.

echo [1/5] Checking Python...
python --version 2>nul || (echo Error: Python not found & exit /b 1)

echo [2/5] Creating virtual environment...
if exist "venv" if not exist "venv\Scripts\activate.bat" (
    echo Removing broken venv...
    rmdir /s /q venv
)
if not exist "venv" (
    python -m venv venv || (echo Error: Cannot create venv & exit /b 1)
)

echo [3/5] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo [4/5] Downloading data...
if not exist "data\chroma" python scripts\download_data.py

echo [5/5] Creating .env...
if not exist ".env" (
    echo SILICONFLOW_API_KEY=your_key_here> .env
    echo GROQ_API_KEY=your_key_here>> .env
    echo Please edit .env with your API keys
)

echo.
echo Setup complete!
echo Run: venv\Scripts\activate ^& python scripts\run_app.py
pause
