@echo off
REM Start script for Metal Surface Defect Detection system (Windows)

echo ======================================
echo Metal Surface Defect Detection System
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Setup completed!
echo.
echo Available scripts:
echo    1. python download_dataset.py    - Download NEU dataset
echo    2. python train.py               - Train CNN model
echo    3. python predict.py             - Test inference
echo    4. python web\app.py             - Start web UI
echo.
echo To start training:
echo    python train.py
echo.
echo To start web interface:
echo    python web\app.py
echo.
pause
