@echo off
TITLE CV Industrial Suite v2 — atomcamp

echo ================================================
echo   CV INDUSTRIAL SUITE v2  ^|  atomcamp
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause & exit /b 1
)
echo [1/4] Python found.

:: Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install core deps
echo [3/4] Installing dependencies...
pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0" python-multipart==0.0.9 ^
            ultralytics==8.2.18 opencv-python==4.9.0.80 numpy==1.26.4 Pillow==10.3.0 --quiet
pip install easyocr --quiet

:: Launch
echo [4/4] Starting server...
echo.
echo   Open browser: http://localhost:8000
echo.
python app.py
pause
