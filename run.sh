#!/bin/bash
set -e

echo "================================================"
echo "  CV INDUSTRIAL SUITE v2  |  atomcamp"
echo "================================================"

python3 -m pip install --upgrade pip -q
pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0" python-multipart==0.0.9 \
            ultralytics==8.2.18 opencv-python==4.9.0.80 numpy==1.26.4 Pillow==10.3.0 -q
pip install easyocr -q

echo "→ Open browser: http://localhost:8000"
python3 app.py
