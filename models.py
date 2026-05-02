"""
models.py
=========
Singleton model loader.
All modules import `get_yolo()` and `get_ocr()` rather than
loading models themselves — ensures models are loaded exactly once.
"""

from __future__ import annotations

import logging
from typing import Optional

from ultralytics import YOLO
from config import MODEL_PATH, QC_MODEL_PATH

logger = logging.getLogger(__name__)

# ── Internal singletons (never import directly) ────────────────────────────
_yolo_model = None
_yolo_textile_qc_model = None
_ocr_reader  = None
_ocr_available: Optional[bool] = None


def get_yolo(task: str = "generic") -> object:
    """
    Return the shared YOLOv8 model, loading it on first call.
    Parameters:
    task : str  Optional task-specific model variant (e.g. "personnel", "textile")
                "generic" loads the base model (yolov8n.pt) which can be used for all tasks.
                "textile_qc" loads a custom-trained model optimized for textile quality control.
    """
    global _yolo_model, _yolo_textile_qc_model

    if task == "textile_qc":
        if _yolo_textile_qc_model is None:
            logger.info("Loading YOLOv8 model for Textile QC: %s", QC_MODEL_PATH)
            _yolo_textile_qc_model = YOLO(QC_MODEL_PATH)
            logger.info("YOLOv8 Textile QC model loaded ✓")
        return _yolo_textile_qc_model
    else:
        if _yolo_model is None:
            logger.info("Loading YOLOv8 model: %s", MODEL_PATH)
            _yolo_model = YOLO(MODEL_PATH)
            logger.info("YOLOv8 model loaded ✓")
        return _yolo_model


def get_ocr():
    """
    Return (reader, available) tuple.
    reader     → easyocr.Reader instance, or None if unavailable
    available  → bool
    """
    global _ocr_reader, _ocr_available
    if _ocr_available is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR reader (EN) …")
            _ocr_reader   = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ocr_available = True
            logger.info("EasyOCR loaded ✓")
        except Exception as exc:
            logger.warning("EasyOCR unavailable: %s — plate text will show as N/A", exc)
            _ocr_reader   = None
            _ocr_available = False
    return _ocr_reader, _ocr_available
