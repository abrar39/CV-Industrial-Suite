"""
modules/textile_qc.py
====================
Module 04 — Textile Quality Control
-----------------------------------
Inspects textile surfaces for defects using YOLOv8.
Assigns a sequential ID badge to each detected defect and
classifies the zone density as HIGH / MEDIUM / LOW.

Public API
----------
    run(frame: np.ndarray) -> tuple[np.ndarray, dict]
        Returns the annotated frame and a structured result dict.
"""

from __future__ import annotations
from venv import logger

import cv2
import numpy as np

import config
from models import get_yolo
from modules.base import draw_box, draw_label, draw_circle_id, draw_hud


# ── Public entry point ─────────────────────────────────────────────────────

def run(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run Textile Defect Identification on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    annotated : np.ndarray  frame with drawn annotations
    result    : dict        structured detection data for the dashboard
    """
    model      = get_yolo(task="textile_qc")

    # Verify if the correct model is loaded
    logger.info("Model file: %s", model.ckpt_path)
    logger.info("Model classes: %s", model.names)
    logger.info("Model config: %s", config.CONF_DEFECT)
    # End of verification logs
    results    = model(frame, conf=config.CONF_DEFECT, verbose=False)[0]
    annotated  = frame.copy()
    detections = []
    pid        = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names.get(cls_id, f"cls{cls_id}") # use the model's class names
        conf = float(box.conf[0])
        

        pid  += 1
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1

        # Bounding box
        draw_box(annotated, x1, y1, x2, y2, config.COLOR_GREEN)

        # Label with confidence
        draw_label(annotated, f"{label}  {conf:.0%}", x1, y1, config.COLOR_GREEN)

        # ID badge circle at top-center of box
        cx = x1 + bw // 2
        draw_circle_id(annotated, f"D{pid:02d}", cx, y1 - 14, config.COLOR_BRIGHT)

        detections.append({
            "id":         pid,
            "label":      label,
            "confidence": round(conf, 3),
            "bbox":       [x1, y1, x2, y2],
            "width_px":   bw,
            "height_px":  bh,
        })

    density      = _classify_density(pid)
    high_conf_n  = sum(1 for d in detections if d["confidence"] > 0.70)

    draw_hud(annotated, [
        f"DEFECT COUNT : {pid}",
        f"ZONE DENSITY    : {density}",
        f"HIGH CONF(>70%) : {high_conf_n}",
    ], config.COLOR_GREEN)

    return annotated, {
        "module":           "textile_qc",
        "total_defects":    pid,
        "zone_density":     density,
        "high_confidence":  high_conf_n,
        "detections":       detections,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _classify_density(count: int) -> str:
    """Classify zone density based on defect count."""
    if count > 10:
        return "HIGH"
    if count > 4:
        return "MEDIUM"
    return "LOW"
