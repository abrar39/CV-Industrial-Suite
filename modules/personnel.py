"""
modules/personnel.py
====================
Module 01 — Personnel Detection
--------------------------------
Detects and counts people in a frame using YOLOv8.
Assigns a sequential ID badge to each detected person and
classifies the zone density as HIGH / MEDIUM / LOW.

Public API
----------
    run(frame: np.ndarray) -> tuple[np.ndarray, dict]
        Returns the annotated frame and a structured result dict.
"""

from __future__ import annotations

import cv2
import numpy as np

import config
from models import get_yolo
from modules.base import draw_box, draw_label, draw_circle_id, draw_hud


# ── Public entry point ─────────────────────────────────────────────────────

def run(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run personnel detection on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    annotated : np.ndarray  frame with drawn annotations
    result    : dict        structured detection data for the dashboard
    """
    model      = get_yolo()
    results    = model(frame, conf=config.CONF_PERSONNEL, verbose=False)[0]
    annotated  = frame.copy()
    detections = []
    pid        = 0

    for box in results.boxes:
        if int(box.cls[0]) != config.CLASS_PERSON:
            continue

        pid  += 1
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1

        # Bounding box
        draw_box(annotated, x1, y1, x2, y2, config.COLOR_GREEN)

        # Label with confidence
        draw_label(annotated, f"Person  {conf:.0%}", x1, y1, config.COLOR_GREEN)

        # ID badge circle at top-center of box
        cx = x1 + bw // 2
        draw_circle_id(annotated, f"P{pid:02d}", cx, y1 - 14, config.COLOR_BRIGHT)

        detections.append({
            "id":         pid,
            "confidence": round(conf, 3),
            "bbox":       [x1, y1, x2, y2],
            "width_px":   bw,
            "height_px":  bh,
        })

    density      = _classify_density(pid)
    high_conf_n  = sum(1 for d in detections if d["confidence"] > 0.70)

    draw_hud(annotated, [
        f"PERSONNEL COUNT : {pid}",
        f"ZONE DENSITY    : {density}",
        f"HIGH CONF(>70%) : {high_conf_n}",
    ], config.COLOR_GREEN)

    return annotated, {
        "module":           "personnel",
        "total_persons":    pid,
        "zone_density":     density,
        "high_confidence":  high_conf_n,
        "detections":       detections,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _classify_density(count: int) -> str:
    """Classify zone density based on person count."""
    if count > 10:
        return "HIGH"
    if count > 4:
        return "MEDIUM"
    return "LOW"
