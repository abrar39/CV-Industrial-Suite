"""
modules/motor_pdm_thermal.py
============================
Module 05 — Motor PDM Thermal Analysis
--------------------------------------
Analyzes motor thermal images for anomalies using YOLO26 (yolo26l-cls).
This module demonstrates how to integrate a custom-trained YOLO model for classification tasks.
The model is trained to classify thermal images of motor PDMs into categories

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
    Run Motor PDM Thermal Analysis on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    annotated : np.ndarray  frame with drawn annotations
    result    : dict        structured detection data for the dashboard
    """
    model      = get_yolo(task="motor_pdm_thermal")

    # Verify if the correct model is loaded
    logger.info("Model file: %s", model.ckpt_path)
    logger.info("Model classes: %s", model.names)
    logger.info("Model config: %s", config.CONF_MOTOR_PDM_THERMAL)
    # End of verification logs
    results    = model.predict(frame, conf=config.CONF_MOTOR_PDM_THERMAL, verbose=False)[0]
    annotated  = frame.copy()
    detections = []

    # Classification model — no bounding boxes, just class probabilities
    if results.probs is not None:
        # Get top-1 prediction
        top_class_id = int(results.probs.top1)
        top_conf = float(results.probs.top1conf)
        top_label = results.names.get(top_class_id, f"cls{top_class_id}")

        # Get all class probabilities
        class_probs = results.probs.data.cpu().numpy()

        detections.append({
            "id":         1,
            "label":      top_label,
            "confidence": round(top_conf, 3),
            "bbox":       None,
            "width_px":   frame.shape[1],
            "height_px":  frame.shape[0],
        })

        # Draw label on image
        draw_label(annotated, f"{top_label}  {top_conf:.0%}", 20, 30, config.COLOR_GREEN)

        # Draw HUD with top-3 predictions
        top_3_text = []
        for idx in sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)[:3]:
            class_name = results.names.get(idx, f"cls{idx}")
            class_conf = float(class_probs[idx])
            top_3_text.append(f"{class_name}  {class_conf:.1%}")

        draw_hud(annotated, top_3_text, config.COLOR_GREEN)

    return annotated, {
        "module":           "motor_pdm_thermal",
        "total_defects":    1 if results.probs is not None else 0,
        "zone_density":     "SINGLE" if results.probs is not None else "NONE",
        "high_confidence":  1 if (results.probs is not None and float(results.probs.top1conf) > 0.70) else 0,
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
