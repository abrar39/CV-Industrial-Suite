"""
modules/vehicle.py
==================
Module 03 — Vehicle Detection & Number Plate Recognition
----------------------------------------------------------
Detects vehicles (Car, Motorcycle, Bus, Truck) using YOLOv8 COCO weights,
then extracts and reads the license plate text from the bottom third of
each vehicle bounding box using EasyOCR.

OCR pipeline (per vehicle):
    1. Crop bottom 32% of bounding box (typical plate region)
    2. Upscale 2.5× with bicubic interpolation
    3. Bilateral filter (denoise while preserving edges)
    4. Otsu binarisation for contrast
    5. EasyOCR read with alphanumeric allowlist

When EasyOCR is unavailable the module still runs — plates show as "N/A".

Public API
----------
    run(frame: np.ndarray) -> tuple[np.ndarray, dict]
"""

from __future__ import annotations

import cv2
import numpy as np

import config
from models import get_yolo, get_ocr
from modules.base import draw_box, draw_label, draw_hud

# Characters allowed in OCR output (plates only contain these)
_PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "


# ── Public entry point ─────────────────────────────────────────────────────

def run(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run vehicle detection and plate OCR on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    annotated : np.ndarray  frame with drawn annotations
    result    : dict        structured detection data for the dashboard
    """
    model              = get_yolo()
    ocr_reader, ocr_ok = get_ocr()
    results            = model(frame, conf=config.CONF_VEHICLE, verbose=False)[0]
    annotated          = frame.copy()
    detections: list[dict] = []
    vid                = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in config.VEHICLE_CLASSES:
            continue

        vid  += 1
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bh     = y2 - y1
        vtype  = config.VEHICLE_CLASSES[cls]

        # Vehicle bounding box
        draw_box(annotated, x1, y1, x2, y2, config.COLOR_VEHICLE)
        draw_label(annotated, f"{vtype} #{vid}  {conf:.0%}", x1, y1, config.COLOR_VEHICLE)

        # Plate region: bottom 32% of vehicle bbox
        plate_y1   = y2 - int(bh * 0.32)
        plate_crop = frame[max(0, plate_y1):y2, x1:x2]

        plate_text, plate_conf = _read_plate(plate_crop, ocr_reader, ocr_ok)

        if plate_text != "N/A":
            # Highlight plate region on the annotated frame
            cv2.rectangle(annotated, (x1, plate_y1), (x2, y2), config.COLOR_PLATE, 2)
            _draw_plate_label(annotated, plate_text, x1, y2)

        detections.append({
            "id":               vid,
            "type":             vtype,
            "confidence":       round(conf, 3),
            "bbox":             [x1, y1, x2, y2],
            "plate_text":       plate_text,
            "plate_confidence": plate_conf,
        })

    plates_found = [d["plate_text"] for d in detections if d["plate_text"] != "N/A"]
    type_counts  = _count_by_type(detections)

    draw_hud(annotated, [
        f"TOTAL VEHICLES  : {vid}",
        f"PLATES DETECTED : {len(plates_found)}",
        f"OCR ENGINE      : {'ACTIVE' if ocr_ok else 'OFFLINE'}",
    ], config.COLOR_VEHICLE)

    return annotated, {
        "module":          "vehicle",
        "total_vehicles":  vid,
        "plates_detected": len(plates_found),
        "plates":          plates_found,
        "vehicle_types":   type_counts,
        "ocr_active":      ocr_ok,
        "detections":      detections,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _read_plate(
    crop: np.ndarray,
    reader,
    ocr_available: bool,
) -> tuple[str, float]:
    """
    Run OCR on a plate crop and return (text, confidence).
    Returns ("N/A", 0.0) on failure or when OCR is disabled.
    """
    if not ocr_available or reader is None:
        return "N/A", 0.0

    if crop is None or crop.size == 0:
        return "N/A", 0.0

    if crop.shape[0] < 8 or crop.shape[1] < 16:
        return "N/A", 0.0

    try:
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray  = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray  = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        hits = reader.readtext(
            thresh,
            detail=1,
            allowlist=_PLATE_ALLOWLIST,
        )

        if not hits:
            return "N/A", 0.0

        best = max(hits, key=lambda h: h[2])
        text = best[1].upper().strip().replace(" ", "")
        conf = best[2]

        if conf > 0.15 and len(text) >= 2:
            return text, round(conf, 3)

    except Exception:
        pass

    return "N/A", 0.0


def _draw_plate_label(
    img: np.ndarray,
    text: str,
    x: int,
    y_bottom: int,
) -> None:
    """Draw a plate text label below the vehicle bounding box."""
    padded = f"  {text}  "
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.6
    thick  = 2
    (tw, th), _ = cv2.getTextSize(padded, font, scale, thick)
    py = y_bottom + 24
    cv2.rectangle(img, (x, py - th - 4), (x + tw, py + 4), config.COLOR_PLATE, -1)
    cv2.putText(img, padded, (x, py), font, scale, (0, 0, 0), thick)


def _count_by_type(detections: list[dict]) -> dict[str, int]:
    """Return a {vehicle_type: count} summary."""
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    return counts
