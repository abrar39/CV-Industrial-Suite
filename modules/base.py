"""
modules/base.py
===============
Shared drawing primitives and frame encode/decode helpers
used by all CV modules. No inference logic lives here.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

import config


# ═══════════════════════════════════════════════════════════
# Frame encode / decode
# ═══════════════════════════════════════════════════════════

def encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 JPEG string."""
    _, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    )
    return base64.b64encode(buf).decode()


def decode_frame(b64: str) -> np.ndarray | None:
    """
    Decode a base64 string (with or without data-URI prefix) to a BGR frame.
    Returns None if decoding fails.
    """
    if "," in b64:
        b64 = b64.split(",")[1]
    try:
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════

def draw_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int = 2,
) -> None:
    """Draw a bounding box rectangle."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_label(
    img: np.ndarray,
    text: str,
    x1: int, y1: int,
    color: tuple,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw a filled colour label above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    rect_y1 = max(0, y1 - th - pad * 2)
    rect_y2 = y1
    cv2.rectangle(img, (x1, rect_y1), (x1 + tw + pad * 2, rect_y2), color, -1)
    cv2.putText(
        img, text,
        (x1 + pad, y1 - pad),
        font, font_scale, (0, 0, 0), thickness,
    )


def draw_hud(
    img: np.ndarray,
    lines: list[str],
    color: tuple = config.COLOR_GREEN,
) -> None:
    """
    Draw a semi-transparent HUD box with text lines at the top-left corner.
    Uses atomcamp green as default accent colour.
    """
    h, w = img.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness  = 1
    pad        = 12
    line_gap   = 26

    # Measure widest line
    max_tw = max(
        cv2.getTextSize(l, font, font_scale, thickness)[0][0]
        for l in lines
    )
    box_w = max_tw + pad * 2
    box_h = pad + line_gap * len(lines) + pad // 2

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, config.HUD_ALPHA, img, 1 - config.HUD_ALPHA, 0, img)

    # Accent border
    cv2.rectangle(img, (8, 8), (8 + box_w, 8 + box_h), color, 1)

    # Text lines
    for i, line in enumerate(lines):
        y = 8 + pad + (i + 1) * line_gap - 4
        cv2.putText(img, line, (8 + pad, y), font, font_scale, color, thickness)


def draw_circle_id(
    img: np.ndarray,
    label: str,
    cx: int, cy: int,
    color: tuple,
    radius: int = 14,
) -> None:
    """Draw a filled circle with a short ID label (e.g. 'P01')."""
    cv2.circle(img, (cx, cy), radius, color, -1)
    font_scale = 0.40
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(
        img, label,
        (cx - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
    )


# ═══════════════════════════════════════════════════════════
# Aggregate stats helper (used by video endpoint)
# ═══════════════════════════════════════════════════════════

def aggregate_stats(module: str, frames_data: list[dict]) -> dict:
    """Compute summary statistics across all processed video frames."""
    if not frames_data:
        return {}

    if module == "personnel":
        counts = [d.get("total_persons", 0) for d in frames_data]
        return {
            "avg":    round(sum(counts) / len(counts), 1),
            "max":    max(counts),
            "min":    min(counts),
            "frames": len(frames_data),
        }

    if module == "safety":
        unsafe = [d.get("unsafe_count", 0) for d in frames_data]
        return {
            "total_violation_frames": sum(1 for u in unsafe if u > 0),
            "max_unsafe":             max(unsafe),
            "alert_frames":           sum(1 for u in unsafe if u > 0),
            "frames":                 len(frames_data),
        }

    if module == "vehicle":
        counts = [d.get("total_vehicles", 0) for d in frames_data]
        all_plates: set[str] = set()
        for d in frames_data:
            all_plates.update(d.get("plates", []))
        return {
            "avg_vehicles":  round(sum(counts) / len(counts), 1),
            "max_vehicles":  max(counts),
            "unique_plates": sorted(all_plates),
            "frames":        len(frames_data),
        }

    return {}
