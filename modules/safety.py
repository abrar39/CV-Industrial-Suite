"""
modules/safety.py
=================
Module 02 — Health & Safety Compliance
----------------------------------------
Detects persons using YOLOv8 and applies three heuristic
safety rules to each detection:

  Rule 1 — Restricted Zone  : person in outer N% of frame width
  Rule 2 — Proximity Alert  : two persons within SAFETY_PROXIMITY_PX of each other
  Rule 3 — Posture Alert    : bounding box width > height × ratio (possible fall)

Alert levels
  CRITICAL  → unsafe_count > 3
  WARNING   → unsafe_count > 0
  ALL CLEAR → no violations

NOTE
----
For production PPE detection (helmets, vests, gloves) swap the model loaded
in models.py with a specialist YOLOv8 weights file and update the class-ID
logic in _check_violations() below.

Public API
----------
    run(frame: np.ndarray) -> tuple[np.ndarray, dict]
"""

from __future__ import annotations

import cv2
import numpy as np

import config
from models import get_yolo
from modules.base import draw_box, draw_label, draw_hud


# ── Public entry point ─────────────────────────────────────────────────────

def run(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run health & safety analysis on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    annotated : np.ndarray  frame with drawn annotations
    result    : dict        structured detection data for the dashboard
    """
    model     = get_yolo()
    results   = model.track(frame, conf=config.CONF_SAFETY, persist=True, verbose=False)[0]
    h_img, w_img = frame.shape[:2]
    annotated = frame.copy()

    # Collect raw person bounding boxes
    raw_persons: list[dict] = []
    for box in results.boxes:
        if int(box.cls[0]) != config.CLASS_PERSON:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        tid = int(box.id[0]) if box.id is not None else len(raw_persons) + 1
        raw_persons.append({
            "id":   tid,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "cx":   (x1 + x2) // 2,
            "cy":   (y1 + y2) // 2,
            "bw":   bw,
            "bh":   bh,
        })

    # Evaluate violations and annotate
    detections        = []
    all_violations    = []
    safe_count        = 0
    unsafe_count      = 0

    for i, person in enumerate(raw_persons):
        violations = _check_violations(person, i, raw_persons, w_img)
        x1, y1, x2, y2 = person["bbox"]
        bh = person["bh"]

        is_safe = len(violations) == 0
        color   = config.COLOR_SAFE if is_safe else config.COLOR_UNSAFE
        label   = f"{'SAFE' if is_safe else 'UNSAFE'}  {person['conf']:.0%}"

        draw_box(annotated, x1, y1, x2, y2, color)

        # Highlight head region
        head_h = max(1, int(bh * 0.22))
        cv2.rectangle(annotated, (x1, y1), (x2, y1 + head_h), color, 1)

        draw_label(annotated, label, x1, y1, color, font_scale=0.55)

        # Violation text below box
        if violations:
            vtext = " | ".join(violations)[:45]
            cv2.putText(
                annotated, vtext,
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.COLOR_UNSAFE, 1,
            )
            unsafe_count += 1
            all_violations.extend(violations)
        else:
            safe_count += 1

        detections.append({
            "id":         person["id"],
            "status":     "safe" if is_safe else "unsafe",
            "confidence": round(person["conf"], 3),
            "bbox":       person["bbox"],
            "violations": list(set(violations)),
        })

    alert_level  = _classify_alert(unsafe_count)
    hud_color    = config.COLOR_UNSAFE if unsafe_count > 0 else config.COLOR_SAFE

    draw_hud(annotated, [
        f"STATUS          : {alert_level}",
        f"SAFE PERSONNEL  : {safe_count}",
        f"UNSAFE PERSONNEL: {unsafe_count}",
        f"TOTAL VIOLATIONS: {len(all_violations)}",
    ], hud_color)

    return annotated, {
        "module":        "safety",
        "alert_level":   alert_level,
        "total_persons": len(detections),
        "safe_count":    safe_count,
        "unsafe_count":  unsafe_count,
        "violations":    list(set(all_violations)),
        "detections":    detections,
    }


# ── Private helpers ────────────────────────────────────────────────────────

def _check_violations(
    person: dict,
    idx: int,
    all_persons: list[dict],
    frame_width: int,
) -> list[str]:
    """
    Apply all safety rules to a single person detection.
    Returns a (possibly empty) list of violation strings.
    """
    violations: list[str] = []

    x1, _, x2, _ = person["bbox"]
    bw, bh = person["bw"], person["bh"]

    # Rule 1 — Restricted zone (left/right margins)
    margin = frame_width * config.SAFETY_RESTRICTED_ZONE_PCT
    if x1 < margin or x2 > frame_width - margin:
        violations.append("Restricted Zone")

    # Rule 2 — Proximity alert
    for j, other in enumerate(all_persons):
        if j == idx:
            continue
        dist = _euclidean(person["cx"], person["cy"], other["cx"], other["cy"])
        if dist < config.SAFETY_PROXIMITY_PX:
            violations.append("Proximity Alert")
            break  # one alert per person is enough

    # Rule 3 — Posture anomaly (wide bounding box → possible fall)
    if bh > 0 and (bw / bh) > config.SAFETY_POSTURE_RATIO:
        violations.append("Posture Alert — Possible Fall")

    return violations


def _euclidean(x1: int, y1: int, x2: int, y2: int) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _classify_alert(unsafe_count: int) -> str:
    if unsafe_count > 3:
        return "CRITICAL"
    if unsafe_count > 0:
        return "WARNING"
    return "ALL CLEAR"
