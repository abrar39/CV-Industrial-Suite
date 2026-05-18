"""
app.py
======
CV Industrial Suite — FastAPI application entry point.
This file contains ONLY routing and request handling.
All inference logic lives in modules/  and utilities in modules/base.py.

Routes
------
POST /api/{module}/image   → process a single uploaded image
POST /api/{module}/video   → process an uploaded video file
WS   /ws/{module}          → live WebSocket camera stream
GET  /                     → serves static/index.html (landing page)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess

import config
from modules import REGISTRY
from modules.base import encode_frame, decode_frame, aggregate_stats

# Suppress the Windows ConnectionResetError noise from video streaming
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── FastAPI setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="CV Industrial Suite",
    description="YOLOv8-powered computer vision platform for manufacturing & facilities.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.TEMP_DIR.mkdir(parents=True, exist_ok=True) # to hold temporary files


# ── Startup: pre-load models ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 55)
    logger.info("  CV INDUSTRIAL SUITE v2 — STARTING UP")
    logger.info("=" * 55)
    from models import get_yolo, get_ocr
    get_yolo()   # warm up YOLO
    get_yolo("textile_qc") # warm up textile QC model
    get_ocr()    # warm up EasyOCR (or mark as unavailable)
    logger.info("All models ready. Serving at http://localhost:8000")


# ═══════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════

@app.post("/api/{module}/image")
async def infer_image(module: str, file: UploadFile = File(...)):
    """
    Accept a single image, run the requested module, and return
    the annotated JPEG (base64) alongside structured inference data.
    """
    if module not in REGISTRY:
        return JSONResponse({"error": f"Unknown module '{module}'"}, status_code=400)

    raw   = await file.read()
    frame = _decode_upload(raw)
    if frame is None:
        return JSONResponse({"error": "Cannot decode uploaded image."}, status_code=400)

    logger.info("IMAGE  module=%s  size=%s", module, frame.shape[:2])
    annotated, result = REGISTRY[module](frame)

    return JSONResponse({
        "success": True,
        "image":   encode_frame(annotated),
        "data":    result,
    })


@app.post("/api/{module}/video")
async def infer_video(module: str, file: UploadFile = File(...)):
    """
    Accept a video file, process every Nth frame through the requested module,
    write an annotated video, and return a URL to download it.
    """
    if module not in REGISTRY:
        return JSONResponse({"error": f"Unknown module '{module}'"}, status_code=400)

    raw      = await file.read()
    #in_path  = f"/tmp/cv_in_{uuid.uuid4().hex}.mp4"
    in_name = f"cv_in_{uuid.uuid4().hex}.mp4"
    in_path = str(config.TEMP_DIR / in_name) # works on windows as well
    out_name = f"cv_out_{uuid.uuid4().hex}.mp4"
    out_path = str(config.OUTPUT_DIR / out_name)

    with open(in_path, "wb") as f:
        f.write(raw)
        f.flush() # force the os to write buffer to disk
        os.fsync(f.fileno()) # ensure data is fully written before processing

    try:
        all_data, total_frames = _process_video(in_path, out_path, module)
    finally:
        if os.path.exists(in_path):
            os.unlink(in_path)

    json_name = out_name.replace('.mp4', '.json')
    json_path = str(config.OUTPUT_DIR / json_name)
    with open(json_path, "w") as f:
        json.dump(all_data, f)

    stats = aggregate_stats(module, all_data)
    logger.info(
        "VIDEO  module=%s  total=%d  processed=%d",
        module, total_frames, len(all_data),
    )

    return JSONResponse({
        "success":          True,
        "video_url":        f"/outputs/{out_name}",
        "json_url":         f"/outputs/{json_name}",
        "stats":            stats,
        "total_frames":     total_frames,
        "processed_frames": len(all_data),
    })


# ═══════════════════════════════════════════════════════════
# WebSocket — Live Camera
# ═══════════════════════════════════════════════════════════

@app.websocket("/ws/{module}")
async def ws_camera(websocket: WebSocket, module: str):
    """
    Accept base64-encoded camera frames over WebSocket and stream
    back annotated frames with inference results.

    Message format (client → server):
        { "type": "frame", "data": "<base64-jpeg>" }

    Message format (server → client):
        { "type": "result", "image": "<base64-jpeg>", "data": { ... } }
    """
    await websocket.accept()

    if module not in REGISTRY:
        await websocket.close(code=1003)
        return

    logger.info("WS opened  module=%s", module)
    try:
        while True:
            raw_msg = await websocket.receive_text()
            msg     = json.loads(raw_msg)

            if msg.get("type") != "frame":
                continue

            frame = decode_frame(msg["data"])
            if frame is None:
                continue

            annotated, result = REGISTRY[module](frame)

            await websocket.send_text(json.dumps({
                "type":  "result",
                "image": encode_frame(annotated),
                "data":  result,
            }))

    except WebSocketDisconnect:
        logger.info("WS closed  module=%s", module)
    except Exception as exc:
        logger.error("WS error  module=%s  error=%s", module, exc)


# ═══════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════

def _decode_upload(raw: bytes) -> np.ndarray | None:
    """Decode raw bytes to a BGR numpy frame."""
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _process_video(
    in_path: str,
    out_path: str,
    module: str,
) -> tuple[list[dict], int]:
    """
    Read `in_path`, run inference on every Nth frame (per config),
    write annotated output to `out_path`, and return
    (list_of_frame_results, total_frame_count).
    """
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_out = out_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, (W, H))

    all_data: list[dict] = []
    fnum = 0

    try:
        while cap.isOpened() and len(all_data) < config.VIDEO_MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            fnum += 1

            if fnum % config.VIDEO_SKIP_FRAMES != 0:
                writer.write(frame)
                continue

            annotated, data = REGISTRY[module](frame)
            writer.write(annotated)
            all_data.append(data)
    finally:
        cap.release()
        writer.release()
        
    # Re-encode with ffmpeg if available, otherwise just rename
    try:
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-vcodec", "libx264", "-f", "mp4", out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(temp_out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(out_path):
            os.unlink(out_path)
        os.rename(temp_out, out_path)

    return all_data, fnum


# ═══════════════════════════════════════════════════════════
# Static files (must come LAST — catches all unmatched routes)
# ═══════════════════════════════════════════════════════════
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
