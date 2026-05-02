"""
config.py
=========
Central configuration for CV Industrial Suite.
All thresholds, class IDs, colors, and paths live here.
Modify this file to tune inference behavior without touching module logic.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "static" / "outputs"
MODEL_PATH  = "yolov8n.pt"   # auto-downloaded by ultralytics on first run
QC_MODEL_PATH = "yolo_textile_custom.pt" # path to the custom textile QC model
TEMP_DIR    = BASE_DIR / "static" / "temp"   # temporary files (e.g. uploaded videos)
# ── Inference thresholds ───────────────────────────────────────────────────
CONF_PERSONNEL  = 0.40   # minimum confidence for person detections
CONF_SAFETY     = 0.40   # minimum confidence for safety module
CONF_VEHICLE    = 0.35   # lower threshold to catch partially visible vehicles
CONF_DEFECT     = 0.30   # minimum confidence for textile defect detections

# ── Video processing ───────────────────────────────────────────────────────
VIDEO_SKIP_FRAMES   = 2      # process every Nth frame (1 = all frames)
VIDEO_MAX_FRAMES    = 500    # hard cap on processed frames per video
JPEG_QUALITY        = 88     # output JPEG compression quality (0-100)
CAMERA_FPS_CAP      = 6      # max frames per second sent over WebSocket

# ── COCO class IDs ─────────────────────────────────────────────────────────
CLASS_PERSON = 0

VEHICLE_CLASSES: dict[int, str] = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# ── Safety rules ───────────────────────────────────────────────────────────
SAFETY_RESTRICTED_ZONE_PCT  = 0.10   # left/right % of frame width = restricted
SAFETY_PROXIMITY_PX         = 80     # center-to-center pixels = proximity alert
SAFETY_POSTURE_RATIO        = 1.4    # bbox width/height ratio = possible fall

# ── Annotation colors (BGR) ────────────────────────────────────────────────
# atomcamp brand palette adapted for OpenCV BGR
COLOR_GREEN   = (0, 138, 30)     # #1E8A00 atomcamp green
COLOR_BRIGHT  = (0, 183, 58)     # #3AB700 bright green
COLOR_SAFE    = (0, 183, 58)     # green  → safe personnel
COLOR_UNSAFE  = (0, 80, 255)     # red    → unsafe / violation
COLOR_VEHICLE = (210, 160, 0)    # amber  → vehicle box
COLOR_PLATE   = (255, 200, 0)    # yellow → plate highlight
COLOR_INFO    = (200, 200, 200)  # light gray → neutral info
COLOR_WARN    = (0, 140, 255)    # orange → warning

# ── HUD overlay alpha ──────────────────────────────────────────────────────
HUD_ALPHA = 0.65   # opacity of the dark HUD box (0=transparent, 1=opaque)
