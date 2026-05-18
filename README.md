# CV Industrial Suite v2 — atomcamp

AI-powered computer vision platform for manufacturing & facilities.
Built with **YOLOv8 · FastAPI · EasyOCR** — branded for atomcamp.

---

## Project Structure

```
cv_suite_v2/
│
├── app.py               ← FastAPI entry point (routes only)
├── config.py            ← All constants, thresholds, colors
├── models.py            ← Singleton model loader (YOLO + OCR)
│
├── modules/
│   ├── __init__.py      ← Module registry (REGISTRY dict)
│   ├── base.py          ← Shared drawing & encode/decode utils
│   ├── personnel.py     ← Module 01: Personnel Detection
│   ├── safety.py        ← Module 02: Health & Safety
│   ├── vehicle.py       ← Module 03: Vehicle & Number Plates
|   └── textile_qc.py    ← Module 04: Textile Quality Control
│
├── static/
│   ├── index.html       ← Landing page (atomcamp branded)
│   ├── module.html      ← Module workspace
│   └── outputs/         ← Processed video files (auto-created)
│
├── requirements.txt
├── run.bat              ← Windows launcher
└── run.sh               ← Linux / macOS launcher
```

---

## Quick Start

### Windows
```bat
run.bat
```
Then open **http://localhost:8000**

### Linux / macOS
```bash
bash run.sh
```
Then open **http://localhost:8000**

### Manual
```bash
pip install -r requirements.txt
python app.py          # Windows
python3 app.py         # Linux/macOS
```

---

## Modules

| # | Module | File | Description |
|---|--------|------|-------------|
| 01 | Personnel Detection | `modules/personnel.py` | Count & track people, zone density |
| 02 | Health & Safety | `modules/safety.py` | Boundary, proximity & posture violations |
| 03 | Vehicle & Plates | `modules/vehicle.py` | Vehicle detection + OCR plate reading |
| 04 | Textile Defect QC | `modules/textile_qc.py` | Detect fabric defects |

Each module exposes a single public function: `run(frame) -> (annotated_frame, result_dict)`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/{module}/image` | Process single image upload |
| POST | `/api/{module}/video` | Process video upload |
| WS   | `/ws/{module}`        | Live camera WebSocket stream |

`module` ∈ `{ personnel, safety, vehicle, textile_qc }`

---

## Adding a New Module

1. Create `modules/mymodule.py` with a `run(frame: np.ndarray)` function
2. Register it in `modules/__init__.py`:
   ```python
   from modules.mymodule import run as run_mymodule
   REGISTRY["mymodule"] = run_mymodule
   ```
3. Add a card in `static/index.html` linking to `module.html?mod=mymodule`
4. Add a `renderMymodule(d)` function in `static/module.html`

---

## Tuning Inference

All thresholds are in `config.py` — no need to touch module files:

```python
CONF_PERSONNEL = 0.40     # person detection threshold
CONF_VEHICLE   = 0.35     # vehicle detection threshold
SAFETY_PROXIMITY_PX = 80  # proximity alert distance
VIDEO_SKIP_FRAMES = 2     # process every Nth frame
```

## Using a Custom YOLO Model

In `models.py`, change:
```python
_yolo_model = YOLO("yolov8n.pt")
# to:
_yolo_model = YOLO("path/to/custom_weights.pt")
```
