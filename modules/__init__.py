"""
modules/__init__.py
===================
Module registry — maps module slug → runner function.
Adding a new CV module only requires:
  1. Create modules/mymodule.py with a `run(frame)` function
  2. Register it here.
"""

from modules.personnel import run as run_personnel
from modules.safety     import run as run_safety
from modules.vehicle    import run as run_vehicle

# Registry: slug → callable(frame: np.ndarray) → (annotated_frame, result_dict)
REGISTRY: dict = {
    "personnel": run_personnel,
    "safety":    run_safety,
    "vehicle":   run_vehicle,
}

__all__ = ["REGISTRY"]
