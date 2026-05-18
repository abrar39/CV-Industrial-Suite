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
from modules.textile_qc import run as run_textile_qc # import the textile QC module runner
from modules.motor_pdm_thermal import run as run_motor_pdm_thermal # import the motor PDM thermal module runner

# Registry: slug → callable(frame: np.ndarray) → (annotated_frame, result_dict)
REGISTRY: dict = {
    "personnel": run_personnel,
    "safety":    run_safety,
    "vehicle":   run_vehicle,
    "textile_qc": run_textile_qc, # added textile QC module to registry
    "motor_pdm_thermal": run_motor_pdm_thermal, # added motor PDM thermal module to registry
}

__all__ = ["REGISTRY"]
