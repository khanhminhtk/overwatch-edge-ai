import os
import sys

# Ensure the service package root is on sys.path so `from src.xxx` imports work
_service_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)
