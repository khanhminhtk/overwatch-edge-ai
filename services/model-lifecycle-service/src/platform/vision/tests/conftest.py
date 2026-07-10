import os
import sys

_service_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)
