from __future__ import annotations

from typing import Any

import pandas as pd
from mlflow.pyfunc import PythonModel


class RecognizerCheckpointPyfuncModel(PythonModel):
    def __init__(self, checkpoint_path: str, metadata: dict[str, int | float | str | None]):
        self._checkpoint_path = checkpoint_path
        self._metadata = metadata

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        input_frame = pd.DataFrame(model_input)
        num_rows = len(input_frame.index)
        return pd.DataFrame(
            {
                "text": [""] * num_rows,
                "confidence": [0.0] * num_rows,
                "checkpoint_path": [self._checkpoint_path] * num_rows,
                "checkpoint_version": [str(self._metadata.get("version"))] * num_rows,
            }
        )
