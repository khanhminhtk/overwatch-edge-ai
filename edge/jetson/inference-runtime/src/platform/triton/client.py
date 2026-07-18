from __future__ import annotations

from typing import Any

import numpy as np

from .config import TritonConfig


class TritonClient:
    """Implements the model port used by detector and recognizer services."""

    def __init__(self, config: TritonConfig) -> None:
        self._config = config
        grpcclient = _grpcclient()
        self._client = grpcclient.InferenceServerClient(
            url=config.url, verbose=config.verbose
        )

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not inputs:
            raise ValueError("Triton inference requires at least one input")

        triton_inputs: list[Any] = []
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            request_input = _grpcclient().InferInput(
                name, array.shape, _triton_dtype(array)
            )
            request_input.set_data_from_numpy(array)
            triton_inputs.append(request_input)

        result = self._client.infer(
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            inputs=triton_inputs,
        )
        return {
            output.name: result.as_numpy(output.name)
            for output in result.get_response().outputs
        }


def _triton_dtype(array: np.ndarray) -> str:
    dtype = _grpcclient().np_to_triton_dtype(array.dtype)
    if dtype is None:
        raise ValueError(f"unsupported Triton input dtype: {array.dtype}")
    return dtype


def _grpcclient() -> Any:
    try:
        import tritonclient.grpc as grpcclient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Triton gRPC client is unavailable; install project dependencies first"
        ) from exc
    return grpcclient
