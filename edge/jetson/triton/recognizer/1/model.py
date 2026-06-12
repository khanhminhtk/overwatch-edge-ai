from __future__ import annotations

import json

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:  # pragma: no cover - local unit tests do not load Triton.
    pb_utils = None


VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"


def _token_to_char(token: int) -> str:
    return VOCAB[token % len(VOCAB)]


def generate_recognizer_outputs(
    crops: np.ndarray,
    sequence_length: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(crops.shape[0])
    logits = np.zeros((batch_size, sequence_length, len(VOCAB)), dtype=np.float32)
    confidence = np.zeros((batch_size, 1), dtype=np.float32)
    text = np.empty((batch_size, 1), dtype=object)

    for batch_idx in range(batch_size):
        crop = crops[batch_idx]
        mean_value = float(np.clip(np.mean(crop), 0.0, 1.0))
        energy = float(np.std(crop))
        seed = int(round(mean_value * 1000.0 + energy * 100.0 + batch_idx * 7.0))

        tokens: list[int] = []
        for step_idx in range(sequence_length):
            token = (seed + step_idx * 3) % (len(VOCAB) - 1)
            tokens.append(token)
            logits[batch_idx, step_idx, token] = np.float32(4.0 + mean_value * 2.0)
            logits[batch_idx, step_idx, -1] = np.float32(0.5 + (1.0 - mean_value))

        decoded = "".join(_token_to_char(token) for token in tokens).rstrip("_")
        text[batch_idx, 0] = decoded.encode("utf-8")
        confidence[batch_idx, 0] = np.float32(min(0.99, 0.35 + mean_value * 0.60 + energy * 0.05))

    return text, confidence, logits


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args["model_config"])
        parameters = config.get("parameters", {})
        raw_sequence_length = parameters.get("sequence_length", {}).get("string_value", "12")
        self.sequence_length = int(raw_sequence_length)

    def execute(self, requests):
        if pb_utils is None:  # pragma: no cover - only relevant inside Triton.
            raise RuntimeError("triton_python_backend_utils is required inside Triton runtime")

        responses = []
        for request in requests:
            crop_tensor = pb_utils.get_input_tensor_by_name(request, "CROPS")
            crops = crop_tensor.as_numpy().astype(np.float32, copy=False)

            text, confidence, logits = generate_recognizer_outputs(
                crops,
                sequence_length=self.sequence_length,
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("TEXT", text),
                        pb_utils.Tensor("CONFIDENCE", confidence),
                        pb_utils.Tensor("LOGITS", logits),
                    ]
                )
            )

        return responses

    def finalize(self) -> None:
        return None
