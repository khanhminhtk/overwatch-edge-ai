from __future__ import annotations

import json

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:  # pragma: no cover - local unit tests do not load Triton.
    pb_utils = None


def generate_detector_outputs(
    images: np.ndarray,
    max_detections: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(images.shape[0])
    boxes = np.zeros((batch_size, max_detections, 4), dtype=np.float32)
    scores = np.zeros((batch_size, max_detections), dtype=np.float32)
    class_ids = np.zeros((batch_size, max_detections), dtype=np.int32)
    detections = np.zeros((batch_size, 1), dtype=np.int32)

    for batch_idx in range(batch_size):
        image = images[batch_idx]
        mean_value = float(np.clip(np.mean(image), 0.0, 1.0))
        count = min(max_detections, max(1, int(round(mean_value * max_detections))))
        detections[batch_idx, 0] = count

        for det_idx in range(count):
            x1 = 0.05 + 0.08 * det_idx
            y1 = 0.10 + 0.06 * det_idx
            x2 = min(0.95, x1 + 0.20 + mean_value * 0.10)
            y2 = min(0.95, y1 + 0.12 + mean_value * 0.08)
            boxes[batch_idx, det_idx] = np.array([x1, y1, x2, y2], dtype=np.float32)
            scores[batch_idx, det_idx] = np.float32(min(0.99, 0.45 + mean_value * 0.50 - det_idx * 0.03))
            class_ids[batch_idx, det_idx] = np.int32(det_idx % 2)

    return boxes, scores, class_ids, detections


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        config = json.loads(args["model_config"])
        parameters = config.get("parameters", {})
        raw_max_detections = parameters.get("max_detections", {}).get("string_value", "8")
        self.max_detections = int(raw_max_detections)

    def execute(self, requests):
        if pb_utils is None:  # pragma: no cover - only relevant inside Triton.
            raise RuntimeError("triton_python_backend_utils is required inside Triton runtime")

        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            images = image_tensor.as_numpy().astype(np.float32, copy=False)

            boxes, scores, class_ids, detections = generate_detector_outputs(
                images,
                max_detections=self.max_detections,
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("BOXES", boxes),
                        pb_utils.Tensor("SCORES", scores),
                        pb_utils.Tensor("CLASS_IDS", class_ids),
                        pb_utils.Tensor("DETECTIONS", detections),
                    ]
                )
            )

        return responses

    def finalize(self) -> None:
        return None
