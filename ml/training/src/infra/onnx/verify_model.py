from pathlib import Path

import cv2
import torch

from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.infra.modeling.detection.yolo import YoloTrainer

def validate_pt(
    path_test: str,
    model: RecognizerCTCModel | YoloTrainer,
    mode_ultralytics: bool = True,
) -> bool:
    image_path = Path(path_test)
    if not image_path.is_file():
        print(f"Input image not found: {image_path}")
        return False

    if not mode_ultralytics and isinstance(model, RecognizerCTCModel):
        try:
            model.eval()
            data_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if data_bgr is None:
                print(f"Failed to read image: {image_path}")
                return False

            data_rgb = cv2.cvtColor(data_bgr, cv2.COLOR_BGR2RGB)
            # Build 5D tensor [B, P, C, H, W]
            data_tensor = torch.from_numpy(data_rgb).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float() / 255.0
            model_device = next(model.parameters()).device
            data_tensor = data_tensor.to(model_device)
            with torch.no_grad():
                output = model(data_tensor)
            print("Recognizer model output:", output)
            return True
        except Exception as e:
            print(f"Error validating recognizer model: {e}")
            return False
    elif mode_ultralytics and isinstance(model, YoloTrainer):
        try:
            result = model.predict(str(image_path))
            print("Yolo model result:", result)
            return True
        except Exception as e:
            print(f"Error validating Yolo model: {e}")
            return False

    print(
        "Invalid validator mode/model pairing: "
        f"mode_ultralytics={mode_ultralytics}, model_type={type(model).__name__}"
    )
    return False


