from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn

from src.infra.modeling.recognizer.backbones.torch_base_backbone import TorchBaseBackBone

class MobileNetBackBone(TorchBaseBackBone):
    def __init__(
        self,
        device: str,
        in_feature: int | None,
        out_feature: int
    ):
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        extractor_out_feature = model.classifier[0].in_features
        model.classifier = nn.Identity()
        super().__init__(
            device=device,
            model=model,
            in_feature=in_feature if in_feature is not None else extractor_out_feature,
            out_feature=out_feature
        )
