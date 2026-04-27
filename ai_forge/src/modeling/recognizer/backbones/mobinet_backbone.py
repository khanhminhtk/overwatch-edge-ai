from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

from worktree.ai_forge.ai_forge.src.modeling.recognizer.backbones.base_backbone import BaseBackBone

class MobileNetBackBone(BaseBackBone):
    def __init__(
        self,
        device: str,
        in_feature: int | None,
        out_feature: int
    ):
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        super().__init__(
            device=device,
            model=model,
            in_feature=in_feature,
            out_feature=out_feature
        )