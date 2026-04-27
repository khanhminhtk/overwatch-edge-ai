import torch

from src.modeling.recognizer.backbones.mobinet_backbone import MobileNetBackBone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_mobilenet_backbone_forward_shape():
    backbone = MobileNetBackBone(
        device=DEVICE,
        in_feature=None,
        out_feature=128
    )
    x = torch.randn(4, 3, 224, 224)
    y = backbone(x)
    assert y.shape == (4, 1, 128)
    x = torch.randn(4, 2, 3, 224, 224)
    y = backbone(x)
    assert y.shape == (4, 2, 128)

def test_mobilenet_backbone_unfreeze_feature_extractor():
    backbone = MobileNetBackBone(
        device=DEVICE,
        in_feature=None,
        out_feature=128
    )

    for param in backbone.model_.parameters():
        assert param.requires_grad is False

    backbone.unfreeze_feature_extractor(classifier=True)
    for name, param in backbone.model_.named_parameters():
        if "classifier" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    backbone.unfreeze_feature_extractor(classifier=False, num_layers_from_classification=2)
    feature_layers = list(backbone.model_.features.children())
    for i, layer in enumerate(feature_layers):
        for param in layer.parameters():
            if i >= len(feature_layers) - 2:
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False