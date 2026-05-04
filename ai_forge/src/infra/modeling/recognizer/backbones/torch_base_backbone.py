from __future__ import annotations

import torch
import torch.nn as nn

from src.domain.ports.recognizer.backbone_port import BackBonePort


def _infer_classifier_dim(model: nn.Module) -> None | int:
    if hasattr(model, "classifier"):
        for module in reversed(list(model.classifier.modules())):
            out_features = getattr(module, "out_features", None)
            if isinstance(out_features, int):
                return out_features

    return None


class TorchBaseBackBone(nn.Module, BackBonePort):
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device | None,
        in_feature: int | None,
        out_feature: int,
    ):
        super().__init__()

        if out_feature <= 0:
            raise ValueError(f"TorchBaseBackBone.__init__ out_feature must be > 0, got {out_feature}")

        self.in_feature = in_feature if in_feature is not None else _infer_classifier_dim(model)
        self.model_ = model
        self._device = device
        self.out_feature = out_feature

        self._mark_backbone_modules_skip_init()

        if self.in_feature is None:
            self.proj = nn.LazyLinear(
                out_features=out_feature,
                device=device,
                bias=False,
            )
        else:
            self.proj = nn.Linear(
                in_features=self.in_feature,
                out_features=out_feature,
                device=device,
                bias=False,
            )

        self.freeze_feature_extractor()

    def _mark_backbone_modules_skip_init(self) -> None:
        for module in self.model_.modules():
            setattr(module, "_skip_custom_init", True)

    def freeze_feature_extractor(self) -> None:
        for param in self.model_.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(
        self,
        classifier: bool,
        num_layers_from_classification: int = 0,
    ) -> None:
        if num_layers_from_classification < 0:
            raise ValueError(
                "TorchBaseBackBone.unfreeze_feature_extractor: "
                f"num_layers_from_classification must be >= 0, got {num_layers_from_classification}"
            )

        if classifier:
            if not hasattr(self.model_, "classifier"):
                raise ValueError(
                    "TorchBaseBackBone.unfreeze_feature_extractor: "
                    "model does not have a classifier attribute"
                )

            for param in self.model_.classifier.parameters():
                param.requires_grad = True

        if num_layers_from_classification == 0:
            return

        if hasattr(self.model_, "features"):
            feature_layers = list(self.model_.features.children())
        else:
            feature_layers = list(self.model_.children())
            classifier_module = getattr(self.model_, "classifier", None)
            if classifier_module is not None:
                feature_layers = [m for m in feature_layers if m is not classifier_module]

        total_layers = len(feature_layers)
        if total_layers == 0:
            raise ValueError(
                "TorchBaseBackBone.unfreeze_feature_extractor: could not infer feature extractor layers"
            )
        if num_layers_from_classification > total_layers:
            raise ValueError(
                "TorchBaseBackBone.unfreeze_feature_extractor: "
                f"num_layers_from_classification must be <= total feature layers ({total_layers}), "
                f"got {num_layers_from_classification}"
            )

        for layer in feature_layers[-num_layers_from_classification:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: (B, P, C, H, W) -> (B, P, out_feature)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() != 5:
            raise ValueError(
                "TorchBaseBackBone.forward: expected input with 4 or 5 dimensions, "
                f"got {x.dim()}"
            )

        batch_size, num_patches, channels, height, width = x.shape
        patch_image = x.reshape(batch_size * num_patches, channels, height, width)
        patch_features = self.model_(patch_image)
        if patch_features.dim() != 2:
            raise ValueError(
                "TorchBaseBackBone.forward: expected output from model to have 2 dimensions, "
                f"got {patch_features.dim()}"
            )
        patch_embeddings = self.proj(patch_features)
        return patch_embeddings.reshape(batch_size, num_patches, self.out_feature)
