from __future__ import annotations

import src.infra.modeling.hooks as hooks


def test_hooks_public_exports() -> None:
    expected = {
        "shape_of",
        "tensor_norm",
        "assert_finite",
        "FeatureCaptureHook",
        "ShapeLoggerHook",
        "NaNInfGuardHook",
        "GradNormHook",
        "TensorBoardScalarHook",
        "tqdm_wrap",
    }
    assert set(hooks.__all__) == expected
