from __future__ import annotations

import os

import pytest

from tests.recognizer_test_helpers import build_loader
from src.application.use_cases.recognizer.load_model_vit_ctc import load_model_vit_ctc


@pytest.mark.skipif(
    os.getenv("RUN_ONLINE_MODEL_TESTS", "0") != "1",
    reason="Set RUN_ONLINE_MODEL_TESTS=1 to run model loading test (may download backbone weights).",
)
def test_load_model_vit_ctc_integration(tmp_path) -> None:
    loader, _ = build_loader(tmp_path=tmp_path)
    model = load_model_vit_ctc(config=loader, device="cpu")
    assert model.d_model == 32
    assert len(model.layers) == 1
