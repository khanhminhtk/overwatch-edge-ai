import pytest
import torch
from torchvision.transforms import v2

from src.infra.data.transforms.recognizer_transforms import (
    build_recognizer_train_transform,
    build_recognizer_val_transform,
)


def test_build_recognizer_train_transform_default_pipeline():
    transform = build_recognizer_train_transform()

    assert isinstance(transform, v2.Compose)
    assert len(transform.transforms) == 5
    assert isinstance(transform.transforms[0], v2.ColorJitter)
    assert isinstance(transform.transforms[1], v2.RandomGrayscale)
    assert isinstance(transform.transforms[2], v2.GaussianBlur)
    assert isinstance(transform.transforms[3], v2.RandomAffine)
    assert isinstance(transform.transforms[4], v2.Normalize)


def test_build_recognizer_train_transform_default_params():
    transform = build_recognizer_train_transform()
    jitter, grayscale, blur, affine, norm = transform.transforms

    assert jitter.brightness == pytest.approx((0.8, 1.2))
    assert jitter.contrast == pytest.approx((0.8, 1.2))
    assert jitter.saturation == pytest.approx((0.9, 1.1))
    assert jitter.hue == pytest.approx((-0.02, 0.02))
    assert grayscale.p == pytest.approx(0.1)
    assert blur.kernel_size == (3, 3)
    assert blur.sigma == pytest.approx([0.1, 1.2])
    assert affine.degrees == pytest.approx([-2.0, 2.0])
    assert affine.translate == pytest.approx((0.02, 0.02))
    assert affine.scale == pytest.approx((0.97, 1.03))
    assert norm.mean == pytest.approx([0.485, 0.456, 0.406])
    assert norm.std == pytest.approx([0.229, 0.224, 0.225])


def test_build_recognizer_train_transform_custom_params():
    transform = build_recognizer_train_transform(
        brightness=0.3,
        contrast=0.4,
        saturation=0.2,
        hue=0.01,
        p_grayscale=0.25,
        blur_kernel_size=(5, 5),
        blur_sigma=(0.3, 1.7),
        affine_degrees=4.0,
        affine_translate=(0.05, 0.03),
        affine_scale=(0.9, 1.1),
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )
    jitter, grayscale, blur, affine, norm = transform.transforms

    assert jitter.brightness == pytest.approx((0.7, 1.3))
    assert jitter.contrast == pytest.approx((0.6, 1.4))
    assert jitter.saturation == pytest.approx((0.8, 1.2))
    assert jitter.hue == pytest.approx((-0.01, 0.01))
    assert grayscale.p == pytest.approx(0.25)
    assert blur.kernel_size == (5, 5)
    assert blur.sigma == pytest.approx([0.3, 1.7])
    assert affine.degrees == pytest.approx([-4.0, 4.0])
    assert affine.translate == pytest.approx((0.05, 0.03))
    assert affine.scale == pytest.approx((0.9, 1.1))
    assert norm.mean == pytest.approx([0.1, 0.2, 0.3])
    assert norm.std == pytest.approx([0.4, 0.5, 0.6])


def test_train_transform_preserves_input_shape():
    transform = build_recognizer_train_transform()
    image = torch.rand(8, 3, 64, 64, dtype=torch.float32)

    transformed = transform(image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == image.shape


def test_build_recognizer_val_transform_default_and_custom_params():
    default_transform = build_recognizer_val_transform()
    custom_transform = build_recognizer_val_transform(
        mean=(0.3, 0.4, 0.5),
        std=(0.1, 0.2, 0.3),
    )

    assert isinstance(default_transform, v2.Compose)
    assert len(default_transform.transforms) == 1
    assert isinstance(default_transform.transforms[0], v2.Normalize)
    assert default_transform.transforms[0].mean == pytest.approx([0.485, 0.456, 0.406])
    assert default_transform.transforms[0].std == pytest.approx([0.229, 0.224, 0.225])

    assert isinstance(custom_transform.transforms[0], v2.Normalize)
    assert custom_transform.transforms[0].mean == pytest.approx([0.3, 0.4, 0.5])
    assert custom_transform.transforms[0].std == pytest.approx([0.1, 0.2, 0.3])
