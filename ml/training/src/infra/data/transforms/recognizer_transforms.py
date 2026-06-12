from typing import Tuple

from torchvision.transforms import v2


def build_recognizer_train_transform(
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    hue: float = 0.02,
    p_grayscale: float = 0.1,
    blur_kernel_size: Tuple[int, int] = (3, 3),
    blur_sigma: Tuple[float, float] = (0.1, 1.2),
    affine_degrees: float = 2.0,
    affine_translate: Tuple[float, float] = (0.02, 0.02),
    affine_scale: Tuple[float, float] = (0.97, 1.03),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    return v2.Compose([
        v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        v2.RandomGrayscale(p=p_grayscale),
        v2.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma),
        v2.RandomAffine(
            degrees=affine_degrees,
            translate=affine_translate,
            scale=affine_scale,
        ),
        v2.Normalize(
            mean=list(mean),
            std=list(std),
        ),
    ])


def build_recognizer_val_transform(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    return v2.Compose([
        v2.Normalize(
            mean=list(mean),
            std=list(std),
        ),
    ])
