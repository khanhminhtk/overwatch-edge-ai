from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic.dataclasses import dataclass


Point = Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Bounding box coordinates must be non-negative")

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @classmethod
    def from_polygon(
        cls,
        points: Sequence[Point],
    ) -> BoundingBox:
        """
        Tạo axis-aligned bounding box từ polygon.

        Input:
            [
                {"x": 280, "y": 364},
                {"x": 460, "y": 364},
                {"x": 460, "y": 448},
                {"x": 280, "y": 448},
            ]
        """
        if len(points) < 2:
            raise ValueError("Polygon must contain at least 2 points")

        try:
            xs = [float(point["x"]) for point in points]
            ys = [float(point["y"]) for point in points]
        except KeyError as error:
            raise ValueError(
                f"Polygon point is missing coordinate: {error.args[0]}"
            ) from error

        return cls(
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
        )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    def to_yolo(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Trả về YOLO normalized format:

            x_center, y_center, width, height
        """
        if image_width <= 0:
            raise ValueError("image_width must be greater than 0")

        if image_height <= 0:
            raise ValueError("image_height must be greater than 0")

        if self.x_max > image_width:
            raise ValueError(
                f"x_max={self.x_max} exceeds image_width={image_width}"
            )

        if self.y_max > image_height:
            raise ValueError(
                f"y_max={self.y_max} exceeds image_height={image_height}"
            )

        return (
            self.center_x / image_width,
            self.center_y / image_height,
            self.width / image_width,
            self.height / image_height,
        )


@dataclass(frozen=True, slots=True)
class TextRegionPrediction:
    bounding_box: BoundingBox
    text: str
    confidence: float | None = None