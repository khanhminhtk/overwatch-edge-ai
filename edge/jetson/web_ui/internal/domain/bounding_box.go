package domain

import "fmt"

type BoundingBox struct {
	X      int
	Y      int
	Width  int
	Height int
}

func NewBoundingBox(x, y, width, height int) (BoundingBox, error) {
	if x < 0 {
		return BoundingBox{}, fmt.Errorf("bounding box x must be non-negative")
	}
	if y < 0 {
		return BoundingBox{}, fmt.Errorf("bounding box y must be non-negative")
	}
	if width <= 0 {
		return BoundingBox{}, fmt.Errorf("bounding box width must be positive")
	}
	if height <= 0 {
		return BoundingBox{}, fmt.Errorf("bounding box height must be positive")
	}

	return BoundingBox{
		X:      x,
		Y:      y,
		Width:  width,
		Height: height,
	}, nil
}
