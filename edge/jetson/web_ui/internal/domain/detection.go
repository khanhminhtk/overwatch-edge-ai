package domain

import (
	"fmt"
	"strings"
)

type Detection struct {
	Label string
	Score float64
	Box   BoundingBox
}

func NewDetection(label string, score float64, box BoundingBox) (Detection, error) {
	if strings.TrimSpace(label) == "" {
		return Detection{}, fmt.Errorf("detection label must not be blank")
	}
	if score < 0 || score > 1 {
		return Detection{}, fmt.Errorf("detection score must be within [0,1]")
	}
	if !isValidBoundingBox(box) {
		return Detection{}, fmt.Errorf("detection bounding box must be valid")
	}

	return Detection{
		Label: label,
		Score: score,
		Box:   box,
	}, nil
}

func isValidBoundingBox(box BoundingBox) bool {
	return box.X >= 0 && box.Y >= 0 && box.Width > 0 && box.Height > 0
}
