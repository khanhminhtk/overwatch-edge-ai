package mockstream

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"web_ui/internal/domain"
)

type Generator struct {
	now   func() time.Time
	seqNo uint64
}

const mockFrameIDPrefix = "mock-frame-"

func NewGenerator(now func() time.Time) *Generator {
	if now == nil {
		now = time.Now
	}

	return &Generator{now: now}
}

func (g *Generator) Next() (domain.InferenceFrame, error) {
	g.seqNo++

	timestamp := g.now().UTC()
	box, err := domain.NewBoundingBox(510, 172, 220, 474)
	if err != nil {
		return domain.InferenceFrame{}, err
	}

	detection, err := domain.NewDetection("person", 0.98, box)
	if err != nil {
		return domain.InferenceFrame{}, err
	}

	imagePayload := buildImagePayload()

	return domain.NewInferenceFrame(
		fmt.Sprintf("%s%06d", mockFrameIDPrefix, g.seqNo),
		timestamp,
		imagePayload,
		[]domain.Detection{detection},
	)
}

func IsMockFrame(frame domain.InferenceFrame) bool {
	return strings.HasPrefix(frame.FrameID, mockFrameIDPrefix)
}

func buildImagePayload() string {
	svg := `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0b1726"/>
<stop offset="100%" stop-color="#17304b"/>
</linearGradient>
</defs>
<rect width="1280" height="720" fill="url(#bg)"/>
<rect x="140" y="72" width="260" height="576" rx="18" fill="#101a24" opacity="0.55"/>
<rect x="452" y="128" width="356" height="464" rx="28" fill="#f2f6f9" opacity="0.14"/>
<rect x="862" y="104" width="220" height="520" rx="20" fill="#1f3a56" opacity="0.7"/>
<rect x="560" y="172" width="120" height="340" rx="58" fill="#d7e6f3" opacity="0.82"/>
<rect x="510" y="222" width="220" height="284" rx="76" fill="#2c4257" opacity="0.56"/>
<rect x="575" y="520" width="54" height="126" rx="22" fill="#16283c" opacity="0.84"/>
<rect x="652" y="520" width="54" height="126" rx="22" fill="#16283c" opacity="0.84"/>
<circle cx="958" cy="204" r="38" fill="#ffb347" opacity="0.85"/>
<text x="66" y="666" fill="#c5d5e6" font-family="Arial, sans-serif" font-size="28">Local mock inference stream</text>
</svg>`

	return "data:image/svg+xml;utf8," + url.PathEscape(svg)
}
