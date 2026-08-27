package httpui

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"web_ui/internal/domain"
)

type ingestRequest struct {
	FrameID      string                   `json:"frame_id"`
	Timestamp    string                   `json:"timestamp"`
	ImagePayload string                   `json:"image_payload"`
	Detections   []ingestDetectionRequest `json:"detections"`
}

type ingestDetectionRequest struct {
	Label string                   `json:"label"`
	Score float64                  `json:"score"`
	BBox  ingestBoundingBoxRequest `json:"bbox"`
}

type ingestBoundingBoxRequest struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

type ingestHandler struct {
	publisher inferenceFramePublisher
}

func newIngestHandler(publisher inferenceFramePublisher) http.Handler {
	return &ingestHandler{publisher: publisher}
}

func (h *ingestHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.publisher == nil {
		http.Error(w, http.StatusText(http.StatusServiceUnavailable), http.StatusServiceUnavailable)
		return
	}

	payload, err := decodeIngestRequest(r.Body)
	if err != nil {
		log.Printf("httpui ingest decode failed: %v", err)
		http.Error(w, http.StatusText(http.StatusBadRequest), http.StatusBadRequest)
		return
	}

	frame, err := payload.toDomain()
	if err != nil {
		log.Printf("httpui ingest validation failed: %v", err)
		http.Error(w, http.StatusText(http.StatusBadRequest), http.StatusBadRequest)
		return
	}

	if err := h.publisher.Execute(r.Context(), frame); err != nil {
		log.Printf("httpui ingest publish failed: %v", err)
		http.Error(w, http.StatusText(http.StatusServiceUnavailable), http.StatusServiceUnavailable)
		return
	}

	w.WriteHeader(http.StatusAccepted)
}

func decodeIngestRequest(r io.Reader) (ingestRequest, error) {
	var payload ingestRequest

	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&payload); err != nil {
		return ingestRequest{}, err
	}

	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return ingestRequest{}, io.ErrUnexpectedEOF
		}
		return ingestRequest{}, err
	}

	return payload, nil
}

func (r ingestRequest) toDomain() (domain.InferenceFrame, error) {
	timestamp, err := time.Parse(time.RFC3339Nano, r.Timestamp)
	if err != nil {
		return domain.InferenceFrame{}, err
	}

	detections := make([]domain.Detection, 0, len(r.Detections))
	for _, detection := range r.Detections {
		box, err := domain.NewBoundingBox(detection.BBox.X, detection.BBox.Y, detection.BBox.Width, detection.BBox.Height)
		if err != nil {
			return domain.InferenceFrame{}, err
		}

		builtDetection, err := domain.NewDetection(detection.Label, detection.Score, box)
		if err != nil {
			return domain.InferenceFrame{}, err
		}

		detections = append(detections, builtDetection)
	}

	return domain.NewInferenceFrame(r.FrameID, timestamp, r.ImagePayload, detections)
}

func newIngestRequestFromFrame(frame domain.InferenceFrame) ingestRequest {
	detections := make([]ingestDetectionRequest, 0, len(frame.Detections))
	for _, detection := range frame.Detections {
		detections = append(detections, ingestDetectionRequest{
			Label: detection.Label,
			Score: detection.Score,
			BBox: ingestBoundingBoxRequest{
				X:      detection.Box.X,
				Y:      detection.Box.Y,
				Width:  detection.Box.Width,
				Height: detection.Box.Height,
			},
		})
	}

	return ingestRequest{
		FrameID:      frame.FrameID,
		Timestamp:    frame.Timestamp.UTC().Format(time.RFC3339Nano),
		ImagePayload: frame.ImagePayload,
		Detections:   detections,
	}
}
