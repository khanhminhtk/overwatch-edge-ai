package domain

import (
	"fmt"
	"strings"
)

// ExportRequest identifies the model artifact that must be built as TensorRT.
// It intentionally contains no process or Docker details.
type ExportRequest struct {
	modelName    string
	modelVersion string
}

func NewExportRequest(modelName, modelVersion string) (ExportRequest, error) {
	modelName = strings.TrimSpace(modelName)
	modelVersion = strings.TrimSpace(modelVersion)
	if modelName == "" {
		return ExportRequest{}, fmt.Errorf("model name must not be empty")
	}
	if modelVersion == "" {
		return ExportRequest{}, fmt.Errorf("model version must not be empty")
	}
	return ExportRequest{modelName: modelName, modelVersion: modelVersion}, nil
}

func (r ExportRequest) ModelName() string    { return r.modelName }
func (r ExportRequest) ModelVersion() string { return r.modelVersion }
