package dto

import (
	"fmt"
	"strings"

	"orchestrator/internal/modules/export_tensorrt/domain"
)

type ExportRequest struct {
	ModelName      string
	ModelVersion   string
	RepositoryRoot string
}

func (r ExportRequest) ToDomain() (domain.ExportRequest, error) {
	request, err := domain.NewExportRequest(r.ModelName, r.ModelVersion)
	if err != nil {
		return domain.ExportRequest{}, err
	}
	if strings.TrimSpace(r.RepositoryRoot) == "" {
		return domain.ExportRequest{}, fmt.Errorf("repository root must not be empty")
	}
	return request, nil
}
