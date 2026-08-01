package ports

import (
	"context"

	"orchestrator/internal/modules/export_tensorrt/domain"
)

type TensorRTBuilder interface {
	Build(context.Context, domain.ExportRequest, domain.Config, string) error
}
