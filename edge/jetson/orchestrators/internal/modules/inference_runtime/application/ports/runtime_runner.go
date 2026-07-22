package ports

import (
	"context"

	"orchestrator/internal/modules/inference_runtime/domain"
)

type RuntimeRunner interface {
	Run(context.Context, domain.Invocation) error
}
