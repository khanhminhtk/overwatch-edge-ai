package ports

import (
	"context"

	"orchestrator/internal/modules/inference_runtime/domain"
)

// RuntimeRunner executes an inference invocation through an external runtime.
type RuntimeRunner interface {
	Run(context.Context, domain.Invocation) error
}
