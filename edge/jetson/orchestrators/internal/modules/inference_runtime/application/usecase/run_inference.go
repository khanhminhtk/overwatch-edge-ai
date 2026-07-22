package usecase

import (
	"context"
	"fmt"

	"orchestrator/internal/modules/inference_runtime/application/dto"
	"orchestrator/internal/modules/inference_runtime/application/ports"
	"orchestrator/internal/modules/inference_runtime/domain"
)

type RunInference struct{ runner ports.RuntimeRunner }

func NewRunInference(runner ports.RuntimeRunner) (RunInference, error) {
	if runner == nil {
		return RunInference{}, fmt.Errorf("inference runtime runner must not be nil")
	}
	return RunInference{runner: runner}, nil
}
func (u RunInference) Execute(ctx context.Context, request dto.RunRequest) error {
	if ctx == nil {
		return fmt.Errorf("run context must not be nil")
	}
	invocation, err := domain.NewInvocation(request.Display, request.Arguments)
	if err != nil {
		return err
	}
	return u.runner.Run(ctx, invocation)
}
