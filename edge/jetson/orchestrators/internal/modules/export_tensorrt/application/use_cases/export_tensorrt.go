package usecases

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"orchestrator/internal/modules/export_tensorrt/application/dto"
	"orchestrator/internal/modules/export_tensorrt/application/ports"
	"orchestrator/internal/modules/export_tensorrt/domain"
)

var ErrExportAlreadyRunning = errors.New("TensorRT export is already running")

type Logger interface {
	Info(string, ...any)
}

type ExportTensorrtUseCase struct {
	builder ports.TensorRTBuilder
	config  domain.Config
	logger  Logger
	running atomic.Bool
}

func NewExportTensorrtUseCase(builder ports.TensorRTBuilder, config domain.Config, logger Logger) (*ExportTensorrtUseCase, error) {
	if builder == nil {
		return nil, fmt.Errorf("TensorRT builder must not be nil")
	}
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("validate export TensorRT config: %w", err)
	}
	return &ExportTensorrtUseCase{builder: builder, config: config, logger: logger}, nil
}

func (e *ExportTensorrtUseCase) Execute(ctx context.Context, input dto.ExportRequest) error {
	if ctx == nil {
		return fmt.Errorf("export context must not be nil")
	}
	request, err := input.ToDomain()
	if err != nil {
		return fmt.Errorf("validate export request: %w", err)
	}
	if !e.running.CompareAndSwap(false, true) {
		return ErrExportAlreadyRunning
	}
	defer e.running.Store(false)

	started := time.Now()
	if e.logger != nil {
		e.logger.Info("[export_tensorrt.application.usecases.ExportTensorrtUseCase.Execute] - TensorRT export started.", "model", request.ModelName(), "version", request.ModelVersion())
	}
	if err := e.builder.Build(ctx, request, e.config, input.RepositoryRoot); err != nil {
		return err
	}
	if e.logger != nil {
		e.logger.Info("[export_tensorrt.application.usecases.ExportTensorrtUseCase.Execute] - TensorRT export completed.", "model", request.ModelName(), "version", request.ModelVersion(), "duration_ms", time.Since(started).Milliseconds())
	}
	return nil
}
