package usecases

import (
	"context"
	"fmt"
	"sync"

	"orchestrator/internal/modules/export_tensorrt/application/dto"
)

type ExportTensorrtModelRegistryUseCase struct {
	mu      sync.Mutex
	useCase *ExportTensorrtUseCase
	models  map[string]dto.ExportRequest
}

func NewExportTensorrtModelRegistryUseCase(useCase *ExportTensorrtUseCase) (*ExportTensorrtModelRegistryUseCase, error) {
	if useCase == nil {
		return nil, fmt.Errorf("export TensorRT use case must not be nil")
	}
	return &ExportTensorrtModelRegistryUseCase{useCase: useCase, models: make(map[string]dto.ExportRequest)}, nil
}

func (e *ExportTensorrtModelRegistryUseCase) Register(request dto.ExportRequest) error {
	domainRequest, err := request.ToDomain()
	if err != nil {
		return fmt.Errorf("validate export request: %w", err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.models[domainRequest.ModelName()] = request
	return nil
}

func (e *ExportTensorrtModelRegistryUseCase) Remove(modelName string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.models, modelName)
}

func (e *ExportTensorrtModelRegistryUseCase) Export(ctx context.Context, modelName string) (bool, error) {
	e.mu.Lock()
	request, found := e.models[modelName]
	e.mu.Unlock()
	if !found {
		return false, fmt.Errorf("model not found: %s", modelName)
	}
	if err := e.useCase.Execute(ctx, request); err != nil {
		return false, err
	}
	return true, nil
}
