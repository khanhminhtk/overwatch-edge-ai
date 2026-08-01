package main

import (
	"fmt"
	"path/filepath"
	"time"

	deployrunner "orchestrator/internal/modules/deploy_model/application/use_cases/runner"
	exportcli "orchestrator/internal/modules/export_tensorrt/adapters/outbound/cli"
	exportusecase "orchestrator/internal/modules/export_tensorrt/application/use_cases"
	inferencecli "orchestrator/internal/modules/inference_runtime/adapters/outbound/cli"
	jobworker "orchestrator/internal/modules/job_control/adapters/inbound/worker"
	"orchestrator/internal/modules/job_control/adapters/outbound/persistence/postgres"
	"orchestrator/internal/modules/job_control/adapters/outbound/scheduler"
	jobports "orchestrator/internal/modules/job_control/application/ports"
	"orchestrator/internal/modules/job_control/application/usecase"
	platformlog "orchestrator/internal/platform/log"
	postgresplatform "orchestrator/internal/platform/persistence/postgres"
)

const (
	deployEventType    = "deploy_requested"
	exportEventType    = "export_tensorrt_requested"
	inferenceEventType = "inference_runtime_requested"
)

func newPollingRunners(configDir string, settings appConfig, transaction postgresplatform.Transaction, logger *platformlog.Logger) ([]*scheduler.PollingRunner, error) {
	runtimeDir, err := filepath.Abs(filepath.Join(filepath.Dir(configDir), settings.InferenceRuntime.Directory))
	if err != nil {
		return nil, fmt.Errorf("resolve inference runtime directory: %w", err)
	}

	runtime, err := inferencecli.NewRunner(inferencecli.Config{
		RuntimeDir:   runtimeDir,
		UVPath:       settings.InferenceRuntime.UVPath,
		DaemonModule: settings.InferenceRuntime.Module,
	})
	if err != nil {
		return nil, fmt.Errorf("create inference runtime worker: %w", err)
	}
	runtime.Logger = logger.With("job", "inference_runtime")

	deploy, err := deployrunner.NewRunner(logger.With("job", "deploy_model"), settings.JobControl.ServerID)
	if err != nil {
		return nil, fmt.Errorf("create deploy worker: %w", err)
	}
	export, err := exportusecase.NewExportTensorrtUseCase(exportcli.NewTensorRTBuilder(), settings.ExportTensorRT, logger.With("job", "export_tensorrt"))
	if err != nil {
		return nil, fmt.Errorf("create TensorRT export worker: %w", err)
	}
	repositoryRoot, err := filepath.Abs(filepath.Join(configDir, "..", "..", "..", ".."))
	if err != nil {
		return nil, fmt.Errorf("resolve repository root: %w", err)
	}

	pollInterval := time.Duration(settings.JobControl.PollIntervalMS) * time.Millisecond
	definitions := []struct {
		eventType string
		handler   jobports.JobHandler
	}{
		{eventType: deployEventType, handler: jobworker.NewDeployHandler(deploy, settings.DeployModel.Triton)},
		{eventType: exportEventType, handler: jobworker.NewExportTensorRTHandler(export, repositoryRoot)},
		{eventType: inferenceEventType, handler: jobworker.NewInferenceRuntimeHandler(runtime)},
	}
	workers := make([]*scheduler.PollingRunner, 0, len(definitions))
	for _, definition := range definitions {
		worker, err := newPollingRunner(transaction, definition.eventType, definition.handler, settings.JobControl.ServerID, pollInterval, logger)
		if err != nil {
			return nil, err
		}
		workers = append(workers, worker)
	}
	return workers, nil
}

func newPollingRunner(transaction postgresplatform.Transaction, eventType string, handler jobports.JobHandler, serverID string, pollInterval time.Duration, logger *platformlog.Logger) (*scheduler.PollingRunner, error) {
	repository, err := postgres.NewRepository(transaction, eventType, 1800)
	if err != nil {
		return nil, fmt.Errorf("create SQL worker repository for %s: %w", eventType, err)
	}
	process, err := usecase.NewProcessNextJob(repository, handler, repository, nil, logger.With("event_type", eventType), eventType)
	if err != nil {
		return nil, fmt.Errorf("create SQL worker use case for %s: %w", eventType, err)
	}
	return scheduler.NewPollingRunner(process, serverID, pollInterval, logger.With("event_type", eventType)), nil
}
