package worker

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	deploydto "orchestrator/internal/modules/deploy_model/application/dto"
	deployports "orchestrator/internal/modules/deploy_model/application/ports"
	deploydomain "orchestrator/internal/modules/deploy_model/domain/value_objects"
	exportdto "orchestrator/internal/modules/export_tensorrt/application/dto"
	exportusecase "orchestrator/internal/modules/export_tensorrt/application/use_cases"
	inferencecli "orchestrator/internal/modules/inference_runtime/adapters/outbound/cli"
	inferencedto "orchestrator/internal/modules/inference_runtime/application/dto"
	jobdto "orchestrator/internal/modules/job_control/application/dto"
	mlflowdownload "orchestrator/internal/modules/mlflow_download/application"
)

// HandlerFunc adapts a function to the job-control JobHandler port.
type HandlerFunc func(context.Context, jobdto.ClaimedJob) jobdto.JobResult

func (f HandlerFunc) Handle(ctx context.Context, job jobdto.ClaimedJob) jobdto.JobResult {
	return f(ctx, job)
}

func NewMLflowDownloadHandler(downloader *mlflowdownload.Downloader) HandlerFunc {
	return func(ctx context.Context, job jobdto.ClaimedJob) jobdto.JobResult {
		if downloader == nil {
			return failed(job, fmt.Errorf("MLflow downloader is not configured"))
		}
		modelName, _ := job.Payload["model_name"].(string)
		if strings.TrimSpace(modelName) == "" {
			switch job.Payload["model_type"] {
			case "detection":
				modelName = "yolo_detector"
			case "recognizer":
				modelName = "vit_ctc_deepseek"
			default:
				return failed(job, fmt.Errorf("job payload requires model_name or supported model_type"))
			}
		}
		version, _ := job.Payload["model_version"].(string)
		result, err := downloader.Download(ctx, modelName, version)
		if err != nil {
			return failed(job, err)
		}
		return succeeded(job, map[string]any{"model_name": modelName, "model_version": result.ModelVersion, "model_path": result.OutputPath})
	}
}

func NewDeployHandler(runner deployports.DeploymentRunner, triton deploydomain.TritonConfig) HandlerFunc {
	return func(ctx context.Context, job jobdto.ClaimedJob) jobdto.JobResult {
		format, err := requiredString(job.Payload, "format")
		if err != nil {
			return failed(job, err)
		}
		version, err := requiredPositiveInt(job.Payload, "version")
		if err != nil {
			return failed(job, err)
		}
		request := deploydto.DeployRequest{Triton: triton, Format: deploydomain.ModelFormat(format), Version: version, Policy: deploydomain.DefaultDeploymentPolicy()}
		if err := runner.Run(ctx, request); err != nil {
			return failed(job, err)
		}
		return succeeded(job, map[string]any{"format": format, "version": version})
	}
}

func NewExportTensorRTHandler(export *exportusecase.ExportTensorrtUseCase, repositoryRoot string) HandlerFunc {
	return func(ctx context.Context, job jobdto.ClaimedJob) jobdto.JobResult {
		modelName, err := requiredString(job.Payload, "model_name")
		if err != nil {
			return failed(job, err)
		}
		modelVersion, err := requiredString(job.Payload, "model_version")
		if err != nil {
			return failed(job, err)
		}
		if err := export.Execute(ctx, exportdto.ExportRequest{ModelName: modelName, ModelVersion: modelVersion, RepositoryRoot: repositoryRoot}); err != nil {
			return failed(job, err)
		}
		return succeeded(job, map[string]any{"model_name": modelName, "model_version": modelVersion})
	}
}

// NewInferenceRuntimeHandler starts the runtime asynchronously. The job is
// marked processed once the OS process has started; runtime failures remain in
// the runtime process logs instead of blocking the SQL polling worker forever.
func NewInferenceRuntimeHandler(runtime *inferencecli.Runner) HandlerFunc {
	return func(ctx context.Context, job jobdto.ClaimedJob) jobdto.JobResult {
		display, err := optionalBool(job.Payload, "display", false)
		if err != nil {
			return failed(job, err)
		}
		arguments, err := optionalStrings(job.Payload, "arguments")
		if err != nil {
			return failed(job, err)
		}
		if err := runtime.Start(ctx, inferencedto.RunRequest{Display: display, Arguments: arguments}); err != nil {
			return failed(job, err)
		}
		return succeeded(job, map[string]any{"display": display, "arguments": arguments})
	}
}

func succeeded(job jobdto.ClaimedJob, result map[string]any) jobdto.JobResult {
	return jobdto.JobResult{RequestID: job.RequestID, Success: true, Result: result}
}

func failed(job jobdto.ClaimedJob, err error) jobdto.JobResult {
	return jobdto.JobResult{RequestID: job.RequestID, Success: false, ErrorMessage: err.Error()}
}

func requiredString(payload map[string]any, key string) (string, error) {
	value, ok := payload[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", fmt.Errorf("job payload requires non-empty %q", key)
	}
	return strings.TrimSpace(value), nil
}

func requiredPositiveInt(payload map[string]any, key string) (int, error) {
	value, ok := payload[key]
	if !ok {
		return 0, fmt.Errorf("job payload requires %q", key)
	}
	switch number := value.(type) {
	case float64:
		if number <= 0 || number != float64(int(number)) {
			return 0, fmt.Errorf("job payload %q must be a positive integer", key)
		}
		return int(number), nil
	case string:
		parsed, err := strconv.Atoi(number)
		if err != nil || parsed <= 0 {
			return 0, fmt.Errorf("job payload %q must be a positive integer", key)
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("job payload %q must be a positive integer", key)
	}
}

func optionalBool(payload map[string]any, key string, fallback bool) (bool, error) {
	value, ok := payload[key]
	if !ok {
		return fallback, nil
	}
	parsed, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("job payload %q must be a boolean", key)
	}
	return parsed, nil
}

func optionalStrings(payload map[string]any, key string) ([]string, error) {
	value, ok := payload[key]
	if !ok || value == nil {
		return nil, nil
	}
	values, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("job payload %q must be an array of strings", key)
	}
	result := make([]string, 0, len(values))
	for _, value := range values {
		argument, ok := value.(string)
		if !ok || strings.TrimSpace(argument) == "" {
			return nil, fmt.Errorf("job payload %q must contain only non-empty strings", key)
		}
		result = append(result, argument)
	}
	return result, nil
}
