package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	deploydomain "orchestrator/internal/modules/deploy_model/domain/value_objects"
	exportdomain "orchestrator/internal/modules/export_tensorrt/domain"
	"orchestrator/internal/platform/config"
	kafkaPkg "orchestrator/internal/platform/messaging/kafka"
	postgresPkg "orchestrator/internal/platform/persistence/postgres"
)

const (
	envFile  = ".env"
	yamlFile = "orchestrator.yaml"
)

// appConfig contains only settings used by this command's composition root.
// Module-specific settings keep the types owned by their respective modules.
type appConfig struct {
	Kafka            kafkaPkg.Config
	Postgres         postgresPkg.Config
	Log              loggingConfig
	JobControl       workerConfig
	DeployModel      deployConfig
	ExportTensorRT   exportdomain.Config
	InferenceRuntime inferenceRuntimeConfig
	MLflowDownload   mlflowDownloadConfig
}

type loggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

type workerConfig struct {
	ServerID       string `yaml:"server_id"`
	PollIntervalMS int    `yaml:"poll_interval_ms"`
}

type deployConfig struct {
	Triton deploydomain.TritonConfig `yaml:"triton"`
}

type inferenceRuntimeConfig struct {
	Directory string `yaml:"directory"`
	UVPath    string `yaml:"uv_path"`
	Module    string `yaml:"module"`
}
type mlflowDownloadConfig struct {
	TrackingURI         string `yaml:"tracking_uri"`
	StagingDirectory    string `yaml:"staging_directory"`
	DefaultArtifactPath string `yaml:"default_artifact_path"`
}

func resolveConfigDir() (string, error) {
	directory, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory: %w", err)
	}
	for {
		for _, candidate := range []string{filepath.Join(directory, "config"), filepath.Join(directory, "edge", "jetson", "orchestrators", "config")} {
			if info, err := os.Stat(filepath.Join(candidate, yamlFile)); err == nil && !info.IsDir() {
				return candidate, nil
			}
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
		directory = parent
	}
	return "", fmt.Errorf("cannot locate edge/jetson/orchestrators/config/%s", yamlFile)
}

func loadAppConfig(configDir string) (appConfig, error) {
	options := func(section string) config.Options {
		return config.Options{YAMLFiles: []string{filepath.Join(configDir, yamlFile)}, EnvFiles: []string{filepath.Join(configDir, envFile)}, IncludeOSEnv: true, Section: section}
	}
	logConfig, err := config.Load[loggingConfig](options("log"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load logging config: %w", err)
	}
	kafkaConfig, err := config.Load[kafkaPkg.Config](options("kafka"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load Kafka config: %w", err)
	}
	postgresConfig, err := config.Load[postgresPkg.Config](options("postgres"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load PostgreSQL config: %w", err)
	}
	jobControlConfig, err := config.Load[workerConfig](options("job_control"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load job-control worker config: %w", err)
	}
	deployModelConfig, err := config.Load[deployConfig](options("deploy_model"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load deploy worker config: %w", err)
	}
	exportTensorRTConfig, err := config.Load[exportdomain.Config](options("export_tensorrt"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load TensorRT export worker config: %w", err)
	}
	inferenceRuntimeConfig, err := config.Load[inferenceRuntimeConfig](options("inference_runtime"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load inference runtime worker config: %w", err)
	}
	mlflowDownloadConfig, err := config.Load[mlflowDownloadConfig](options("mlflow_download"))
	if err != nil {
		return appConfig{}, fmt.Errorf("load MLflow download worker config: %w", err)
	}
	settings := appConfig{Kafka: kafkaConfig, Postgres: postgresConfig, Log: logConfig, JobControl: jobControlConfig, DeployModel: deployModelConfig, ExportTensorRT: exportTensorRTConfig, InferenceRuntime: inferenceRuntimeConfig, MLflowDownload: mlflowDownloadConfig}
	if err := settings.validate(); err != nil {
		return appConfig{}, err
	}
	return settings, nil
}

func (c appConfig) validate() error {
	if err := c.Kafka.Validate(); err != nil {
		return fmt.Errorf("validate Kafka config: %w", err)
	}
	if err := c.Postgres.Validate(); err != nil {
		return fmt.Errorf("validate PostgreSQL config: %w", err)
	}
	if strings.TrimSpace(c.JobControl.ServerID) == "" {
		return fmt.Errorf("job-control worker server_id must not be empty")
	}
	if c.JobControl.PollIntervalMS <= 0 {
		return fmt.Errorf("job-control worker poll_interval_ms must be positive")
	}
	if err := c.DeployModel.Triton.Validate(); err != nil {
		return fmt.Errorf("validate deploy worker config: %w", err)
	}
	if err := c.ExportTensorRT.Validate(); err != nil {
		return fmt.Errorf("validate TensorRT export worker config: %w", err)
	}
	if strings.TrimSpace(c.InferenceRuntime.Directory) == "" {
		return fmt.Errorf("inference runtime directory must not be empty")
	}
	if strings.TrimSpace(c.InferenceRuntime.UVPath) == "" {
		return fmt.Errorf("inference runtime uv_path must not be empty")
	}
	if strings.TrimSpace(c.InferenceRuntime.Module) == "" {
		return fmt.Errorf("inference runtime module must not be empty")
	}
	return nil
}
