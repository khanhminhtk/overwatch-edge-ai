package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"orchestrator/internal/modules/export_tensorrt/adapters/outbound/cli"
	"orchestrator/internal/modules/export_tensorrt/application/dto"
	"orchestrator/internal/modules/export_tensorrt/application/use_cases"
	"orchestrator/internal/modules/export_tensorrt/domain"
	platformconfig "orchestrator/internal/platform/config"
)

func main() {
	configDir := flag.String("config-dir", "", "path to orchestrator config directory")
	modelName := flag.String("model", "recognizer", "model name to export")
	modelVersion := flag.String("version", "1", "model version to export")
	flag.Parse()

	resolvedConfigDir, err := resolveConfigDir(*configDir)
	if err != nil {
		fail(err)
	}
	settings, err := platformconfig.Load[domain.Config](platformconfig.Options{
		YAMLFiles:    []string{filepath.Join(resolvedConfigDir, "orchestrator.yaml")},
		EnvFiles:     []string{filepath.Join(resolvedConfigDir, ".env")},
		IncludeOSEnv: true,
		Section:      "export_tensorrt",
	})
	if err != nil {
		fail(fmt.Errorf("load export TensorRT config: %w", err))
	}
	if err := settings.Validate(); err != nil {
		fail(fmt.Errorf("validate export TensorRT config: %w", err))
	}

	repoRoot, err := filepath.Abs(filepath.Join(resolvedConfigDir, "..", "..", "..", ".."))
	if err != nil {
		fail(fmt.Errorf("resolve repository root: %w", err))
	}
	useCase, err := usecases.NewExportTensorrtUseCase(cli.NewTensorRTBuilder(), settings, nil)
	if err != nil {
		fail(fmt.Errorf("create export TensorRT use case: %w", err))
	}

	if err := useCase.Execute(context.Background(), dto.ExportRequest{
		ModelName: *modelName, ModelVersion: *modelVersion, RepositoryRoot: repoRoot,
	}); err != nil {
		fail(err)
	}
}

func resolveConfigDir(configured string) (string, error) {
	if configured != "" {
		return filepath.Abs(configured)
	}
	directory, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory: %w", err)
	}
	for {
		candidate := filepath.Join(directory, "edge", "jetson", "orchestrators", "config")
		if info, err := os.Stat(filepath.Join(candidate, "orchestrator.yaml")); err == nil && !info.IsDir() {
			return candidate, nil
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
		directory = parent
	}
	return "", fmt.Errorf("cannot locate edge/jetson/orchestrators/config; provide -config-dir")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "export-tensorrt:", err)
	os.Exit(1)
}
