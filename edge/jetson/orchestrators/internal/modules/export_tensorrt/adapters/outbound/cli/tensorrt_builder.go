package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"orchestrator/internal/modules/export_tensorrt/domain"
)

// TensorRTBuilder invokes the configured Docker Compose builder without a shell.
type TensorRTBuilder struct {
	Stdout io.Writer
	Stderr io.Writer
}

func NewTensorRTBuilder() *TensorRTBuilder {
	return &TensorRTBuilder{Stdout: os.Stdout, Stderr: os.Stderr}
}

func (b *TensorRTBuilder) Build(ctx context.Context, request domain.ExportRequest, config domain.Config, repositoryRoot string) error {
	if ctx == nil {
		return fmt.Errorf("build context must not be nil")
	}
	if err := config.Validate(); err != nil {
		return fmt.Errorf("validate TensorRT builder config: %w", err)
	}
	if repositoryRoot == "" {
		return fmt.Errorf("repository root must not be empty")
	}
	command, err := b.Command(ctx, request, config, repositoryRoot)
	if err != nil {
		return err
	}
	if err := command.Run(); err != nil {
		return fmt.Errorf("run TensorRT builder for model %s version %s: %w", request.ModelName(), request.ModelVersion(), err)
	}
	return nil
}

// Command builds the Docker invocation separately so it can be inspected in tests.
func (b *TensorRTBuilder) Command(ctx context.Context, request domain.ExportRequest, config domain.Config, repositoryRoot string) (*exec.Cmd, error) {
	if ctx == nil {
		return nil, fmt.Errorf("build context must not be nil")
	}
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("validate TensorRT builder config: %w", err)
	}
	resolvedRoot, err := filepath.Abs(repositoryRoot)
	if err != nil {
		return nil, fmt.Errorf("resolve repository root: %w", err)
	}
	command := exec.CommandContext(ctx, "docker", "compose",
		"--env-file", resolvePath(resolvedRoot, config.TritonDockerEnvFile),
		"-f", resolvePath(resolvedRoot, config.TritonDockerComposeFile),
		"--profile", fmt.Sprintf(config.TritonBuilderProfileFormat, request.ModelName()),
		"run", "--rm", "--build",
		fmt.Sprintf(config.TritonBuilderServiceNameFormat, request.ModelName()),
	)
	command.Dir = resolvedRoot
	command.Env = append(os.Environ(), "TRITON_MODEL_VERSION="+request.ModelVersion())
	command.Stdout, command.Stderr = b.Stdout, b.Stderr
	return command, nil
}

func resolvePath(base, path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(base, path)
}
