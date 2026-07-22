package runner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"orchestrator/internal/modules/deploy_model/application/dto"
	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
	platformconfig "orchestrator/internal/platform/config"
)

// TestRunFullPipelineIntegration uses real artifacts, a real Triton server,
// and the configured model repository. It changes model files and Triton
// configuration, so it is intentionally opt-in.
func TestRunFullPipelineIntegration(t *testing.T) {
	if os.Getenv("RUN_DEPLOY_MODEL_INTEGRATION") != "1" {
		t.Skip("set RUN_DEPLOY_MODEL_INTEGRATION=1 to run against Triton")
	}
	version := os.Getenv("DEPLOY_MODEL_TEST_VERSION")
	if version == "" {
		t.Fatal("DEPLOY_MODEL_TEST_VERSION must identify the real version to deploy")
	}

	configDir, err := locateConfigDir()
	if err != nil {
		t.Fatal(err)
	}
	config, err := platformconfig.Load[domain.TritonConfig](platformconfig.Options{
		YAMLFiles:    []string{filepath.Join(configDir, "orchestrator.yaml")},
		EnvFiles:     []string{filepath.Join(configDir, ".env")},
		IncludeOSEnv: true,
		Section:      "deploy_model.triton",
	})
	if err != nil {
		t.Fatalf("load triton config: %v", err)
	}
	if config.IP == "0.0.0.0" {
		t.Fatal("TRITON_IP must be a reachable address, such as 127.0.0.1")
	}
	workspaceRoot := filepath.Clean(filepath.Join(configDir, "..", "..", "..", ".."))
	previousDirectory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(workspaceRoot); err != nil {
		t.Fatalf("switch to workspace root: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(previousDirectory) })
	if info, err := os.Stat(config.ModelRepositoryPath); err != nil || !info.IsDir() {
		t.Fatalf("TRITON_MODEL_REPOSITORY_PATH must be an existing host directory, got %q: %v", config.ModelRepositoryPath, err)
	}

	runner, err := NewRunner(testLogger{}, "integration-test")
	if err != nil {
		t.Fatal(err)
	}
	parsedVersion, err := strconv.Atoi(version)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Run(context.Background(), dto.DeployRequest{Triton: config, Format: domain.ModelFormatONNX, Version: parsedVersion}); err != nil {
		t.Fatalf("run deploy pipeline: %v", err)
	}
}

func locateConfigDir() (string, error) {
	directory, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		for _, candidate := range []string{
			filepath.Join(directory, "config"),
			filepath.Join(directory, "edge", "jetson", "orchestrators", "config"),
		} {
			if info, err := os.Stat(filepath.Join(candidate, "orchestrator.yaml")); err == nil && !info.IsDir() {
				return candidate, nil
			}
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return "", fmt.Errorf("cannot locate orchestrator config directory")
		}
		directory = parent
	}
}
