package utils

import (
	"path/filepath"
	"runtime"
	"testing"

	"backend_nexus/internal/infra/config"
)

func testConfigPaths(t *testing.T) (string, string) {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}

	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	return filepath.Join(repoRoot, "config", "minio_config.yaml"), filepath.Join(repoRoot, "config", ".env")
}

func TestLoadConfig(t *testing.T) {
	configPath, envPath := testConfigPaths(t)

	cfg, err := NewConfigLoader[config.Minio](
		configPath, envPath).LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadConfig() returned nil config")
	}
}

func TestLoadConfig_FileNotFound(t *testing.T) {
	_, err := NewConfigLoader[config.MinioClient]("nonexistent.yaml", "").LoadConfig()
	if err == nil {
		t.Fatal("Expected error for nonexistent config file, got nil")
	}
}

func TestLoadConfig_InvalidYAML(t *testing.T) {
	_, err := NewConfigLoader[config.MinioClient]("invalid_config.yaml", "").LoadConfig()
	if err == nil {
		t.Fatal("Expected error for invalid YAML, got nil")
	}
}

func TestValueConfig(t *testing.T) {
	configPath, envPath := testConfigPaths(t)
	cfg, err := NewConfigLoader[config.Minio](
		configPath, envPath).LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadConfig() returned nil config")
	}
	expectedEndpoint := "http://127.0.0.1:9000"
	if cfg.MinioServer.Endpoint != expectedEndpoint {
		t.Errorf("Expected Endpoint %s, got %s", expectedEndpoint, cfg.MinioServer.Endpoint)
	}
}
