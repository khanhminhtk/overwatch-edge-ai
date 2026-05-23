package infra

import (
	"path/filepath"
	"runtime"
	"testing"

	"backend_nexus/internal/infra/config"
	"backend_nexus/internal/utils"
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

func TestNewMinioClient(t *testing.T) {
	configPath, envPath := testConfigPaths(t)
	cfg, err := utils.NewConfigLoader[config.Minio](
		configPath, envPath).LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadConfig() returned nil config")
	}

	client, err := NewMinioClient(cfg.MinioServer)
	if err != nil {
		t.Fatalf("NewMinioClient() error = %v", err)
	}
	if client == nil {
		t.Fatal("NewMinioClient() returned nil client")
	}
}
