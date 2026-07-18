package main

import (
	"path/filepath"
	"testing"

	platformconfig "orchestrator/internal/platform/config"
)

func TestOrchestratorConfigLoadsEnvAndYAML(t *testing.T) {
	configDir := filepath.Join("..", "..", "config")
	settings, err := platformconfig.Load[appConfig](platformconfig.Options{YAMLFiles: []string{filepath.Join(configDir, "orchestrator.yaml")}, EnvFiles: []string{filepath.Join(configDir, ".env")}, IncludeOSEnv: false})
	if err != nil {
		t.Fatal(err)
	}
	if settings.InferenceRuntime.UVPath != "uv" || settings.InferenceRuntime.Module != "src.entrypoints.inference_daemon.main" {
		t.Fatalf("unexpected inference settings: %#v", settings.InferenceRuntime)
	}
}
