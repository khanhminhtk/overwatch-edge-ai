package valueobjects

import (
	"path/filepath"
	"testing"

	platformconfig "orchestrator/internal/platform/config"
)

func TestSystemConfigLoadsAndValidatesFromOrchestratorConfig(t *testing.T) {
	configDir := filepath.Join("..", "..", "..", "..", "..", "config")
	settings, err := platformconfig.Load[SystemConfig](platformconfig.Options{
		YAMLFiles: []string{filepath.Join(configDir, "orchestrator.yaml")},
		EnvFiles:  []string{filepath.Join(configDir, ".env")},
		Section:   "deploy_model",
	})
	if err != nil {
		t.Fatalf("load deploy_model config: %v", err)
	}

	if err := settings.Validate(); err != nil {
		t.Fatalf("validate configured docker path: %v", err)
	}
}
