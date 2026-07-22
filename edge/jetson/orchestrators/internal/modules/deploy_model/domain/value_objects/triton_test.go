package valueobjects

import (
	"path/filepath"
	"testing"

	platformconfig "orchestrator/internal/platform/config"
)

func TestTritonConfigLoadsAndValidatesFromOrchestratorConfig(t *testing.T) {
	configDir := filepath.Join("..", "..", "..", "..", "..", "config")
	settings, err := platformconfig.Load[TritonConfig](platformconfig.Options{
		YAMLFiles: []string{filepath.Join(configDir, "orchestrator.yaml")},
		EnvFiles:  []string{filepath.Join(configDir, ".env")},
		Section:   "deploy_model.triton",
	})
	if err != nil {
		t.Fatalf("load deploy_model.triton config: %v", err)
	}

	if err := settings.Validate(); err != nil {
		t.Fatalf("validate triton config: %v", err)
	}
}

func TestTritonConfigMissingImageFailsValidation(t *testing.T) {
	cfg := TritonConfig{
		ModelRepositoryPath: "/models",
		IP:                  "0.0.0.0",
		HostPort:            8000,
		Model: TritonModelConfig{
			Detector:   TritonModelSpec{ModelPathRaw: "/models/detector", ModelPathONNX: "/models/detector.onnx", ModelPathTRT: "/models/detector.trt", Name: "detector", Version: 1},
			Recognizer: TritonModelSpec{ModelPathRaw: "/models/recognizer", ModelPathONNX: "/models/recognizer.onnx", ModelPathTRT: "/models/recognizer.trt", Name: "recognizer", Version: 1},
		},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected validation error for empty image")
	}
}

func TestTritonConfigInvalidHostPortFailsValidation(t *testing.T) {
	cfg := TritonConfig{
		Image:               "nvcr.io/nvidia/tritonserver:23.06-py3",
		ModelRepositoryPath: "/models",
		IP:                  "0.0.0.0",
		HostPort:            0,
		Model: TritonModelConfig{
			Detector:   TritonModelSpec{ModelPathRaw: "/models/detector", ModelPathONNX: "/models/detector.onnx", ModelPathTRT: "/models/detector.trt", Name: "detector", Version: 1},
			Recognizer: TritonModelSpec{ModelPathRaw: "/models/recognizer", ModelPathONNX: "/models/recognizer.onnx", ModelPathTRT: "/models/recognizer.trt", Name: "recognizer", Version: 1},
		},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected validation error for invalid host port")
	}
}
