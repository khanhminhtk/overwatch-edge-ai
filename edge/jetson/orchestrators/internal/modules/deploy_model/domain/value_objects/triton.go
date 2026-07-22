package valueobjects

import "fmt"

type TritonModelSpec struct {
	ModelPathRaw  string `yaml:"model_path_raw"`
	ModelPathONNX string `yaml:"model_path_onnx"`
	ModelPathTRT  string `yaml:"model_path_trt"`
	Name          string `yaml:"name"`
	Version       int    `yaml:"version"`
}

func (s TritonModelSpec) Validate() error {
	if s.ModelPathRaw == "" {
		return fmt.Errorf("triton model path raw must not be empty")
	}
	if s.ModelPathONNX == "" {
		return fmt.Errorf("triton model path onnx must not be empty")
	}
	if s.ModelPathTRT == "" {
		return fmt.Errorf("triton model path trt must not be empty")
	}
	if s.Name == "" {
		return fmt.Errorf("triton model name must not be empty")
	}
	if s.Version <= 0 {
		return fmt.Errorf("triton model version must be positive")
	}
	return nil
}

type TritonModelConfig struct {
	Detector   TritonModelSpec `yaml:"detector"`
	Recognizer TritonModelSpec `yaml:"recognizer"`
}

func (c TritonModelConfig) Validate() error {
	if err := c.Detector.Validate(); err != nil {
		return fmt.Errorf("detector: %w", err)
	}
	if err := c.Recognizer.Validate(); err != nil {
		return fmt.Errorf("recognizer: %w", err)
	}
	return nil
}

type TritonConfig struct {
	Image               string            `yaml:"image"`
	ModelRepositoryPath string            `yaml:"model_repository_path"`
	IP                  string            `yaml:"ip"`
	HostPort            int               `yaml:"host_port"`
	Model               TritonModelConfig `yaml:"model"`
}

func (t TritonConfig) Validate() error {
	if t.Image == "" {
		return fmt.Errorf("triton image must not be empty")
	}
	if t.ModelRepositoryPath == "" {
		return fmt.Errorf("triton model repository path must not be empty")
	}
	if t.IP == "" {
		return fmt.Errorf("triton ip must not be empty")
	}
	if t.HostPort <= 0 || t.HostPort > 65535 {
		return fmt.Errorf("triton host port must be between 1 and 65535")
	}
	if err := t.Model.Validate(); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	return nil
}
