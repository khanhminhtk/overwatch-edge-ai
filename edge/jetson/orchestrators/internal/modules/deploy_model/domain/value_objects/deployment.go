package valueobjects

import (
	"fmt"
	"time"
)

type ModelFormat string

const (
	ModelFormatPyTorch  ModelFormat = "pt"
	ModelFormatONNX     ModelFormat = "onnx"
	ModelFormatTensorRT ModelFormat = "trt"
)

func (f ModelFormat) Validate() error {
	switch f {
	case ModelFormatPyTorch, ModelFormatONNX, ModelFormatTensorRT:
		return nil
	default:
		return fmt.Errorf("unsupported model format %q", f)
	}
}

type DeploymentPolicy struct {
	HTTPTimeout      time.Duration
	ReadinessTimeout time.Duration
	PollInterval     time.Duration
}

func DefaultDeploymentPolicy() DeploymentPolicy {
	return DeploymentPolicy{HTTPTimeout: 10 * time.Second, ReadinessTimeout: 30 * time.Second, PollInterval: 500 * time.Millisecond}
}

func (p DeploymentPolicy) WithDefaults() DeploymentPolicy {
	defaults := DefaultDeploymentPolicy()
	if p.HTTPTimeout <= 0 {
		p.HTTPTimeout = defaults.HTTPTimeout
	}
	if p.ReadinessTimeout <= 0 {
		p.ReadinessTimeout = defaults.ReadinessTimeout
	}
	if p.PollInterval <= 0 {
		p.PollInterval = defaults.PollInterval
	}
	return p
}

func (p DeploymentPolicy) Validate() error {
	p = p.WithDefaults()
	if p.PollInterval > p.ReadinessTimeout {
		return fmt.Errorf("poll interval must not exceed readiness timeout")
	}
	return nil
}
