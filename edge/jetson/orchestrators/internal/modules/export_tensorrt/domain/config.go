package domain

import (
	"fmt"
	"strings"
)

type Config struct {
	TritonDockerEnvFile            string `yaml:"triton_docker_env_file"`
	TritonDockerComposeFile        string `yaml:"triton_docker_compose_file"`
	TritonBuilderProfileFormat     string `yaml:"triton_builder_profile_format"`
	TritonBuilderServiceNameFormat string `yaml:"triton_builder_service_name_format"`
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.TritonDockerEnvFile) == "" {
		return fmt.Errorf("triton docker env file must not be empty")
	}
	if strings.TrimSpace(c.TritonDockerComposeFile) == "" {
		return fmt.Errorf("triton docker compose file must not be empty")
	}
	if !strings.Contains(c.TritonBuilderProfileFormat, "%s") {
		return fmt.Errorf("triton builder profile format must contain %%s")
	}
	if !strings.Contains(c.TritonBuilderServiceNameFormat, "%s") {
		return fmt.Errorf("triton builder service-name format must contain %%s")
	}
	return nil
}
