package valueobjects

import (
	"os/exec"
)

type SystemConfig struct {
	DockerPath string `yaml:"docker_path"`
}

func (s SystemConfig) Validate() error {
	if _, err := exec.LookPath(s.DockerPath); err != nil {
		return err
	}
	if _, err := exec.Command(s.DockerPath, "--version").CombinedOutput(); err != nil {
		return err
	}
	return nil
}
