package utils

import (
	"errors"
	"fmt"
	"os"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

type IConfigLoader[T any] interface {
	LoadConfig() (*T, error)
}

type ConfigLoader[T any] struct {
	ConfigPath string
	EnvPath    string
}

func NewConfigLoader[T any](configPath, envPath string) *ConfigLoader[T] {
	return &ConfigLoader[T]{
		ConfigPath: configPath,
		EnvPath:    envPath,
	}
}

func (cl *ConfigLoader[T]) LoadConfig() (*T, error) {
	if cl.EnvPath != "" {
		if err := godotenv.Load(cl.EnvPath); err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				return nil, fmt.Errorf("load env file failed: %w", err)
			}
		}
	}

	data, err := os.ReadFile(cl.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("read config file failed: %w", err)
	}

	expanded := string(data)
	for i := 0; i < 3; i++ {
		next := os.ExpandEnv(expanded)
		if next == expanded {
			break
		}
		expanded = next
	}

	var config T
	if err := yaml.Unmarshal([]byte(expanded), &config); err != nil {
		return nil, fmt.Errorf("unmarshal config file failed: %w", err)
	}
	return &config, nil
}
