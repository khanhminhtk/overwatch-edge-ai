package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// YAMLConfigSource loads YAML files in order. Later files override earlier
// values; nested mappings are merged and lists are replaced.
type YAMLConfigSource struct {
	Paths []string
}

func (s YAMLConfigSource) Load() (map[string]any, error) {
	merged := map[string]any{}
	for _, path := range s.Paths {
		contents, err := os.ReadFile(filepath.Clean(path))
		if err != nil {
			return nil, fmt.Errorf("read YAML config %q: %w", path, err)
		}
		loaded := map[string]any{}
		if err := yaml.Unmarshal(contents, &loaded); err != nil {
			return nil, fmt.Errorf("parse YAML config %q: %w", path, err)
		}
		if loaded == nil {
			loaded = map[string]any{}
		}
		merged = DeepMerge(merged, loaded).(map[string]any)
	}
	return merged, nil
}

// DotenvEnvSource loads .env files in order without mutating the process
// environment. Later files override earlier values.
type DotenvEnvSource struct {
	Paths []string
}

func (s DotenvEnvSource) Load() (map[string]string, error) {
	merged := map[string]string{}
	for _, path := range s.Paths {
		file, err := os.Open(filepath.Clean(path))
		if err != nil {
			return nil, fmt.Errorf("read dotenv file %q: %w", path, err)
		}
		values, parseErr := parseDotenv(file)
		closeErr := file.Close()
		if parseErr != nil {
			return nil, fmt.Errorf("parse dotenv file %q: %w", path, parseErr)
		}
		if closeErr != nil {
			return nil, fmt.Errorf("close dotenv file %q: %w", path, closeErr)
		}
		for key, value := range values {
			merged[key] = value
		}
	}
	return merged, nil
}

func parseDotenv(file *os.File) (map[string]string, error) {
	values := map[string]string{}
	scanner := bufio.NewScanner(file)
	line := 0
	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		text = strings.TrimPrefix(text, "export ")
		key, value, found := strings.Cut(text, "=")
		key = strings.TrimSpace(key)
		if !found || key == "" {
			return nil, fmt.Errorf("line %d: expected KEY=VALUE", line)
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 && ((value[0] == '\'' && value[len(value)-1] == '\'') || (value[0] == '"' && value[len(value)-1] == '"')) {
			value = value[1 : len(value)-1]
		}
		values[key] = value
	}
	return values, scanner.Err()
}

// EnvironmentEnvSource reads Values, or the operating-system environment when
// Values is nil. It is useful for deterministic tests as well as production.
type EnvironmentEnvSource struct {
	Values map[string]string
}

func (s EnvironmentEnvSource) Load() (map[string]string, error) {
	if s.Values != nil {
		return cloneStrings(s.Values), nil
	}
	values := map[string]string{}
	for _, entry := range os.Environ() {
		key, value, found := strings.Cut(entry, "=")
		if found {
			values[key] = value
		}
	}
	return values, nil
}
