package config

import (
	"os"
	"path/filepath"
	"testing"
)

type serverConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	Enabled bool   `yaml:"enabled"`
}
type appConfig struct {
	Server serverConfig `yaml:"server"`
	Tags   []string     `yaml:"tags"`
}

func TestLoadMergesSourcesResolvesEnvAndDecodesStruct(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.yaml")
	override := filepath.Join(dir, "override.yaml")
	if err := os.WriteFile(base, []byte("server:\n  host: ${HOST}\n  port: 8080\n  enabled: false\ntags: [base]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(override, []byte("server:\n  port: ${PORT:-9090}\n  enabled: 'yes'\ntags: [override]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load[appConfig](Options{YAMLFiles: []string{base, override}, Env: map[string]string{"HOST": "edge.local"}})
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Server.Host != "edge.local" || loaded.Server.Port != 9090 || !loaded.Server.Enabled {
		t.Fatalf("unexpected server: %#v", loaded.Server)
	}
	if len(loaded.Tags) != 1 || loaded.Tags[0] != "override" {
		t.Fatalf("lists must be replaced, got %#v", loaded.Tags)
	}
}

func TestProviderSelectsSectionAndReportsMissingEnvironment(t *testing.T) {
	provider := Provider{ConfigSources: []ConfigSource{staticConfig{"app": map[string]any{"name": "${NAME}"}}}}
	raw, err := provider.Raw("app")
	if err != nil || raw["name"] != "${NAME}" {
		t.Fatalf("raw section: %#v, %v", raw, err)
	}
	var target struct {
		Name string `yaml:"name"`
	}
	if err := provider.Require(&target, "app"); err == nil {
		t.Fatal("expected missing environment error")
	}
}

func TestDotenvDoesNotMutateProcessEnvironment(t *testing.T) {
	file := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(file, []byte("CONFIG_TEST_ONLY=value\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := (DotenvEnvSource{Paths: []string{file}}).Load()
	if err != nil {
		t.Fatal(err)
	}
	if _, found := os.LookupEnv("CONFIG_TEST_ONLY"); found {
		t.Fatal("dotenv source mutated process environment")
	}
}

type staticConfig map[string]any

func (s staticConfig) Load() (map[string]any, error) { return s, nil }
