package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"orchestrator/internal/modules/inference_runtime/adapters/outbound/cli"
	"orchestrator/internal/modules/inference_runtime/application/dto"
	"orchestrator/internal/modules/inference_runtime/application/usecase"
	platformconfig "orchestrator/internal/platform/config"
	platformlog "orchestrator/internal/platform/log"
)

type appConfig struct {
	Log              logConfig     `yaml:"log"`
	InferenceRuntime runtimeConfig `yaml:"inference_runtime"`
}
type logConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}
type runtimeConfig struct {
	Directory string `yaml:"directory"`
	UVPath    string `yaml:"uv_path"`
	Module    string `yaml:"module"`
	Display   bool   `yaml:"display"`
}

func main() {
	configDir := flag.String("config-dir", "", "path to orchestrator config directory")
	runtimeDirOverride := flag.String("runtime-dir", "", "override inference-runtime directory")
	display := flag.Bool("display", false, "override configured --display value")
	flag.Parse()

	resolvedConfigDir, err := resolveConfigDir(*configDir)
	if err != nil {
		fail(err)
	}
	settings, err := platformconfig.Load[appConfig](platformconfig.Options{YAMLFiles: []string{filepath.Join(resolvedConfigDir, "orchestrator.yaml")}, EnvFiles: []string{filepath.Join(resolvedConfigDir, ".env")}, IncludeOSEnv: true})
	if err != nil {
		fail(fmt.Errorf("load config: %w", err))
	}
	level, err := platformlog.ParseLevel(settings.Log.Level)
	if err != nil {
		fail(err)
	}
	if err := platformlog.Configure(platformlog.Config{Level: level, Format: platformlog.Format(settings.Log.Format)}); err != nil {
		fail(err)
	}

	configuredRuntimeDir := settings.InferenceRuntime.Directory
	if *runtimeDirOverride != "" {
		configuredRuntimeDir = *runtimeDirOverride
	}
	resolvedRuntimeDir, err := absoluteRuntimeDir(filepath.Dir(resolvedConfigDir), configuredRuntimeDir)
	if err != nil {
		fail(err)
	}
	runner, err := cli.NewRunner(cli.Config{RuntimeDir: resolvedRuntimeDir, UVPath: settings.InferenceRuntime.UVPath, DaemonModule: settings.InferenceRuntime.Module})
	if err != nil {
		fail(err)
	}
	runner.Logger = platformlog.New("inference-runtime")
	runInference, err := usecase.NewRunInference(runner)
	if err != nil {
		fail(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	runDisplay := settings.InferenceRuntime.Display
	flag.Visit(func(item *flag.Flag) {
		if item.Name == "display" {
			runDisplay = *display
		}
	})
	err = runInference.Execute(ctx, dto.RunRequest{Display: runDisplay, Arguments: flag.Args()})
	if err != nil {
		fail(err)
	}
}

func resolveConfigDir(configured string) (string, error) {
	if configured != "" {
		return filepath.Abs(configured)
	}
	directory, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory: %w", err)
	}
	for {
		candidate := filepath.Join(directory, "edge", "jetson", "orchestrator", "config")
		if info, err := os.Stat(filepath.Join(candidate, "orchestrator.yaml")); err == nil && !info.IsDir() {
			return candidate, nil
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
		directory = parent
	}
	return "", fmt.Errorf("cannot locate edge/jetson/orchestrator/config; provide -config-dir")
}

func absoluteRuntimeDir(orchestratorDir, configured string) (string, error) {
	if configured == "" {
		return "", fmt.Errorf("inference runtime directory must not be empty")
	}
	if filepath.IsAbs(configured) {
		return configured, nil
	}
	return filepath.Abs(filepath.Join(orchestratorDir, configured))
}

func fail(err error) { fmt.Fprintln(os.Stderr, "inference-runtime:", err); os.Exit(1) }
