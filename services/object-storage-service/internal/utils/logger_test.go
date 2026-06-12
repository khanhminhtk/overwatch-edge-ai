package utils

import (
	"bytes"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
)

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	os.Stdout = w

	defer func() {
		os.Stdout = old
	}()

	fn()

	_ = w.Close()
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	return buf.String()
}

func TestParseLevel(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected slog.Level
	}{
		{name: "debug lowercase", input: "debug", expected: slog.LevelDebug},
		{name: "debug uppercase", input: "DEBUG", expected: slog.LevelDebug},
		{name: "info", input: "info", expected: slog.LevelInfo},
		{name: "warn", input: "warn", expected: slog.LevelWarn},
		{name: "warning alias", input: "warning", expected: slog.LevelWarn},
		{name: "error", input: "error", expected: slog.LevelError},
		{name: "unknown defaults to info", input: "trace", expected: slog.LevelInfo},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseLevel(tc.input)
			if got != tc.expected {
				t.Fatalf("parseLevel(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestNewLogger_LocalEnv_UsesTextHandlerAndAddsMetadata(t *testing.T) {
	output := captureStdout(t, func() {
		config := ConfigLogger{
			Env:     "local",
			Level:   "debug",
			Service: "test-service",
			Version: "1.0.0",
		}
		logger := NewLogger(config)
		if logger == nil {
			t.Fatal("expected logger to be created, got nil")
		}
		logger.Info("hello local", "k", "v")
	})

	if !strings.Contains(output, "hello local") {
		t.Fatalf("expected output to contain log message, got: %q", output)
	}
	if !strings.Contains(output, "service=test-service") {
		t.Fatalf("expected output to contain service metadata, got: %q", output)
	}
	if !strings.Contains(output, "env=local") {
		t.Fatalf("expected output to contain env metadata, got: %q", output)
	}
	if !strings.Contains(output, "version=1.0.0") {
		t.Fatalf("expected output to contain version metadata, got: %q", output)
	}
}

func TestNewLogger_NonLocalEnv_UsesJSONHandlerAndAddsMetadata(t *testing.T) {
	output := captureStdout(t, func() {
		config := ConfigLogger{
			Env:     "prod",
			Level:   "info",
			Service: "test-service",
			Version: "1.0.0",
		}
		logger := NewLogger(config)
		if logger == nil {
			t.Fatal("expected logger to be created, got nil")
		}
		logger.Info("hello prod", "k", "v")
	})

	if !strings.Contains(output, "\"msg\":\"hello prod\"") {
		t.Fatalf("expected JSON log output, got: %q", output)
	}
	if !strings.Contains(output, "\"service\":\"test-service\"") {
		t.Fatalf("expected output to contain service metadata, got: %q", output)
	}
	if !strings.Contains(output, "\"env\":\"prod\"") {
		t.Fatalf("expected output to contain env metadata, got: %q", output)
	}
	if !strings.Contains(output, "\"version\":\"1.0.0\"") {
		t.Fatalf("expected output to contain version metadata, got: %q", output)
	}
}
