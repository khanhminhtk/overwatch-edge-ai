package log

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestConfigureJSONAndComponent(t *testing.T) {
	var output bytes.Buffer
	if err := Configure(Config{Level: slog.LevelDebug, Format: FormatJSON, Writer: &output}); err != nil {
		t.Fatal(err)
	}
	logger := New("kafka")
	logger.Info("published", "topic", "jobs")
	line := output.String()
	if !strings.Contains(line, `"component":"kafka"`) || !strings.Contains(line, `"topic":"jobs"`) {
		t.Fatalf("unexpected log: %s", line)
	}
}
func TestParseLevel(t *testing.T) {
	level, err := ParseLevel("warning")
	if err != nil || level != slog.LevelWarn {
		t.Fatalf("got %v, %v", level, err)
	}
	if _, err := ParseLevel("verbose"); err == nil {
		t.Fatal("expected invalid level error")
	}
}
func TestConfigureRejectsUnknownFormat(t *testing.T) {
	if err := Configure(Config{Format: "xml"}); err == nil {
		t.Fatal("expected invalid format error")
	}
}
