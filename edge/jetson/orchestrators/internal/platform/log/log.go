package log

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
)

type Format string

const (
	FormatText Format = "text"
	FormatJSON Format = "json"
)

// Config controls the process-wide default slog logger.
type Config struct {
	Level     slog.Level
	Format    Format
	Writer    io.Writer
	AddSource bool
}

func DefaultConfig() Config {
	return Config{Level: slog.LevelInfo, Format: FormatText, Writer: os.Stdout}
}

// ConfigFromEnv applies LOG_LEVEL and LOG_FORMAT to the defaults. Supported
// formats are text and json; supported levels are debug, info, warn, error.
func ConfigFromEnv() (Config, error) {
	config := DefaultConfig()
	if value := os.Getenv("LOG_LEVEL"); value != "" {
		level, err := ParseLevel(value)
		if err != nil {
			return Config{}, err
		}
		config.Level = level
	}
	if value := os.Getenv("LOG_FORMAT"); value != "" {
		config.Format = Format(strings.ToLower(value))
	}
	if config.Format != FormatText && config.Format != FormatJSON {
		return Config{}, fmt.Errorf("unsupported log format %q", config.Format)
	}
	return config, nil
}

func ParseLevel(value string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "debug":
		return slog.LevelDebug, nil
	case "info":
		return slog.LevelInfo, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("unsupported log level %q", value)
	}
}

// Configure replaces the process-wide default logger. Call it once during
// application bootstrap before constructing platform adapters.
func Configure(config Config) error {
	if config.Writer == nil {
		config.Writer = os.Stdout
	}
	if config.Format == "" {
		config.Format = FormatText
	}
	options := &slog.HandlerOptions{Level: config.Level, AddSource: config.AddSource}
	var handler slog.Handler
	switch config.Format {
	case FormatText:
		handler = slog.NewTextHandler(config.Writer, options)
	case FormatJSON:
		handler = slog.NewJSONHandler(config.Writer, options)
	default:
		return fmt.Errorf("unsupported log format %q", config.Format)
	}
	slog.SetDefault(slog.New(handler))
	return nil
}

// Logger adds a stable component field to a standard slog logger. Its methods
// satisfy the small logging interfaces used by persistence and messaging.
type Logger struct {
	name   string
	logger *slog.Logger
}

func New(name string) *Logger {
	return &Logger{name: name, logger: slog.Default().With("component", name)}
}
func (l *Logger) Name() string                      { return l.name }
func (l *Logger) Debug(message string, args ...any) { l.logger.Debug(message, args...) }
func (l *Logger) Info(message string, args ...any)  { l.logger.Info(message, args...) }
func (l *Logger) Warn(message string, args ...any)  { l.logger.Warn(message, args...) }
func (l *Logger) Error(message string, args ...any) { l.logger.Error(message, args...) }
func (l *Logger) With(args ...any) *Logger {
	return &Logger{name: l.name, logger: l.logger.With(args...)}
}
