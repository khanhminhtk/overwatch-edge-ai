package utils

import (
	"log/slog"
	"os"
	"strings"
)

type ConfigLogger struct {
	Env     string
	Level   string
	Service string
	Version string
}

func parseLevel(level string) slog.Level {
	switch strings.ToLower(level) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

type Logger struct {
	*slog.Logger
}

func NewLogger(config ConfigLogger) *Logger {
	level := parseLevel(config.Level)
	opts := slog.HandlerOptions{
		Level:     level,
		AddSource: config.Env != "prod",
	}

	var handler slog.Handler
	if config.Env == "local" {
		handler = slog.NewTextHandler(os.Stdout, &opts)
	} else {
		handler = slog.NewJSONHandler(os.Stdout, &opts)
	}
	base := slog.New(handler).With(
		slog.String("service", config.Service),
		slog.String("env", config.Env),
		slog.String("version", config.Version),
	)

	return &Logger{base}
}

func (l *Logger) Debug(msg string, args ...any) {
	l.Logger.Debug(msg, args...)
}

func (l *Logger) Info(msg string, args ...any) {
	l.Logger.Info(msg, args...)
}

func (l *Logger) Warn(msg string, args ...any) {
	l.Logger.Warn(msg, args...)
}

func (l *Logger) Error(msg string, args ...any) {
	l.Logger.Error(msg, args...)
}
