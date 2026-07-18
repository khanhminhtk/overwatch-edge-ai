# Structured logging

The platform logger wraps Go's standard `log/slog` package. Configure it once
at startup, then inject named loggers into adapters:

```go
cfg, err := log.ConfigFromEnv() // LOG_LEVEL, LOG_FORMAT=text|json
if err != nil { return err }
if err := log.Configure(cfg); err != nil { return err }

logger := log.New("kafka-producer")
logger.Info("message published", "topic", topic, "offset", offset)
```

`*log.Logger` implements `Info`, `Warn`, and `Error` methods compatible with
the PostgreSQL and Kafka platform components.
