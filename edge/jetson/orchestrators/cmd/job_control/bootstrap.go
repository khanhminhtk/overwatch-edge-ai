package main

import (
	"context"
	"fmt"

	"orchestrator/internal/modules/job_control/adapters/outbound/scheduler"
	platformlog "orchestrator/internal/platform/log"
	kafkaPkg "orchestrator/internal/platform/messaging/kafka"
	postgresPkg "orchestrator/internal/platform/persistence/postgres"
)

// App owns command-level infrastructure and the long-running adapters it wires.
// It is intentionally not a service layer: domain behavior remains in modules.
type App struct {
	ConfigDir string
	Configs   appConfig
	Logger    *platformlog.Logger
	Postgres  *postgresPkg.Pool
	Consumers []*kafkaPkg.Consumer
	Workers   []*scheduler.PollingRunner
}

func newApp(ctx context.Context, configDir string) (*App, error) {
	settings, err := loadAppConfig(configDir)
	if err != nil {
		return nil, err
	}
	logger, err := buildLogger(settings.Log)
	if err != nil {
		return nil, err
	}
	pool, err := buildPostgresPool(ctx, settings.Postgres, logger)
	if err != nil {
		return nil, err
	}
	cleanup := func(err error) (*App, error) { pool.Stop(); return nil, err }
	if err := postgresPkg.NewKafkaEventSchemaGuard(logger).Ensure(ctx, pool); err != nil {
		return cleanup(fmt.Errorf("ensure job schema: %w", err))
	}
	transaction := postgresPkg.NewTransaction(pool)
	consumers, err := newConsumers(settings.Kafka, transaction, logger)
	if err != nil {
		return cleanup(err)
	}
	workers, err := newPollingRunners(configDir, settings, transaction, logger)
	if err != nil {
		return cleanup(err)
	}
	return &App{ConfigDir: configDir, Configs: settings, Logger: logger, Postgres: pool, Consumers: consumers, Workers: workers}, nil
}

func (a *App) Close() {
	if a != nil && a.Postgres != nil {
		a.Postgres.Stop()
	}
}

func buildLogger(settings loggingConfig) (*platformlog.Logger, error) {
	level, err := platformlog.ParseLevel(settings.Level)
	if err != nil {
		return nil, fmt.Errorf("parse logging level: %w", err)
	}
	if err := platformlog.Configure(platformlog.Config{Level: level, Format: platformlog.Format(settings.Format)}); err != nil {
		return nil, fmt.Errorf("configure logger: %w", err)
	}
	return platformlog.New("job-control"), nil
}

func buildPostgresPool(ctx context.Context, settings postgresPkg.Config, logger *platformlog.Logger) (*postgresPkg.Pool, error) {
	dsn, err := settings.DSN()
	if err != nil {
		return nil, fmt.Errorf("build PostgreSQL DSN: %w", err)
	}
	pool, err := postgresPkg.NewPool(dsn, logger, postgresPkg.PoolOptions{MinConns: settings.MinConnections, MaxConns: settings.MaxConnections})
	if err != nil {
		return nil, fmt.Errorf("create PostgreSQL pool: %w", err)
	}
	if err := pool.Start(ctx); err != nil {
		return nil, fmt.Errorf("start PostgreSQL pool: %w", err)
	}
	return pool, nil
}
