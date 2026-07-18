# PostgreSQL persistence

This package is the Go equivalent of the model-lifecycle service persistence
platform. It provides validated connection configuration, an explicit `pgxpool`
lifecycle, health checks, transactional callbacks, and the Kafka-event schema
guard.

```go
dsn, err := config.DSN()
pool, err := postgres.NewPool(dsn, logger, postgres.PoolOptions{MaxConns: 10})
if err != nil { return err }
if err := pool.Start(ctx); err != nil { return err }
defer pool.Stop()

err = postgres.NewTransaction(pool).Run(ctx, func(tx pgx.Tx) error {
	_, err := tx.Exec(ctx, "INSERT INTO ...")
	return err
})
```

`Start` is idempotent, `Stop` is safe before startup, and `Acquire` returns
`ErrNotStarted` until the pool has started. All operations take a context.
