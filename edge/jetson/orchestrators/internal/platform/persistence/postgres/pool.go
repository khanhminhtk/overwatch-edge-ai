package postgres

import (
	"context"
	"fmt"
	"sync"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Logger is deliberately small so persistence does not depend on a logging
// implementation. A nil logger is accepted.
type Logger interface {
	Info(string, ...any)
	Warn(string, ...any)
	Error(string, ...any)
}

// PoolOptions controls pgx's pool sizing and connection lifetime.
type PoolOptions struct {
	MinConns int32
	MaxConns int32
}

func (o PoolOptions) withDefaults() PoolOptions {
	if o.MinConns == 0 {
		o.MinConns = 1
	}
	if o.MaxConns == 0 {
		o.MaxConns = 10
	}
	return o
}
func (o PoolOptions) validate() error {
	if o.MinConns < 0 {
		return fmt.Errorf("minimum connections cannot be negative")
	}
	if o.MaxConns < 1 {
		return fmt.Errorf("maximum connections must be positive")
	}
	if o.MinConns > o.MaxConns {
		return fmt.Errorf("minimum connections cannot exceed maximum connections")
	}
	return nil
}

// Executor is the smallest database capability used by health checks and
// schema guards. It makes those components easy to test without PostgreSQL.
type Executor interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}

type acquiredConnection interface {
	Executor
	Begin(context.Context) (pgx.Tx, error)
	Release()
}
type poolClient interface {
	Acquire(context.Context) (acquiredConnection, error)
	Close()
}
type poolFactory func(context.Context, string, PoolOptions) (poolClient, error)

type pgxPoolClient struct{ pool *pgxpool.Pool }

func (p pgxPoolClient) Acquire(ctx context.Context) (acquiredConnection, error) {
	return p.pool.Acquire(ctx)
}
func (p pgxPoolClient) Close() { p.pool.Close() }
func openPGXPool(ctx context.Context, dsn string, options PoolOptions) (poolClient, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	cfg.MinConns, cfg.MaxConns = options.MinConns, options.MaxConns
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return pgxPoolClient{pool}, nil
}

// Pool manages the lifecycle of a pgx connection pool.
type Pool struct {
	dsn     string
	options PoolOptions
	logger  Logger
	factory poolFactory
	mutex   sync.RWMutex
	client  poolClient
}

func NewPool(dsn string, logger Logger, options PoolOptions) (*Pool, error) {
	options = options.withDefaults()
	if err := options.validate(); err != nil {
		return nil, err
	}
	if dsn == "" {
		return nil, fmt.Errorf("%w: DSN must not be empty", ErrConnection)
	}
	return &Pool{dsn: dsn, options: options, logger: logger, factory: openPGXPool}, nil
}

func (p *Pool) Start(ctx context.Context) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.client != nil {
		p.warn("[POSTGRES_POOL_ALREADY_STARTED]")
		return nil
	}
	client, err := p.factory(ctx, p.dsn, p.options)
	if err != nil {
		return fmt.Errorf("%w: create pool: %w", ErrConnection, err)
	}
	p.client = client
	p.info("[POSTGRES_POOL_STARTED]")
	return nil
}
func (p *Pool) Stop() {
	p.mutex.Lock()
	client := p.client
	p.client = nil
	p.mutex.Unlock()
	if client == nil {
		p.warn("[POSTGRES_POOL_NOT_STARTED]")
		return
	}
	client.Close()
	p.info("[POSTGRES_POOL_STOPPED]")
}
func (p *Pool) Acquire(ctx context.Context) (acquiredConnection, error) {
	p.mutex.RLock()
	client := p.client
	p.mutex.RUnlock()
	if client == nil {
		p.error("[POSTGRES_POOL_NOT_STARTED]")
		return nil, ErrNotStarted
	}
	connection, err := client.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: acquire connection: %w", ErrConnection, err)
	}
	return connection, nil
}
func (p *Pool) info(message string) {
	if p.logger != nil {
		p.logger.Info(message)
	}
}
func (p *Pool) warn(message string) {
	if p.logger != nil {
		p.logger.Warn(message)
	}
}
func (p *Pool) error(message string) {
	if p.logger != nil {
		p.logger.Error(message)
	}
}
