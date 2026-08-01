package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestConfigValidationAndEscapedDSN(t *testing.T) {
	cfg := Config{Host: "localhost", Port: 5432, Database: "test", User: "user", Password: "p@ss:w?rd"}
	dsn, err := cfg.DSN()
	if err != nil {
		t.Fatal(err)
	}
	if dsn == "" || dsn == "postgres://" {
		t.Fatalf("unexpected DSN %q", dsn)
	}
	cfg.Port = 0
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected validation error")
	}
	cfg.Port = 5432
	cfg.MinConnections = 2
	cfg.MaxConnections = 1
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected invalid pool configuration")
	}
}
func TestPoolOptionsValidation(t *testing.T) {
	if _, err := NewPool("postgres://example", nil, PoolOptions{MinConns: 2, MaxConns: 1}); err == nil {
		t.Fatal("expected invalid pool options")
	}
}
func TestHealthCheck(t *testing.T) {
	pool := &Pool{client: fakePool{connection: fakeConnection{tag: pgconn.NewCommandTag("SELECT 1")}}}
	if !NewHealthCheck(pool).Check(context.Background()) {
		t.Fatal("expected healthy")
	}
	pool.client = fakePool{connection: fakeConnection{err: errors.New("down")}}
	if NewHealthCheck(pool).Check(context.Background()) {
		t.Fatal("expected unhealthy")
	}
}
func TestSchemaGuardRunsEveryStatement(t *testing.T) {
	executor := &fakeExecutor{}
	if err := NewKafkaEventSchemaGuard(nil).Ensure(context.Background(), executor); err != nil {
		t.Fatal(err)
	}
	if len(executor.statements) != 5 {
		t.Fatalf("got %d statements", len(executor.statements))
	}
}

type fakeExecutor struct {
	statements []string
	err        error
}

func (f *fakeExecutor) Exec(_ context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
	f.statements = append(f.statements, sql)
	return pgconn.NewCommandTag("SELECT 1"), f.err
}

type fakeConnection struct {
	tag pgconn.CommandTag
	err error
}

func (f fakeConnection) Exec(_ context.Context, _ string, _ ...any) (pgconn.CommandTag, error) {
	return f.tag, f.err
}
func (f fakeConnection) Begin(context.Context) (pgx.Tx, error) {
	return nil, errors.New("not implemented")
}
func (f fakeConnection) Release() {}

type fakePool struct{ connection acquiredConnection }

func (f fakePool) Acquire(context.Context) (acquiredConnection, error) { return f.connection, nil }
func (f fakePool) Close()                                              {}
