package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// Transaction executes work atomically, committing on nil and rolling back on
// an error. The acquired connection is always released.
type Transaction struct{ pool *Pool }

func NewTransaction(pool *Pool) Transaction { return Transaction{pool: pool} }
func (t Transaction) Run(ctx context.Context, work func(pgx.Tx) error) error {
	if work == nil {
		return fmt.Errorf("%w: work function must not be nil", ErrTransaction)
	}
	connection, err := t.pool.Acquire(ctx)
	if err != nil {
		return err
	}
	defer connection.Release()
	tx, err := connection.Begin(ctx)
	if err != nil {
		return fmt.Errorf("%w: begin: %w", ErrTransaction, err)
	}
	if err := work(tx); err != nil {
		return errors.Join(fmt.Errorf("%w: work: %w", ErrTransaction, err), tx.Rollback(ctx))
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("%w: commit: %w", ErrTransaction, err)
	}
	return nil
}
