package postgres

import "context"

// HealthCheck verifies that a connection can execute a trivial query.
type HealthCheck struct{ pool *Pool }

func NewHealthCheck(pool *Pool) HealthCheck { return HealthCheck{pool: pool} }
func (h HealthCheck) Check(ctx context.Context) bool {
	connection, err := h.pool.Acquire(ctx)
	if err != nil {
		return false
	}
	defer connection.Release()
	tag, err := connection.Exec(ctx, "SELECT 1")
	return err == nil && tag.String() == "SELECT 1"
}
