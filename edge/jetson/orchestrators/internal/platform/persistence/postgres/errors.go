package postgres

import "errors"

var (
	ErrPostgres    = errors.New("postgres error")
	ErrNotStarted  = errors.New("postgres pool is not started")
	ErrConnection  = errors.New("postgres connection error")
	ErrTransaction = errors.New("postgres transaction error")
)
