package postgres

import (
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
)

// Config contains the connection settings required by PostgreSQL.
type Config struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	Database       string `yaml:"database"`
	User           string `yaml:"user"`
	Password       string `yaml:"password"`
	MinConnections int32  `yaml:"min_connections"`
	MaxConnections int32  `yaml:"max_connections"`
}

// Validate rejects incomplete or invalid connection settings before a network
// connection is attempted.
func (c Config) Validate() error {
	if strings.TrimSpace(c.Host) == "" {
		return fmt.Errorf("Postgres host must not be empty")
	}
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("Postgres port must be in the range 1-65535: %d", c.Port)
	}
	if strings.TrimSpace(c.Database) == "" {
		return fmt.Errorf("Postgres database must not be empty")
	}
	if strings.TrimSpace(c.User) == "" {
		return fmt.Errorf("Postgres user must not be empty")
	}
	if strings.TrimSpace(c.Password) == "" {
		return fmt.Errorf("Postgres password must not be empty")
	}
	if err := (PoolOptions{MinConns: c.MinConnections, MaxConns: c.MaxConnections}).withDefaults().validate(); err != nil {
		return fmt.Errorf("Postgres pool configuration: %w", err)
	}
	return nil
}

// DSN returns an escaped PostgreSQL connection URI.
func (c Config) DSN() (string, error) {
	if err := c.Validate(); err != nil {
		return "", err
	}
	return (&url.URL{Scheme: "postgres", User: url.UserPassword(c.User, c.Password), Host: net.JoinHostPort(c.Host, strconv.Itoa(c.Port)), Path: c.Database}).String(), nil
}

// Values returns a copy suitable for diagnostics. Do not log Password.
func (c Config) Values() map[string]any {
	return map[string]any{"host": c.Host, "port": c.Port, "database": c.Database, "user": c.User, "password": c.Password}
}
