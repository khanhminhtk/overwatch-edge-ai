package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	postgres "orchestrator/internal/platform/persistence/postgres"
)

type KafkaEvent struct {
	ID            int64
	RequestID     string
	Topic         string
	PartitionID   int32
	MessageOffset int64
	ConsumerGroup string
	MessageKey    *string
	EventType     string
	SchemaName    *string
	SchemaVersion *string
	Payload       json.RawMessage
	Status        string
	ErrorMessage  *string
	ProducedAt    *time.Time
	ConsumedAt    time.Time
	ProcessedAt   *time.Time
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

func main() {
	host := flag.String(
		"host",
		getenv("POSTGRES_HOST", "localhost"),
		"PostgreSQL host",
	)
	port := flag.Int(
		"port",
		getenvInt("POSTGRES_PORT", 5432),
		"PostgreSQL port",
	)
	database := flag.String(
		"database",
		getenv("POSTGRES_DB", "overwatch"),
		"PostgreSQL database",
	)
	user := flag.String(
		"user",
		getenv("POSTGRES_USER", "postgres"),
		"PostgreSQL user",
	)
	password := flag.String(
		"password",
		getenv("POSTGRES_PASSWORD", "postgres"),
		"PostgreSQL password",
	)
	timeout := flag.Duration(
		"timeout",
		10*time.Second,
		"connection and query timeout",
	)

	flag.Parse()

	config := postgres.Config{
		Host:     *host,
		Port:     *port,
		Database: *database,
		User:     *user,
		Password: *password,
	}

	dsn, err := config.DSN()
	if err != nil {
		fail(fmt.Errorf("build PostgreSQL DSN: %w", err))
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	pool, err := postgres.NewPool(
		dsn,
		nil,
		postgres.PoolOptions{
			MaxConns: 1,
		},
	)
	if err != nil {
		fail(fmt.Errorf("connect to PostgreSQL: %w", err))
	}
	if err := pool.Start(ctx); err != nil {
		fail(fmt.Errorf("start PostgreSQL pool: %w", err))
	}
	defer pool.Stop()

	fmt.Printf(
		"connecting host=%s port=%d database=%s user=%s\n",
		config.Host,
		config.Port,
		config.Database,
		config.User,
	)
	if !postgres.NewHealthCheck(pool).Check(ctx) {
		fail(fmt.Errorf("PostgreSQL health check failed"))
	}

	transaction := postgres.NewTransaction(pool)

	err = transaction.Run(ctx, func(tx pgx.Tx) error {
		rows, err := tx.Query(ctx, `
			SELECT
				id,
				request_id,
				topic,
				partition_id,
				message_offset,
				consumer_group,
				message_key,
				event_type,
				schema_name,
				schema_version,
				payload,
				status,
				error_message,
				produced_at,
				consumed_at,
				processed_at,
				created_at,
				updated_at
			FROM kafka_events
			ORDER BY created_at DESC
			LIMIT 10
		`)
		if err != nil {
			return fmt.Errorf("query kafka events: %w", err)
		}
		defer rows.Close()

		count := 0

		for rows.Next() {
			var event KafkaEvent

			if err := rows.Scan(
				&event.ID,
				&event.RequestID,
				&event.Topic,
				&event.PartitionID,
				&event.MessageOffset,
				&event.ConsumerGroup,
				&event.MessageKey,
				&event.EventType,
				&event.SchemaName,
				&event.SchemaVersion,
				&event.Payload,
				&event.Status,
				&event.ErrorMessage,
				&event.ProducedAt,
				&event.ConsumedAt,
				&event.ProcessedAt,
				&event.CreatedAt,
				&event.UpdatedAt,
			); err != nil {
				return fmt.Errorf("scan kafka event: %w", err)
			}

			count++

			fmt.Printf("\n========== Kafka Event #%d ==========\n", count)
			fmt.Printf("ID             : %d\n", event.ID)
			fmt.Printf("Request ID     : %s\n", event.RequestID)
			fmt.Printf("Topic          : %s\n", event.Topic)
			fmt.Printf("Partition ID   : %d\n", event.PartitionID)
			fmt.Printf("Message Offset : %d\n", event.MessageOffset)
			fmt.Printf("Consumer Group : %s\n", event.ConsumerGroup)
			fmt.Printf("Message Key    : %s\n", stringValue(event.MessageKey))
			fmt.Printf("Event Type     : %s\n", event.EventType)
			fmt.Printf("Schema Name    : %s\n", stringValue(event.SchemaName))
			fmt.Printf("Schema Version : %s\n", stringValue(event.SchemaVersion))
			fmt.Printf("Status         : %s\n", event.Status)
			fmt.Printf("Error Message  : %s\n", stringValue(event.ErrorMessage))
			fmt.Printf("Produced At    : %s\n", formatOptionalTime(event.ProducedAt))
			fmt.Printf("Consumed At    : %s\n", formatTime(event.ConsumedAt))
			fmt.Printf("Processed At   : %s\n", formatOptionalTime(event.ProcessedAt))
			fmt.Printf("Created At     : %s\n", formatTime(event.CreatedAt))
			fmt.Printf("Updated At     : %s\n", formatTime(event.UpdatedAt))
			fmt.Printf("Payload        :\n%s\n", formatJSON(event.Payload))
		}

		if err := rows.Err(); err != nil {
			return fmt.Errorf("iterate kafka events: %w", err)
		}

		if count == 0 {
			fmt.Println("Không có record nào trong bảng kafka_events")
			return nil
		}

		fmt.Printf("\nTotal events: %d\n", count)

		return nil
	})
	if err != nil {
		fail(fmt.Errorf("run transaction: %w", err))
	}

	fmt.Println("PostgreSQL kafka_events smoke test passed")
}

func formatJSON(data json.RawMessage) string {
	if len(data) == 0 {
		return "NULL"
	}

	var prettyJSON any

	if err := json.Unmarshal(data, &prettyJSON); err != nil {
		return string(data)
	}

	formatted, err := json.MarshalIndent(prettyJSON, "", "  ")
	if err != nil {
		return string(data)
	}

	return string(formatted)
}

func stringValue(value *string) string {
	if value == nil {
		return "NULL"
	}
	return *value
}

func formatTime(value time.Time) string {
	return value.Format(time.RFC3339Nano)
}

func formatOptionalTime(value *time.Time) string {
	if value == nil {
		return "NULL"
	}
	return formatTime(*value)
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}

	return parsed
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "test-sql:", err)
	os.Exit(1)
}
