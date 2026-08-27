package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"full_pipeline/internal/application"
	"full_pipeline/internal/domain"
	store "full_pipeline/internal/infrastructure/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/segmentio/kafka-go"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	dsn := os.Getenv("FULL_PIPELINE_POSTGRES_DSN")
	if dsn == "" {
		fail(fmt.Errorf("FULL_PIPELINE_POSTGRES_DSN is required"))
	}
	brokers := env("FULL_PIPELINE_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		fail(err)
	}
	defer pool.Close()
	database := store.New(pool)
	if err := database.EnsureSchema(ctx); err != nil {
		fail(err)
	}
	producer := &kafka.Writer{Addr: kafka.TCP(brokers), RequiredAcks: kafka.RequireAll}
	defer producer.Close()
	commandReader := kafka.NewReader(kafka.ReaderConfig{Brokers: []string{brokers}, GroupID: env("FULL_PIPELINE_GROUP_ID", "full-pipeline"), Topic: env("FULL_PIPELINE_COMMAND_TOPIC", "full-pipeline.commands")})
	defer commandReader.Close()
	resultReader := kafka.NewReader(kafka.ReaderConfig{Brokers: []string{brokers}, GroupID: env("FULL_PIPELINE_GROUP_ID", "full-pipeline") + "-results", Topic: env("FULL_PIPELINE_RESULT_TOPIC", "full-pipeline.results")})
	defer resultReader.Close()
	workflow := application.Workflow{}
	for {
		if err := consumeOnce(ctx, commandReader, resultReader, database, producer, workflow); err != nil && ctx.Err() == nil {
			slog.Error("full pipeline cycle failed", "error", err)
			time.Sleep(time.Second)
		} else if ctx.Err() != nil {
			return
		}
	}
}

func consumeOnce(ctx context.Context, commands, results *kafka.Reader, database *store.Store, producer *kafka.Writer, workflow application.Workflow) error {
	resultCtx, cancelResult := context.WithTimeout(ctx, 50*time.Millisecond)
	resultMessage, resultErr := results.FetchMessage(resultCtx)
	cancelResult()
	if resultErr == nil {
		if err := handleResult(ctx, results, resultMessage, database, workflow); err != nil {
			return err
		}
	}
	commandCtx, cancelCommand := context.WithTimeout(ctx, 50*time.Millisecond)
	commandMessage, commandErr := commands.FetchMessage(commandCtx)
	cancelCommand()
	if commandErr == nil {
		if err := handleCommand(ctx, commands, commandMessage, database, workflow); err != nil {
			return err
		}
	}
	for _, item := range mustPending(ctx, database) {
		value, _ := json.Marshal(domain.Command{RequestID: item.RequestID, EventType: item.EventType, Payload: item.Payload})
		if err := producer.WriteMessages(ctx, kafka.Message{Topic: item.Topic, Key: []byte(item.RequestID), Value: value, Time: time.Now()}); err != nil {
			return err
		}
		if err := database.MarkPublished(ctx, item.ID); err != nil {
			return err
		}
	}
	return nil
}
func handleCommand(ctx context.Context, reader *kafka.Reader, message kafka.Message, database *store.Store, workflow application.Workflow) error {
	var command domain.Command
	if err := json.Unmarshal(message.Value, &command); err != nil {
		return err
	}
	if command.EventType == "fine_tuning_requested" {
		if _, exists := command.Payload["dataset_version"].(string); !exists {
			modelType, ok := command.Payload["model_type"].(string)
			if !ok || (modelType != "detection" && modelType != "recognizer") {
				return fmt.Errorf("fine_tuning_requested requires model_type")
			}
			version, err := database.AllocateDatasetVersion(ctx, modelType)
			if err != nil {
				return err
			}
			command.Payload["dataset_version"] = version
		}
	}
	next, err := workflow.Start(command)
	if err != nil {
		return err
	}
	if err = database.Start(ctx, command, next); err != nil {
		return err
	}
	return reader.CommitMessages(ctx, message)
}
func handleResult(ctx context.Context, reader *kafka.Reader, message kafka.Message, database *store.Store, workflow application.Workflow) error {
	var event domain.ResultEvent
	if err := json.Unmarshal(message.Value, &event); err != nil {
		return err
	}
	next, err := workflow.AfterResult(event)
	if err != nil {
		return err
	}
	if err = database.Result(ctx, event, next); err != nil {
		return err
	}
	return reader.CommitMessages(ctx, message)
}
func mustPending(ctx context.Context, database *store.Store) []store.OutboxRecord {
	records, err := database.Pending(ctx)
	if err != nil {
		slog.Error("read outbox failed", "error", err)
		return nil
	}
	return records
}
func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
func fail(err error) { fmt.Fprintln(os.Stderr, "full-pipeline:", err); os.Exit(1) }
