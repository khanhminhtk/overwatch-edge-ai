package main

import (
	"fmt"
	"sort"
	"strings"

	inboundkafka "orchestrator/internal/modules/job_control/adapters/inbound/kafka"
	jobpostgres "orchestrator/internal/modules/job_control/adapters/outbound/persistence/postgres"
	"orchestrator/internal/modules/job_control/application/usecase"
	platformlog "orchestrator/internal/platform/log"
	kafkaPkg "orchestrator/internal/platform/messaging/kafka"
	postgresPkg "orchestrator/internal/platform/persistence/postgres"
)

func newConsumers(kafkaConfig kafkaPkg.Config, transaction postgresPkg.Transaction, logger *platformlog.Logger) ([]*kafkaPkg.Consumer, error) {
	jobNames := make([]string, 0, len(kafkaConfig.Jobs))
	for name := range kafkaConfig.Jobs {
		jobNames = append(jobNames, name)
	}
	sort.Strings(jobNames)
	consumers := make([]*kafkaPkg.Consumer, 0, len(jobNames))
	for _, name := range jobNames {
		job := kafkaConfig.Jobs[name]
		if strings.TrimSpace(job.Topic) == "" || strings.TrimSpace(job.GroupID) == "" {
			return nil, fmt.Errorf("Kafka job %q requires topic and group_id", name)
		}
		eventTypes := job.EventTypes()
		if len(eventTypes) == 0 {
			return nil, fmt.Errorf("Kafka job %q requires event_type", name)
		}
		repository, err := jobpostgres.NewRepository(transaction, eventTypes[0], 1800)
		if err != nil {
			return nil, fmt.Errorf("create repository for Kafka job %q: %w", name, err)
		}
		ingest, err := usecase.NewIngestJob(repository, logger.With("job", name), eventTypes)
		if err != nil {
			return nil, fmt.Errorf("create ingestion use case for Kafka job %q: %w", name, err)
		}
		handler, err := inboundkafka.NewIngestHandler(ingest, job.GroupID, name, "1.0")
		if err != nil {
			return nil, fmt.Errorf("create Kafka handler for job %q: %w", name, err)
		}
		consumer, err := kafkaPkg.NewConsumer(kafkaConfig.WithConsumerOverride(job.Consumer), job.GroupID, []string{job.Topic}, handler.Handle, logger.With("job", name))
		if err != nil {
			return nil, fmt.Errorf("create Kafka consumer for job %q: %w", name, err)
		}
		consumers = append(consumers, consumer)
		logger.Info("[cmd.job_control.newConsumers] - Kafka consumer initialized.", "job", name, "topic", job.Topic, "group_id", job.GroupID, "event_types", eventTypes)
	}
	if len(consumers) == 0 {
		return nil, fmt.Errorf("no Kafka jobs are configured")
	}
	return consumers, nil
}
