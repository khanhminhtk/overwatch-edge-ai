# Orchestrators Full Pipeline

Run this Go service as the control plane for Kafka pipeline commands. It persists a
pipeline run and its outgoing command in the same PostgreSQL transaction, then
an outbox publisher delivers commands to downstream services. Supported ingress
events are `train_requested`, `deploy_requested`, and `fine_tuning_requested`.

The pure transition rules are in `internal/application/workflow.go`; this keeps
topic routing testable without Kafka or PostgreSQL. Run with `go run ./cmd/full-pipeline` after setting `FULL_PIPELINE_POSTGRES_DSN`.
