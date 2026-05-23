package minio

import (
	"context"
	"fmt"
	"net"
	"strings"
	"time"

	inbound "backend_nexus/internal/adapter/inbound"
	"backend_nexus/internal/infra"
)

const (
	DependencyMinioConnection     = "minio_connection"
	DependencyMinioClientActivity = "minio_client_activity"
)

func SyncHealthStatus(
	ctx context.Context,
	healthManager *inbound.HealthManager,
	objectStorage *infra.MinioObjectStorage,
	endpoint string,
	timeout time.Duration,
) {
	if healthManager == nil {
		return
	}

	healthManager.SetReady(true)

	connErr := checkMinioConnection(endpoint, timeout)
	healthManager.SetDependency(DependencyMinioConnection, connErr == nil, messageFromErr(connErr))

	clientErr := checkMinioClientActivity(ctx, objectStorage)
	healthManager.SetDependency(DependencyMinioClientActivity, clientErr == nil, messageFromErr(clientErr))
}

func checkMinioConnection(endpoint string, timeout time.Duration) error {
	address, err := normalizeEndpointAddress(endpoint)
	if err != nil {
		return err
	}

	if timeout <= 0 {
		timeout = 2 * time.Second
	}

	conn, err := net.DialTimeout("tcp", address, timeout)
	if err != nil {
		return fmt.Errorf("tcp dial failed: %w", err)
	}
	_ = conn.Close()
	return nil
}

func checkMinioClientActivity(ctx context.Context, objectStorage *infra.MinioObjectStorage) error {
	if objectStorage == nil || objectStorage.Client == nil {
		return fmt.Errorf("minio client is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	_, err := objectStorage.Client.ListBuckets()
	if err != nil {
		return fmt.Errorf("list buckets failed: %w", err)
	}
	return nil
}

func normalizeEndpointAddress(endpoint string) (string, error) {
	trimmed := strings.TrimSpace(endpoint)
	if trimmed == "" {
		return "", fmt.Errorf("minio endpoint is empty")
	}
	if strings.Contains(trimmed, "://") {
		parts := strings.SplitN(trimmed, "://", 2)
		trimmed = parts[1]
	}
	if strings.Contains(trimmed, "/") {
		trimmed = strings.SplitN(trimmed, "/", 2)[0]
	}
	if !strings.Contains(trimmed, ":") {
		trimmed += ":9000"
	}
	return trimmed, nil
}

func messageFromErr(err error) string {
	if err == nil {
		return "ok"
	}
	return err.Error()
}
