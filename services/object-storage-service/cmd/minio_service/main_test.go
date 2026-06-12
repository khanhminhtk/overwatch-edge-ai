package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunWithPaths_InvalidUploadURLTTL(t *testing.T) {
	t.Setenv("MINIO_ENDPOINT", "127.0.0.1:9000")
	t.Setenv("MINIO_ACCESS_KEY", "minio")
	t.Setenv("MINIO_SECRET_KEY", "minio123")
	t.Setenv("MINIO_SESSION_TOKEN", "")
	t.Setenv("MINIO_USE_SSL", "false")
	t.Setenv("MINIO_REGION", "us-east-1")
	t.Setenv("MINIO_HEALTH_CHECK", "false")
	t.Setenv("MINIO_INSECURE_SKIP_TLS", "true")
	t.Setenv("MINIOSERVICE_GRPC_PORT", "50001")
	t.Setenv("MINIOSERVICE_GRPC_JWT_SECRET", "jwt-secret")
	t.Setenv("MINIOSERVICE_GRPC_REFRESH_TOKEN", "refresh-token")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_ACCESS_TOKEN", "300")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_URL_UPLOAD", "0")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_URL_DOWNLOAD", "300")

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "minio_config.yaml")
	envPath := filepath.Join(tmpDir, ".env")

	if err := os.WriteFile(envPath, []byte(""), 0o644); err != nil {
		t.Fatalf("write env file: %v", err)
	}

	cfg := `minio_server:
  endpoint: ${MINIO_ENDPOINT}
  access_key: ${MINIO_ACCESS_KEY}
  secret_key: ${MINIO_SECRET_KEY}
  session_token: ${MINIO_SESSION_TOKEN}
  use_ssl: ${MINIO_USE_SSL}
  region: ${MINIO_REGION}
  health_check: ${MINIO_HEALTH_CHECK}
  insecure_skip_tls: ${MINIO_INSECURE_SKIP_TLS}
  config_service_grpc:
    grpc_port: ${MINIOSERVICE_GRPC_PORT}
    jwt_secret: ${MINIOSERVICE_GRPC_JWT_SECRET}
    refresh_token: ${MINIOSERVICE_GRPC_REFRESH_TOKEN}
    expires_time_access_token: ${MINIOSERVICE_GRPC_EXPIRES_TIME_ACCESS_TOKEN}
    expires_time_url_upload: ${MINIOSERVICE_GRPC_EXPIRES_TIME_URL_UPLOAD}
    expires_time_url_download: ${MINIOSERVICE_GRPC_EXPIRES_TIME_URL_DOWNLOAD}
`
	if err := os.WriteFile(configPath, []byte(cfg), 0o644); err != nil {
		t.Fatalf("write config file: %v", err)
	}

	err := runWithPaths(configPath, envPath)
	if err == nil {
		t.Fatal("expected error for invalid upload url ttl, got nil")
	}
	if !strings.Contains(err.Error(), "upload url ttl must be > 0") {
		t.Fatalf("expected upload url ttl validation error, got %v", err)
	}
}

func TestRunWithPaths_InvalidDownloadURLTTL(t *testing.T) {
	t.Setenv("MINIO_ENDPOINT", "127.0.0.1:9000")
	t.Setenv("MINIO_ACCESS_KEY", "minio")
	t.Setenv("MINIO_SECRET_KEY", "minio123")
	t.Setenv("MINIO_SESSION_TOKEN", "")
	t.Setenv("MINIO_USE_SSL", "false")
	t.Setenv("MINIO_REGION", "us-east-1")
	t.Setenv("MINIO_HEALTH_CHECK", "false")
	t.Setenv("MINIO_INSECURE_SKIP_TLS", "true")
	t.Setenv("MINIOSERVICE_GRPC_PORT", "50001")
	t.Setenv("MINIOSERVICE_GRPC_JWT_SECRET", "jwt-secret")
	t.Setenv("MINIOSERVICE_GRPC_REFRESH_TOKEN", "refresh-token")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_ACCESS_TOKEN", "300")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_URL_UPLOAD", "300")
	t.Setenv("MINIOSERVICE_GRPC_EXPIRES_TIME_URL_DOWNLOAD", "0")

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "minio_config.yaml")
	envPath := filepath.Join(tmpDir, ".env")

	if err := os.WriteFile(envPath, []byte(""), 0o644); err != nil {
		t.Fatalf("write env file: %v", err)
	}

	cfg := `minio_server:
  endpoint: ${MINIO_ENDPOINT}
  access_key: ${MINIO_ACCESS_KEY}
  secret_key: ${MINIO_SECRET_KEY}
  session_token: ${MINIO_SESSION_TOKEN}
  use_ssl: ${MINIO_USE_SSL}
  region: ${MINIO_REGION}
  health_check: ${MINIO_HEALTH_CHECK}
  insecure_skip_tls: ${MINIO_INSECURE_SKIP_TLS}
  config_service_grpc:
    grpc_port: ${MINIOSERVICE_GRPC_PORT}
    jwt_secret: ${MINIOSERVICE_GRPC_JWT_SECRET}
    refresh_token: ${MINIOSERVICE_GRPC_REFRESH_TOKEN}
    expires_time_access_token: ${MINIOSERVICE_GRPC_EXPIRES_TIME_ACCESS_TOKEN}
    expires_time_url_upload: ${MINIOSERVICE_GRPC_EXPIRES_TIME_URL_UPLOAD}
    expires_time_url_download: ${MINIOSERVICE_GRPC_EXPIRES_TIME_URL_DOWNLOAD}
`
	if err := os.WriteFile(configPath, []byte(cfg), 0o644); err != nil {
		t.Fatalf("write config file: %v", err)
	}

	err := runWithPaths(configPath, envPath)
	if err == nil {
		t.Fatal("expected error for invalid download url ttl, got nil")
	}
	if !strings.Contains(err.Error(), "download url ttl must be > 0") {
		t.Fatalf("expected download url ttl validation error, got %v", err)
	}
}
