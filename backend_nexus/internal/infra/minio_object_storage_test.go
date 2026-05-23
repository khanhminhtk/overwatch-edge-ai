package infra

import (
	"context"
	"strings"
	"testing"

	"github.com/minio/minio-go"

	"backend_nexus/internal/application/dto"
	"backend_nexus/internal/utils"
)

func newUnreachableMinioClient(t *testing.T) *minio.Client {
	t.Helper()
	client, err := minio.New("127.0.0.1:1", "minio", "minio123", false)
	if err != nil {
		t.Fatalf("minio.New() error = %v", err)
	}
	return client
}

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func TestNewMinioObjectStorage(t *testing.T) {
	client := newUnreachableMinioClient(t)
	logger := utils.NewLogger(utils.ConfigLogger{Env: "local", Level: "info", Service: "test", Version: "1"})
	if logger == nil {
		t.Fatal("expected logger not nil")
	}

	storage := NewMinioObjectStorage(client, logger)
	if storage == nil {
		t.Fatal("expected storage not nil")
	}
	if storage.Client == nil {
		t.Fatal("expected storage client not nil")
	}
}

func TestCreateBucket_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	err := storage.CreateBucket(canceledContext(), "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestCreateBucket_ErrorFromClient(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	err := storage.CreateBucket(context.Background(), "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "failed to check if bucket exists") {
		t.Fatalf("expected wrapped client error, got %v", err)
	}
}

func TestBucketExists_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	_, err := storage.BucketExists(canceledContext(), "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestBucketExists_ErrorFromClient(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	_, err := storage.BucketExists(context.Background(), "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "failed to check if bucket exists") {
		t.Fatalf("expected wrapped client error, got %v", err)
	}
}

func TestUpload_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	err := storage.Upload(canceledContext(), dto.UploadParams{}, "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestUpload_ErrorFromClient(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	params := dto.UploadParams{ObjectName: "obj.txt", Data: []byte("hello")}
	err := storage.Upload(context.Background(), params, "bucket-a")
	if err == nil || !strings.Contains(err.Error(), "failed to upload object") {
		t.Fatalf("expected wrapped client error, got %v", err)
	}
}

func TestCreatePresignedURL_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	_, err := storage.CreatePresignedURL(canceledContext(), "bucket-a", "obj.txt", 60)
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestCreatePresignedURL_ErrorFromClient(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	_, err := storage.CreatePresignedURL(context.Background(), "bucket-a", "obj.txt", 60)
	if err == nil || !strings.Contains(err.Error(), "failed to create presigned URL") {
		t.Fatalf("expected wrapped client error, got %v", err)
	}
}

func TestCreatePresignedURL_InvalidExpires(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	_, err := storage.CreatePresignedURL(context.Background(), "bucket-a", "obj.txt", -1)
	if err == nil || !strings.Contains(err.Error(), "invalid expiresInSeconds") {
		t.Fatalf("expected presigned url error for invalid expiration, got %v", err)
	}
}

func TestRemoveObject_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	err := storage.RemoveObject(canceledContext(), "bucket-a", "obj.txt")
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestRemoveObject_ErrorFromClient(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	err := storage.RemoveObject(context.Background(), "bucket-a", "obj.txt")
	if err == nil || !strings.Contains(err.Error(), "failed to remove object") {
		t.Fatalf("expected wrapped client error, got %v", err)
	}
}

func TestCreateUploadURL_ContextCanceled(t *testing.T) {
	storage := &MinioObjectStorage{}
	_, err := storage.CreateUploadURL(canceledContext(), "bucket-a", "obj.txt", 60)
	if err == nil || !strings.Contains(err.Error(), "context error") {
		t.Fatalf("expected context error, got %v", err)
	}
}

func TestCreateUploadURL_InvalidExpires(t *testing.T) {
	storage := NewMinioObjectStorage(newUnreachableMinioClient(t), nil)
	_, err := storage.CreateUploadURL(context.Background(), "bucket-a", "obj.txt", 0)
	if err == nil || !strings.Contains(err.Error(), "invalid expiresInSeconds") {
		t.Fatalf("expected invalid expires error, got %v", err)
	}
}
