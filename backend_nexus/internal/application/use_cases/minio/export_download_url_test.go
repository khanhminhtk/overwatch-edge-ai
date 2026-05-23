package usecases

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"backend_nexus/internal/application/dto"
	"backend_nexus/internal/application/dto/minio"
	"backend_nexus/internal/utils"
)

type mockObjectStorage struct {
	presignedURL        string
	presignedErr        error
	createPresignedCall int
	lastBucket          string
	lastObject          string
	lastExpires         int64
}

func (m *mockObjectStorage) Upload(ctx context.Context, params dto.UploadParams, bucketName string) error {
	return nil
}

func (m *mockObjectStorage) CreatePresignedURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	m.createPresignedCall++
	m.lastBucket = bucketName
	m.lastObject = objectName
	m.lastExpires = expiresInSeconds
	if m.presignedErr != nil {
		return "", m.presignedErr
	}
	return m.presignedURL, nil
}

func (m *mockObjectStorage) CreateBucket(ctx context.Context, bucketName string) error {
	return nil
}

func (m *mockObjectStorage) BucketExists(ctx context.Context, bucketName string) (bool, error) {
	return true, nil
}

func (m *mockObjectStorage) RemoveObject(ctx context.Context, bucketName string, objectName string) error {
	return nil
}

func (m *mockObjectStorage) CreateUploadURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	return "", nil
}

func testLogger(t *testing.T) utils.Logger {
	t.Helper()
	logger := utils.NewLogger(utils.ConfigLogger{Env: "local", Level: "debug", Service: "test", Version: "1"})
	if logger == nil {
		t.Fatal("expected logger not nil")
	}
	return *logger
}

func TestExportDownloadURLUseCase_Execute_UsesValidCache(t *testing.T) {
	mockStorage := &mockObjectStorage{presignedURL: "https://new.example.com/url"}
	uc := NewExportDownloadURLUseCase(mockStorage, testLogger(t), 600)
	request := minio.CreateDownloadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-1",
	}
	uc.cachesManager.caches[buildDownloadCacheKey("bucket-a", "obj.txt")] = DownloadURLCache{
		bucketName: "bucket-a",
		objectName: "obj.txt",
		url:        "https://cached.example.com/url",
		createdAt:  time.Now().UTC(),
	}

	resp, err := uc.Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if !resp.Status {
		t.Fatalf("expected Status true, got false")
	}
	if resp.URL != "https://cached.example.com/url" {
		t.Fatalf("expected cached URL, got %s", resp.URL)
	}
	if mockStorage.createPresignedCall != 0 {
		t.Fatalf("expected no CreatePresignedURL call, got %d", mockStorage.createPresignedCall)
	}
}

func TestExportDownloadURLUseCase_Execute_CacheExpiredCreatesNewURL(t *testing.T) {
	mockStorage := &mockObjectStorage{presignedURL: "https://new.example.com/url"}
	uc := NewExportDownloadURLUseCase(mockStorage, testLogger(t), 300)
	request := minio.CreateDownloadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-2",
	}
	uc.cachesManager.caches[buildDownloadCacheKey("bucket-a", "obj.txt")] = DownloadURLCache{
		bucketName: "bucket-a",
		objectName: "obj.txt",
		url:        "https://old.example.com/url",
		createdAt:  time.Now().UTC().Add(-10 * time.Minute),
	}

	resp, err := uc.Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if !resp.Status {
		t.Fatalf("expected Status true, got false")
	}
	if resp.URL != "https://new.example.com/url" {
		t.Fatalf("expected new URL, got %s", resp.URL)
	}
	if mockStorage.createPresignedCall != 1 {
		t.Fatalf("expected CreatePresignedURL called once, got %d", mockStorage.createPresignedCall)
	}
	if mockStorage.lastBucket != "bucket-a" || mockStorage.lastObject != "obj.txt" || mockStorage.lastExpires != 300 {
		t.Fatalf("unexpected call args: bucket=%s object=%s expires=%d", mockStorage.lastBucket, mockStorage.lastObject, mockStorage.lastExpires)
	}
}

func TestExportDownloadURLUseCase_Execute_CreatePresignedURLError(t *testing.T) {
	mockStorage := &mockObjectStorage{presignedErr: errors.New("minio failed")}
	uc := NewExportDownloadURLUseCase(mockStorage, testLogger(t), 600)
	request := minio.CreateDownloadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-3",
	}

	resp, err := uc.Execute(context.Background(), request)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "ExportDownloadURLUseCase.Execute") {
		t.Fatalf("expected wrapped usecase error, got %v", err)
	}
	if resp.Status {
		t.Fatalf("expected Status false, got true")
	}
	if resp.URL != "" {
		t.Fatalf("expected empty URL, got %s", resp.URL)
	}
	if mockStorage.createPresignedCall != 1 {
		t.Fatalf("expected CreatePresignedURL called once, got %d", mockStorage.createPresignedCall)
	}
}
