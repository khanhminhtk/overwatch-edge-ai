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

type mockUploadObjectStorage struct {
	uploadURL        string
	uploadErr        error
	createUploadCall int
	lastBucket       string
	lastObject       string
	lastExpires      int64
}

func (m *mockUploadObjectStorage) Upload(ctx context.Context, params dto.UploadParams, bucketName string) error {
	return nil
}

func (m *mockUploadObjectStorage) CreatePresignedURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	return "", nil
}

func (m *mockUploadObjectStorage) CreateBucket(ctx context.Context, bucketName string) error {
	return nil
}

func (m *mockUploadObjectStorage) BucketExists(ctx context.Context, bucketName string) (bool, error) {
	return true, nil
}

func (m *mockUploadObjectStorage) RemoveObject(ctx context.Context, bucketName string, objectName string) error {
	return nil
}

func (m *mockUploadObjectStorage) CreateUploadURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	m.createUploadCall++
	m.lastBucket = bucketName
	m.lastObject = objectName
	m.lastExpires = expiresInSeconds
	if m.uploadErr != nil {
		return "", m.uploadErr
	}
	return m.uploadURL, nil
}

func uploadTestLogger(t *testing.T) utils.Logger {
	t.Helper()
	logger := utils.NewLogger(utils.ConfigLogger{Env: "local", Level: "debug", Service: "test", Version: "1"})
	if logger == nil {
		t.Fatal("expected logger not nil")
	}
	return *logger
}

func TestExportUploadURLUseCase_Execute_UsesValidCache(t *testing.T) {
	mockStorage := &mockUploadObjectStorage{uploadURL: "https://new.example.com/upload"}
	uc := NewExportUploadURLUseCase(mockStorage, uploadTestLogger(t), 600)
	request := minio.CreateUploadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-1",
	}
	uc.cachesManager.caches[buildUploadCacheKey("bucket-a", "obj.txt")] = UrlCache{
		bucketName: "bucket-a",
		objectName: "obj.txt",
		url:        "https://cached.example.com/upload",
		createdAt:  time.Now().UTC(),
	}

	resp, err := uc.Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if !resp.Status {
		t.Fatalf("expected Status true, got false")
	}
	if resp.URL != "https://cached.example.com/upload" {
		t.Fatalf("expected cached URL, got %s", resp.URL)
	}
	if mockStorage.createUploadCall != 0 {
		t.Fatalf("expected no CreateUploadURL call, got %d", mockStorage.createUploadCall)
	}
}

func TestExportUploadURLUseCase_Execute_CacheExpiredCreatesNewURL(t *testing.T) {
	mockStorage := &mockUploadObjectStorage{uploadURL: "https://new.example.com/upload"}
	uc := NewExportUploadURLUseCase(mockStorage, uploadTestLogger(t), 300)
	request := minio.CreateUploadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-2",
	}
	uc.cachesManager.caches[buildUploadCacheKey("bucket-a", "obj.txt")] = UrlCache{
		bucketName: "bucket-a",
		objectName: "obj.txt",
		url:        "https://old.example.com/upload",
		createdAt:  time.Now().UTC().Add(-10 * time.Minute),
	}

	resp, err := uc.Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if !resp.Status {
		t.Fatalf("expected Status true, got false")
	}
	if resp.URL != "https://new.example.com/upload" {
		t.Fatalf("expected new URL, got %s", resp.URL)
	}
	if mockStorage.createUploadCall != 1 {
		t.Fatalf("expected CreateUploadURL called once, got %d", mockStorage.createUploadCall)
	}
	if mockStorage.lastBucket != "bucket-a" || mockStorage.lastObject != "obj.txt" || mockStorage.lastExpires != 300 {
		t.Fatalf("unexpected call args: bucket=%s object=%s expires=%d", mockStorage.lastBucket, mockStorage.lastObject, mockStorage.lastExpires)
	}
}

func TestExportUploadURLUseCase_Execute_CreateUploadURLError(t *testing.T) {
	mockStorage := &mockUploadObjectStorage{uploadErr: errors.New("minio failed")}
	uc := NewExportUploadURLUseCase(mockStorage, uploadTestLogger(t), 600)
	request := minio.CreateUploadURLRequest{
		BucketName: "bucket-a",
		ObjectName: "obj.txt",
		RequestID:  "req-3",
	}

	resp, err := uc.Execute(context.Background(), request)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "ExportUploadURLUseCase.Execute") {
		t.Fatalf("expected wrapped usecase error, got %v", err)
	}
	if resp.Status {
		t.Fatalf("expected Status false, got true")
	}
	if resp.URL != "" {
		t.Fatalf("expected empty URL, got %s", resp.URL)
	}
	if mockStorage.createUploadCall != 1 {
		t.Fatalf("expected CreateUploadURL called once, got %d", mockStorage.createUploadCall)
	}
}
