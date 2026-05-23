package usecases

import (
	"context"
	"fmt"
	"sync"
	"time"

	"backend_nexus/internal/application/dto/minio"
	"backend_nexus/internal/application/ports"
	"backend_nexus/internal/utils"
)

type UrlCache struct {
	bucketName string
	objectName string
	url        string
	createdAt  time.Time
} 

type UploadURLManager struct {
	mu     sync.RWMutex
	caches map[string]UrlCache
}

type ExportUploadURLUseCase struct {
	ObjectStorage ports.ObjectStorage
	Logger        utils.Logger
	cachesManager *UploadURLManager
}

func NewExportUploadURLUseCase(objectStorage ports.ObjectStorage, logger utils.Logger) *ExportUploadURLUseCase {
	return &ExportUploadURLUseCase{
		ObjectStorage: objectStorage,
		Logger:        logger,
		cachesManager: &UploadURLManager{
			caches: make(map[string]UrlCache),
		},
	}
}

func (u *ExportUploadURLUseCase) Execute(ctx context.Context, request minio.CreateUploadURLRequest) (minio.CreateUploadURLResponse, error) {
	cacheKey := request.BucketName + "-" + request.ObjectName
	u.cachesManager.mu.RLock()
	cachedData, exists := u.cachesManager.caches[cacheKey]
	u.cachesManager.mu.RUnlock()

	if exists && cachedData.url != "" {
		if time.Since(cachedData.createdAt) < 9*time.Minute { 
			u.Logger.Debug("Found valid cached upload URL", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
			return minio.CreateUploadURLResponse{
				URL:    cachedData.url,
				Status: true,
			}, nil
		}
		u.Logger.Warn("Cached upload URL expired, rotating", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
	}

	presignedURL, err := u.ObjectStorage.CreateUploadURL(ctx, request.BucketName, request.ObjectName, 600)
	if err != nil {
		u.Logger.Error("Failed to create upload URL via MinIO SDK", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID, "error", err)
		return minio.CreateUploadURLResponse{
			URL:    "",
			Status: false,
		}, fmt.Errorf("usecases.ExportUploadURLUseCase.Execute: %w", err)
	}
	u.cachesManager.mu.Lock()
	u.cachesManager.caches[cacheKey] = UrlCache{
		bucketName: request.BucketName,
		objectName: request.ObjectName,
		url:        presignedURL,
		createdAt:  time.Now().UTC(),
	}
	u.cachesManager.mu.Unlock()

	u.Logger.Info("Generate upload URL success", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
	return minio.CreateUploadURLResponse{
		URL:    presignedURL,
		Status: true,
	}, nil
}