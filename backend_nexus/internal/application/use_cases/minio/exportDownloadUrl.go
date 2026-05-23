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

type DownloadURLCache struct {
	bucketName string
	objectName string
	url        string
	createdAt  time.Time
}

type DownloadURLManager struct {
	mu     sync.RWMutex
	caches map[string]DownloadURLCache
}

type ExportDownloadURLUseCase struct {
	ObjectStorage ports.ObjectStorage
	Logger        utils.Logger
	cachesManager *DownloadURLManager
}

func NewExportDownloadURLUseCase(objectStorage ports.ObjectStorage, logger utils.Logger) *ExportDownloadURLUseCase {
	return &ExportDownloadURLUseCase{
		ObjectStorage: objectStorage,
		Logger:        logger,
		cachesManager: &DownloadURLManager{
			caches: make(map[string]DownloadURLCache),
		},
	}
}

func (u *ExportDownloadURLUseCase) Execute(ctx context.Context, request minio.CreateDownloadURLRequest) (minio.CreateDownloadURLResponse, error) {
	cacheKey := request.BucketName + "-" + request.ObjectName
	u.cachesManager.mu.RLock()
	cachedData, exists := u.cachesManager.caches[cacheKey]
	u.cachesManager.mu.RUnlock()

	if exists && cachedData.url != "" {
		if time.Since(cachedData.createdAt) < 9*time.Minute {
			u.Logger.Debug("Found valid cached download URL", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
			return minio.CreateDownloadURLResponse{
				URL:    cachedData.url,
				Status: true,
			}, nil
		}
		u.Logger.Warn("Cached download URL expired, rotating", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
	}

	presignedURL, err := u.ObjectStorage.CreatePresignedURL(ctx, request.BucketName, request.ObjectName, 600)
	if err != nil {
		u.Logger.Error("Failed to create download URL via MinIO SDK", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID, "error", err)
		return minio.CreateDownloadURLResponse{
			URL:    "",
			Status: false,
		}, fmt.Errorf("usecases.ExportDownloadURLUseCase.Execute: %w", err)
	}

	u.cachesManager.mu.Lock()
	u.cachesManager.caches[cacheKey] = DownloadURLCache{
		bucketName: request.BucketName,
		objectName: request.ObjectName,
		url:        presignedURL,
		createdAt:  time.Now().UTC(),
	}
	u.cachesManager.mu.Unlock()

	u.Logger.Info("Generate download URL success", "bucket", request.BucketName, "object", request.ObjectName, "request_id", request.RequestID)
	return minio.CreateDownloadURLResponse{
		URL:    presignedURL,
		Status: true,
	}, nil
}
