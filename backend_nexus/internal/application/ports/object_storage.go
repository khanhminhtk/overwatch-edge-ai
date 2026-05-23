package ports

import (
	"context"

	"backend_nexus/internal/application/dto"
)

type ObjectStorage interface {
	Upload(ctx context.Context, params dto.UploadParams, bucketName string) error
	CreatePresignedURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error)
	CreateBucket(ctx context.Context, bucketName string) error
	BucketExists(ctx context.Context, bucketName string) (bool, error)
	RemoveObject(ctx context.Context, bucketName string, objectName string) error
	CreateUploadURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error)
}
