package infra

import (
	"bytes"
	"context"
	"fmt"
	"net/url"
	"time"

	"github.com/minio/minio-go"

	"backend_nexus/internal/application/dto"
	"backend_nexus/internal/application/ports"
	"backend_nexus/internal/utils"
)

var _ ports.ObjectStorage = (*MinioObjectStorage)(nil)

type MinioObjectStorage struct {
	Client *minio.Client
	Logger *utils.Logger
}

func NewMinioObjectStorage(client *minio.Client, logger *utils.Logger) *MinioObjectStorage {
	return &MinioObjectStorage{
		Client: client,
		Logger: logger,
	}
}

func (m *MinioObjectStorage) wrapError(format string, err error, args ...any) error {
	wrappedErr := fmt.Errorf(format+": %w", append(args, err)...)
	if m.Logger != nil {
		m.Logger.Error("MinioObjectStorage operation failed", "error", wrappedErr)
	}
	return wrappedErr
}

func (m *MinioObjectStorage) CreateBucket(ctx context.Context, bucketName string) error {
	if m.Logger != nil {
		m.Logger.Info("CreateBucket start", "bucket", bucketName)
	}

	if err := ctx.Err(); err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.CreateBucket: context error", err)
	}

	exists, err := m.Client.BucketExists(bucketName)
	if err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.CreateBucket: failed to check if bucket exists", err)
	}
	if !exists {
		if m.Logger != nil {
			m.Logger.Debug("CreateBucket creating missing bucket", "bucket", bucketName)
		}
		err = m.Client.MakeBucket(bucketName, "")
		if err != nil {
			return m.wrapError("internal.infra.minioObjectStorage.CreateBucket: failed to create bucket", err)
		}
		if m.Logger != nil {
			m.Logger.Info("CreateBucket success", "bucket", bucketName, "created", true)
		}
		return nil
	}
	if m.Logger != nil {
		m.Logger.Debug("CreateBucket bucket already exists", "bucket", bucketName)
		m.Logger.Info("CreateBucket success", "bucket", bucketName, "created", false)
	}
	return nil
}

func (m *MinioObjectStorage) BucketExists(ctx context.Context, bucketName string) (bool, error) {
	if m.Logger != nil {
		m.Logger.Debug("BucketExists check", "bucket", bucketName)
	}

	if err := ctx.Err(); err != nil {
		return false, m.wrapError("internal.infra.minioObjectStorage.BucketExists: context error", err)
	}

	exists, err := m.Client.BucketExists(bucketName)
	if err != nil {
		return false, m.wrapError("internal.infra.minioObjectStorage.BucketExists: failed to check if bucket exists", err)
	}
	if m.Logger != nil {
		m.Logger.Debug("BucketExists result", "bucket", bucketName, "exists", exists)
	}
	return exists, nil
}

func (m *MinioObjectStorage) Upload(ctx context.Context, params dto.UploadParams, bucketName string) error {
	if m.Logger != nil {
		m.Logger.Info("Upload start", "bucket", bucketName, "object", params.ObjectName, "size_bytes", len(params.Data))
	}

	if err := ctx.Err(); err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.Upload: context error", err)
	}

	_, err := m.Client.PutObject(bucketName, params.ObjectName, bytes.NewReader(params.Data), int64(len(params.Data)),
		minio.PutObjectOptions{
			ContentType:  params.ContentType,
			UserMetadata: params.UserMetadata,
		},
	)
	if err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.Upload: failed to upload object", err)
	}
	if m.Logger != nil {
		m.Logger.Info("Upload success", "bucket", bucketName, "object", params.ObjectName)
	}
	return nil
}

func (m *MinioObjectStorage) CreatePresignedURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	if m.Logger != nil {
		m.Logger.Info("CreatePresignedURL start", "bucket", bucketName, "object", objectName, "expires_seconds", expiresInSeconds)
	}

	if err := ctx.Err(); err != nil {
		return "", m.wrapError("internal.infra.minioObjectStorage.CreatePresignedURL: context error", err)
	}

	reqParams := make(url.Values)
	presignedURL, err := m.Client.PresignedGetObject(
		bucketName,
		objectName,
		time.Duration(expiresInSeconds)*time.Second,
		reqParams,
	)
	if err != nil {
		return "", m.wrapError("internal.infra.minioObjectStorage.CreatePresignedURL: failed to create presigned URL", err)
	}
	if m.Logger != nil {
		m.Logger.Debug("CreatePresignedURL success", "bucket", bucketName, "object", objectName)
	}
	return presignedURL.String(), nil
}

func (m *MinioObjectStorage) RemoveObject(ctx context.Context, bucketName string, objectName string) error {
	if m.Logger != nil {
		m.Logger.Info("RemoveObject start", "bucket", bucketName, "object", objectName)
	}

	if err := ctx.Err(); err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.RemoveObject: context error", err)
	}

	err := m.Client.RemoveObject(bucketName, objectName)
	if err != nil {
		return m.wrapError("internal.infra.minioObjectStorage.RemoveObject: failed to remove object", err)
	}
	if m.Logger != nil {
		m.Logger.Info("RemoveObject success", "bucket", bucketName, "object", objectName)
	}
	return nil
}

func (m *MinioObjectStorage) CreateUploadURL(ctx context.Context, bucketName string, objectName string, expiresInSeconds int64) (string, error) {
	if m.Logger != nil {
		m.Logger.Info("CreateUploadURL start", "bucket", bucketName, "object", objectName, "expires_seconds", expiresInSeconds)
	}

	if err := ctx.Err(); err != nil {
		return "", m.wrapError("internal.infra.minioObjectStorage.CreateUploadURL: context error", err)
	}
	expiry := time.Duration(10) * time.Minute

	presignedUrl, err := m.Client.PresignedPutObject(
		bucketName, 
		objectName, 
		expiry,
	)
	if err != nil {
		return "", m.wrapError("internal.infra.minioObjectStorage.CreateUploadURL: failed to create presigned URL", err)
	}
	return presignedUrl.String(), nil
}