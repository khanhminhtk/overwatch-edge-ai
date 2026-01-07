package infra

import (
	"backend_nexus/internal/application/ports"
	"backend_nexus/internal/application/dto"
	"bytes"
	"fmt"

	"github.com/minio/minio-go"
)

var _ ports.ObjectStorage = (*MinioAdapter)(nil)

type MinioAdapter struct {
	client     *minio.Client
	bucketname string
}

func NewMinioAdapter(endpoint string, accessKey string, secretKey string, bucketname string, useSSL bool) (*MinioAdapter, error) {
	miniclient, err := minio.New(
		endpoint,
		accessKey,
		secretKey,
		useSSL,
	)
	if err != nil {
		return nil, fmt.Errorf("không thể kết nối Minio: %w", err)
	}

	return &MinioAdapter{
		client: miniclient,
		bucketname: bucketname,
	}, nil
}

func (m *MinioAdapter) Upload(params dto.UploadParams) (string, error) {
	myReader := bytes.NewReader(params.Data)
	objectSize := int64(len(params.Data))

	options := minio.PutObjectOptions{
		ContentType:  params.ContentType, 
		UserMetadata: params.UserMetadata,
	}

	n, err := m.client.PutObject(
		m.bucketname,
		params.ObjectName, 
		myReader,
		objectSize,
		options,
	)

	if err != nil {
		return "", fmt.Errorf("upload thất bại: %w", err)
	}

	fmt.Printf("Đã upload %d bytes lên bucket %s\n", n, m.bucketname)
	return params.ObjectName, nil
}