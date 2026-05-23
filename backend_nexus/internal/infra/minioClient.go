package infra

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/minio/minio-go"

	"backend_nexus/internal/infra/config"
)

func NewMinioClient(
	config config.MinioClient,
) (*minio.Client, error) {
	endpoint := strings.TrimSpace(config.Endpoint)
	if strings.Contains(endpoint, "://") {
		parsedURL, err := url.Parse(endpoint)
		if err != nil {
			return nil, fmt.Errorf("failed to parse Minio endpoint: %w", err)
		}
		endpoint = parsedURL.Host
	}

	client, err := minio.New(
		endpoint,
		config.AccessKey,
		config.SecretKey,
		config.UseSSL,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Minio client: %w", err)
	}
	return client, nil
}
