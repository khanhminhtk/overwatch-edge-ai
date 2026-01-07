package ports

import "backend_nexus/internal/application/dto"

type ObjectStorage interface {
	Upload(params dto.UploadParams) (string, error)
}