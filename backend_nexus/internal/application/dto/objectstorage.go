package dto

type UploadParams struct {
	Data [] byte
	ObjectName string
	UserMetadata map[string]string
	ContentType string
} 