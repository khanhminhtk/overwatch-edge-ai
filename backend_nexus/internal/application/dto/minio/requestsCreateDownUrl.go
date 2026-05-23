package minio

type CreateDownloadURLRequest struct {
	BucketName string `json:"bucketName" validate:"required"`
	ObjectName string `json:"objectName" validate:"required"`
	RequestID  string `json:"requestId" validate:"required"`
}
