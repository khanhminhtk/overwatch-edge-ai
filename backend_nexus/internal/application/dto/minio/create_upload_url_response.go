package minio

type CreateUploadURLResponse struct {
	URL    string `json:"url"`
	Status bool   `json:"status"`
}
