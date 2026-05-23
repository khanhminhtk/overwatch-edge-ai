package minio

type CreateDownloadURLResponse struct {
	URL    string `json:"url"`
	Status bool   `json:"status"`
}
