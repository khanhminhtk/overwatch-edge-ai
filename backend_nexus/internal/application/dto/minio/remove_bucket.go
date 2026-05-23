package minio

type RemoveBucketRequest struct {
	RequestsID     	string `json:"requests_id"`
	BucketName 		string `json:"bucket_name"`
}

type RemoveBucketResponse struct {
    RequestID string `json:"request_id"`
    Success   bool   `json:"success"`
    Code      string `json:"code,omitempty"`    // CREATED, ALREADY_EXISTS, INVALID_INPUT, INTERNAL_ERROR
    Message   string `json:"message,omitempty"` // human-readable
}