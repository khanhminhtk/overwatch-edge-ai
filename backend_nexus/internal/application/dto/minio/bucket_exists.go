package minio

type BucketExistsRequest struct {
	RequestsID     	string `json:"requests_id"`
	BucketName 		string `json:"bucket_name"`
}

type BucketExistsResponse struct {
    RequestID string `json:"request_id"`
    Success   bool   `json:"success"`
    Code      string `json:"code,omitempty"`    // CREATED, ALREADY_EXISTS, INVALID_INPUT, INTERNAL_ERROR
    Message   string `json:"message,omitempty"` // human-readable
}