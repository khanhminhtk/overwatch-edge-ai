package minio

type RefreshAccessTokenRequest struct {
	RequestID    string `json:"request_id"`
	RefreshToken string `json:"refresh_token"`
}

type RefreshAccessTokenResponse struct {
	RequestID   string `json:"request_id"`
	Success     bool   `json:"success"`
	Code        string `json:"code,omitempty"`    // OK, INVALID_REFRESH_TOKEN, EXPIRED_REFRESH_TOKEN, INTERNAL_ERROR
	Message     string `json:"message,omitempty"` // human-readable
	AccessToken string `json:"access_token,omitempty"`
	ExpiresIn   int64  `json:"expires_in,omitempty"`
}
