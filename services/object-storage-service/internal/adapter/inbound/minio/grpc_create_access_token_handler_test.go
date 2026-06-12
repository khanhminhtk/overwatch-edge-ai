package minio

import (
	"context"
	"testing"

	configMinio "backend_nexus/internal/infra/config"
	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func TestRefreshAccessToken_ValidToken(t *testing.T) {
	h := &MinioGRPCServerHandler{
		ConfigServiceGRPC: configMinio.MinioServiceGRPC{
			JWTSecret:              "secret",
			RefreshToken:           "refresh",
			ExpiresTimeAccessToken: 900,
		},
	}

	resp, err := h.RefreshAccessToken(context.Background(), &pb.RefreshAccessTokenRequest{
		RequestId:    "req-1",
		RefreshToken: "refresh",
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !resp.GetSuccess() || resp.GetCode() != utils.CodeOK || resp.GetAccessToken() == "" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}
