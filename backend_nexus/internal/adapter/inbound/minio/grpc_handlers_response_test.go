package minio

import (
	"context"
	"testing"

	configMinio "backend_nexus/internal/infra/config"
	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func TestMinioGRPCServerHandler_BucketExists_InvalidArgument(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.BucketExists(context.Background(), &pb.BucketExistsRequest{
		RequestId:  "",
		BucketName: "bucket-a",
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetSuccess() {
		t.Fatal("expected success=false")
	}
	if resp.GetCode() != utils.CodeInvalidArgument {
		t.Fatalf("expected INVALID_ARGUMENT, got %q", resp.GetCode())
	}
}

func TestMinioGRPCServerHandler_CreateBucket_InvalidArgument(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.CreateBucket(context.Background(), &pb.CreateBucketRequest{
		RequestId:  "req-1",
		BucketName: "",
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetSuccess() {
		t.Fatal("expected success=false")
	}
	if resp.GetCode() != utils.CodeInvalidArgument {
		t.Fatalf("expected INVALID_ARGUMENT, got %q", resp.GetCode())
	}
}

func TestMinioGRPCServerHandler_RemoveBucket_ClientNotConfigured(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.RemoveBucket(context.Background(), &pb.RemoveBucketRequest{
		RequestId:  "req-1",
		BucketName: "bucket-a",
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetSuccess() {
		t.Fatal("expected success=false")
	}
	if resp.GetCode() != utils.CodeInternalError {
		t.Fatalf("expected %q, got %q", utils.CodeInternalError, resp.GetCode())
	}
}

func TestMinioGRPCServerHandler_RefreshAccessToken(t *testing.T) {
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
	if !resp.GetSuccess() {
		t.Fatalf("expected success=true, got false with code=%q message=%q", resp.GetCode(), resp.GetMessage())
	}
	if resp.GetCode() != utils.CodeOK {
		t.Fatalf("expected %q, got %q", utils.CodeOK, resp.GetCode())
	}
	if resp.GetAccessToken() == "" {
		t.Fatal("expected non-empty access token")
	}
}
