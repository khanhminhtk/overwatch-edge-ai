package minio

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestGenerateUploadURL_NilRequest(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	_, err := h.GenerateUploadUrl(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument, got %v", status.Code(err))
	}
}
