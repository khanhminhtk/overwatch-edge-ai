package minio

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestGenerateDownloadURL_NilRequest(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	_, err := h.GenerateDownloadUrl(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument, got %v", status.Code(err))
	}
}
