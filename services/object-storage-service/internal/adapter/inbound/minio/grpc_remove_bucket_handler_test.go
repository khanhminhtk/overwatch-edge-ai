package minio

import (
	"context"
	"testing"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func TestRemoveBucket_NoClientConfigured(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.RemoveBucket(context.Background(), &pb.RemoveBucketRequest{RequestId: "req-1", BucketName: "bucket-a"})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetCode() != utils.CodeInternalError {
		t.Fatalf("expected %q, got %q", utils.CodeInternalError, resp.GetCode())
	}
}
