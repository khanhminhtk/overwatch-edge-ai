package minio

import (
	"context"
	"testing"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func TestBucketExists_InvalidRequest(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.BucketExists(context.Background(), &pb.BucketExistsRequest{RequestId: "", BucketName: "a"})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetCode() != utils.CodeInvalidArgument {
		t.Fatalf("expected %q, got %q", utils.CodeInvalidArgument, resp.GetCode())
	}
}
