package minio

import (
	"context"
	"testing"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func TestCreateBucket_InvalidRequest(t *testing.T) {
	h := &MinioGRPCServerHandler{}

	resp, err := h.CreateBucket(context.Background(), &pb.CreateBucketRequest{RequestId: "req-1", BucketName: ""})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if resp.GetCode() != utils.CodeInvalidArgument {
		t.Fatalf("expected %q, got %q", utils.CodeInvalidArgument, resp.GetCode())
	}
}
