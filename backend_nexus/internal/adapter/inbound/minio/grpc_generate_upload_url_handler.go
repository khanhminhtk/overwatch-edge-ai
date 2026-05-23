package minio

import (
	"context"

	dtoMinio "backend_nexus/internal/application/dto/minio"
	pb "backend_nexus/proto"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (h *MinioGRPCServerHandler) GenerateUploadUrl(ctx context.Context, req *pb.GenerateUploadUrlRequest) (*pb.GenerateUploadUrlResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if req.GetBucketName() == "" || req.GetObjectName() == "" || req.GetRequestId() == "" {
		return nil, status.Error(codes.InvalidArgument, "bucket_name, object_name, request_id are required")
	}

	resp, err := h.ExportUploadURL.Execute(ctx, dtoMinio.CreateUploadURLRequest{
		BucketName: req.GetBucketName(),
		ObjectName: req.GetObjectName(),
		RequestID:  req.GetRequestId(),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate upload url: %v", err)
	}

	return &pb.GenerateUploadUrlResponse{
		Url:    resp.URL,
		Status: resp.Status,
	}, nil
}
