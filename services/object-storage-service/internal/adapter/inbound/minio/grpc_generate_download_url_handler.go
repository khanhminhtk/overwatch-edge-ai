package minio

import (
	"context"

	dtoMinio "backend_nexus/internal/application/dto/minio"
	pb "backend_nexus/proto"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (h *MinioGRPCServerHandler) GenerateDownloadUrl(ctx context.Context, req *pb.GenerateDownloadUrlRequest) (*pb.GenerateDownloadUrlResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if req.GetBucketName() == "" || req.GetObjectName() == "" || req.GetRequestId() == "" {
		return nil, status.Error(codes.InvalidArgument, "bucket_name, object_name, request_id are required")
	}

	resp, err := h.ExportDownloadURL.Execute(ctx, dtoMinio.CreateDownloadURLRequest{
		BucketName: req.GetBucketName(),
		ObjectName: req.GetObjectName(),
		RequestID:  req.GetRequestId(),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate download url: %v", err)
	}

	return &pb.GenerateDownloadUrlResponse{
		Url:    resp.URL,
		Status: resp.Status,
	}, nil
}
