package minio

import (
	"context"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func (h *MinioGRPCServerHandler) BucketExists(ctx context.Context, req *pb.BucketExistsRequest) (*pb.BucketExistsResponse, error) {
	resp := &pb.BucketExistsResponse{}

	if req != nil {
		resp.RequestId = req.GetRequestId()
	}

	if req == nil || req.GetRequestId() == "" || req.GetBucketName() == "" {
		resp.Success = false
		resp.Code = utils.CodeInvalidArgument
		resp.Message = "request_id and bucket_name are required"
		return resp, nil
	}

	if ctx.Err() == context.Canceled {
		if h.Logger.Logger != nil {
			h.Logger.Info("BucketExists: request canceled by client")
		}
		resp.Success = false
		resp.Code = utils.CodeRequestCanceled
		resp.Message = "request canceled by client"
		return resp, nil
	}

	bucketName := req.GetBucketName()

	exists, err := h.Minio.BucketExists(ctx, bucketName)
	if err != nil {
		if h.Logger.Logger != nil {
			h.Logger.Error("BucketExists: error occurred while checking bucket existence", "error", err)
		}
		resp.Success = false
		resp.Code = utils.CodeInternalError
		resp.Message = "error occurred while checking bucket existence"
		return resp, nil
	}

	resp.Success = exists
	resp.Code = utils.CodeOK
	resp.Message = "bucket existence checked"
	return resp, nil
}
