package minio

import (
	"context"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func (h *MinioGRPCServerHandler) RemoveBucket(ctx context.Context, req *pb.RemoveBucketRequest) (*pb.RemoveBucketResponse, error) {
	resp := &pb.RemoveBucketResponse{}
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
			h.Logger.Info("RemoveBucket: request canceled by client")
		}
		resp.Success = false
		resp.Code = utils.CodeRequestCanceled
		resp.Message = "request canceled by client"
		return resp, nil
	}

	if h.Minio.Client == nil {
		resp.Success = false
		resp.Code = utils.CodeInternalError
		resp.Message = "minio client is not configured"
		return resp, nil
	}

	if err := h.Minio.Client.RemoveBucket(req.GetBucketName()); err != nil {
		if h.Logger.Logger != nil {
			h.Logger.Error("RemoveBucket: error occurred while removing bucket", "error", err)
		}
		resp.Success = false
		resp.Code = utils.CodeInternalError
		resp.Message = "error occurred while removing bucket"
		return resp, nil
	}

	resp.Success = true
	resp.Code = utils.CodeOK
	resp.Message = "bucket removed"
	return resp, nil
}
