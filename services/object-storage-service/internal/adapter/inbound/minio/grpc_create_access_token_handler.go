package minio

import (
	"context"
	"strings"

	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"
)

func validateRequest(req *pb.RefreshAccessTokenRequest, resp *pb.RefreshAccessTokenResponse) bool {
	if resp == nil {
		return false
	}
	if req == nil || strings.TrimSpace(req.GetRequestId()) == "" {
		resp.RequestId = ""
		resp.Success = false
		resp.AccessToken = ""
		resp.ExpiresIn = 0
		resp.Code = utils.CodeInvalidArgument
		resp.Message = "request_id is required"
		return false
	}

	if strings.TrimSpace(req.GetRefreshToken()) == "" {
		resp.RequestId = req.GetRequestId()
		resp.Success = false
		resp.AccessToken = ""
		resp.ExpiresIn = 0
		resp.Code = utils.CodeInvalidArgument
		resp.Message = "refresh_token is required"
		return false
	}

	return true
}

func (h *MinioGRPCServerHandler) RefreshAccessToken(ctx context.Context, req *pb.RefreshAccessTokenRequest) (*pb.RefreshAccessTokenResponse, error) {
	resp := &pb.RefreshAccessTokenResponse{}
	if req != nil {
		resp.RequestId = req.GetRequestId()
	}

	if !validateRequest(req, resp) {
		return resp, nil
	}

	if ctx.Err() == context.Canceled {
		if h.Logger.Logger != nil {
			h.Logger.Info("RefreshAccessToken: request canceled by client")
		}
		resp.Success = false
		resp.AccessToken = ""
		resp.ExpiresIn = 0
		resp.Code = utils.CodeRequestCanceled
		resp.Message = "request canceled by client"
		return resp, nil
	}

	if err := utils.ValidateStaticRefreshToken(req.GetRefreshToken(), h.ConfigServiceGRPC.RefreshToken); err != nil {
		resp.Success = false
		resp.Code = utils.CodeInvalidRefreshToken
		resp.Message = "refresh token is invalid"
		resp.AccessToken = ""
		resp.ExpiresIn = 0
		return resp, nil
	}

	accessToken, expiresIn, err := utils.GenerateAccessToken(
		h.ConfigServiceGRPC.JWTSecret,
		h.ConfigServiceGRPC.ExpiresTimeAccessToken,
		"minio_service",
	)
	if err != nil {
		resp.Success = false
		resp.Code = utils.CodeInternalError
		resp.Message = "failed to generate access token"
		resp.AccessToken = ""
		resp.ExpiresIn = 0
		return resp, nil
	}

	resp.Success = true
	resp.Code = utils.CodeOK
	resp.Message = "access token generated"
	resp.AccessToken = accessToken
	resp.ExpiresIn = expiresIn
	return resp, nil
}
