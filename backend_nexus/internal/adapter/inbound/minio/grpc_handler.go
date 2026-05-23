package minio

import (
	"context"
	"fmt"
	"strings"

	usecaseMinio "backend_nexus/internal/application/use_cases/minio"
	"backend_nexus/internal/infra"
	configMinio "backend_nexus/internal/infra/config"
	utils "backend_nexus/internal/utils"
	pb "backend_nexus/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type MinioGRPCServerHandler struct {
	pb.UnimplementedObjectStorageServiceServer
	Minio             infra.MinioObjectStorage
	Logger            utils.Logger
	ConfigServiceGRPC configMinio.MinioServiceGRPC
	ExportDownloadURL *usecaseMinio.ExportDownloadURLUseCase
	ExportUploadURL   *usecaseMinio.ExportUploadURLUseCase
}

func NewMinioGRPCServerHandler(
	minio infra.MinioObjectStorage,
	logger utils.Logger,
	cfg configMinio.MinioServiceGRPC,
	exportUploadURL *usecaseMinio.ExportUploadURLUseCase,
	exportDownloadURL *usecaseMinio.ExportDownloadURLUseCase,
) *MinioGRPCServerHandler {
	return &MinioGRPCServerHandler{
		Minio:             minio,
		Logger:            logger,
		ConfigServiceGRPC: cfg,
		ExportUploadURL:   exportUploadURL,
		ExportDownloadURL: exportDownloadURL,
	}
}

func (h *MinioGRPCServerHandler) Validate() error {
	if h.ExportUploadURL == nil {
		return fmt.Errorf("upload use case is required")
	}
	if h.ExportDownloadURL == nil {
		return fmt.Errorf("download use case is required")
	}
	if strings.TrimSpace(h.ConfigServiceGRPC.JWTSecret) == "" {
		return fmt.Errorf("jwt secret is required")
	}
	if strings.TrimSpace(h.ConfigServiceGRPC.RefreshToken) == "" {
		return fmt.Errorf("static refresh token is required")
	}
	if h.ConfigServiceGRPC.ExpiresTimeAccessToken <= 0 {
		return fmt.Errorf("access token ttl must be > 0")
	}
	return nil
}

func (h *MinioGRPCServerHandler) AuthInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if info.FullMethod == pb.ObjectStorageService_RefreshAccessToken_FullMethodName {
			return handler(ctx, req)
		}
		if strings.HasPrefix(info.FullMethod, "/grpc.health.v1.Health/") {
			return handler(ctx, req)
		}

		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "missing metadata")
		}

		authHeaders := md.Get("authorization")
		if len(authHeaders) == 0 {
			return nil, status.Error(codes.Unauthenticated, "missing authorization header")
		}

		bearer := strings.TrimSpace(authHeaders[0])
		const prefix = "Bearer "
		if !strings.HasPrefix(bearer, prefix) {
			return nil, status.Error(codes.Unauthenticated, "invalid authorization header")
		}

		token := strings.TrimSpace(strings.TrimPrefix(bearer, prefix))
		if _, err := utils.ParseAndValidateAccessToken(token, h.ConfigServiceGRPC.JWTSecret); err != nil {
			return nil, status.Error(codes.Unauthenticated, "invalid or expired access token")
		}

		return handler(ctx, req)
	}
}
