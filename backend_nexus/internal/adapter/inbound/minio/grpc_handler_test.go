package minio

import (
	"context"
	"testing"

	pb "backend_nexus/proto"

	"google.golang.org/grpc"
)

func TestAuthInterceptor_BypassRefreshAccessToken(t *testing.T) {
	h := &MinioGRPCServerHandler{}
	interceptor := h.AuthInterceptor()

	called := false
	_, err := interceptor(
		context.Background(),
		nil,
		&grpc.UnaryServerInfo{FullMethod: pb.ObjectStorageService_RefreshAccessToken_FullMethodName},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			called = true
			return "ok", nil
		},
	)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !called {
		t.Fatal("expected handler to be called")
	}
}
