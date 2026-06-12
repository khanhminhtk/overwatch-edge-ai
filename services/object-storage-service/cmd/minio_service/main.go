package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	inbound "backend_nexus/internal/adapter/inbound"
	inboundMinio "backend_nexus/internal/adapter/inbound/minio"
	usecasesMinio "backend_nexus/internal/application/use_cases/minio"
	"backend_nexus/internal/infra"
	configInfra "backend_nexus/internal/infra/config"
	"backend_nexus/internal/utils"
	pb "backend_nexus/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	grpcHealth "google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func startMinioService(objectStorage *infra.MinioObjectStorage, cfg *configInfra.Minio, logger *utils.Logger) error {
	uploadURLUseCase := usecasesMinio.NewExportUploadURLUseCase(
		objectStorage,
		*logger,
		cfg.MinioServer.ConfigServiceGRPC.ExpiresTimeURLUpload,
	)
	downloadURLUseCase := usecasesMinio.NewExportDownloadURLUseCase(
		objectStorage,
		*logger,
		cfg.MinioServer.ConfigServiceGRPC.ExpiresTimeURLDownload,
	)
	minioService := inboundMinio.NewMinioGRPCServerHandler(
		*objectStorage,
		*logger,
		cfg.MinioServer.ConfigServiceGRPC,
		uploadURLUseCase,
		downloadURLUseCase,
	)

	if err := minioService.Validate(); err != nil {
		return fmt.Errorf("failed to validate minio service: %w", err)
	}

	grpcPort := strings.TrimSpace(cfg.MinioServer.ConfigServiceGRPC.GRPCPort)
	if grpcPort == "" {
		grpcPort = "50001"
	}
	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", grpcPort))
	if err != nil {
		return fmt.Errorf("failed to listen on port %s: %w", grpcPort, err)
	}
	defer lis.Close()

	serverOptions := []grpc.ServerOption{
		grpc.UnaryInterceptor(minioService.AuthInterceptor()),
	}

	certFile := "certs/server.crt"
	keyFile := "certs/server.key"
	creds, tlsErr := credentials.NewServerTLSFromFile(certFile, keyFile)
	if tlsErr != nil {
		return fmt.Errorf("failed to create TLS credentials: %w", tlsErr)
	}
	serverOptions = append(serverOptions, grpc.Creds(creds))
	grpcServer := grpc.NewServer(serverOptions...)
	healthServer := grpcHealth.NewServer()
	healthpb.RegisterHealthServer(grpcServer, healthServer)
	pb.RegisterObjectStorageServiceServer(grpcServer, minioService)

	healthManager := inbound.NewHealthManager(healthServer, pb.ObjectStorageService_ServiceDesc.ServiceName)
	inboundMinio.SyncHealthStatus(
		context.Background(),
		healthManager,
		objectStorage,
		cfg.MinioServer.Endpoint,
		2*time.Second,
	)
	reflection.Register(grpcServer)
	logger.Info("Minio gRPC service started", "port", grpcPort)

	serveErr := make(chan error, 1)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			serveErr <- err
		}
		close(serveErr)
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(quit)

	select {
	case <-quit:
		logger.Info("Shutting down Minio gRPC service...")
		grpcServer.GracefulStop()
		logger.Info("Minio gRPC service stopped")
		return nil
	case err := <-serveErr:
		if err != nil {
			return fmt.Errorf("failed to serve gRPC: %w", err)
		}
		return nil
	}
}

func runWithPaths(configPath, envPath string) error {
	configLoader := utils.NewConfigLoader[configInfra.Minio](
		configPath,
		envPath,
	)
	cfg, err := configLoader.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	configLogger := utils.ConfigLogger{
		Level:   "debug",
		Env:     "local",
		Service: "minio_service",
		Version: "1",
	}
	logger := utils.NewLogger(configLogger)

	minioClient, err := infra.NewMinioClient(cfg.MinioServer)
	if err != nil {
		return fmt.Errorf("failed to create Minio client: %w", err)
	}

	minioStorage := infra.NewMinioObjectStorage(minioClient, logger)

	if err := startMinioService(minioStorage, cfg, logger); err != nil {
		logger.Error("Minio service stopped with error", "error", err)
		return err
	}
	return nil
}

func run() error {
	return runWithPaths("config/minio_config.yaml", "config/.env")
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
