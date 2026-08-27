package main

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	pb "backend_nexus/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"gopkg.in/yaml.v3"
	platformlog "orchestrator/internal/platform/log"
	kafka "orchestrator/internal/platform/messaging/kafka"
)

type config struct {
	EdgeID        string `yaml:"edge_id"`
	ModelType     string `yaml:"model_type"`
	FailDirectory string `yaml:"fail_directory"`
	ArchiveDir    string `yaml:"archive_directory"`
	Bucket        string `yaml:"bucket"`
	ObjectPrefix  string `yaml:"object_prefix"`
	Threshold     int    `yaml:"threshold"`
	ScanSeconds   int    `yaml:"scan_interval_seconds"`
	GRPCTarget    string `yaml:"object_storage_grpc_target"`
	TLS           bool   `yaml:"object_storage_tls"`
	KafkaBrokers  string `yaml:"kafka_bootstrap_servers"`
	KafkaTopic    string `yaml:"kafka_result_topic"`
	KafkaClientID string `yaml:"kafka_client_id_prefix"`
}

type manifest struct {
	RequestID string   `json:"request_id"`
	EdgeID    string   `json:"edge_id"`
	CreatedAt string   `json:"created_at"`
	Files     []string `json:"files"`
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		fail(err)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	ticker := time.NewTicker(time.Duration(cfg.ScanSeconds) * time.Second)
	defer ticker.Stop()
	for {
		if err := uploadAvailableBatch(ctx, cfg); err != nil && ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, "fail-data-uploader:", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func loadConfig() (config, error) {
	path := os.Getenv("FAIL_DATA_UPLOADER_CONFIG")
	if path == "" {
		path = "config/fail-data-uploader.yaml"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return config{}, fmt.Errorf("read config %s: %w", path, err)
	}
	var cfg config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return config{}, fmt.Errorf("parse config: %w", err)
	}
	if value := os.Getenv("EDGE_ID"); value != "" {
		cfg.EdgeID = value
	}
	if value := os.Getenv("KAFKA_BOOTSTRAP_SERVERS"); value != "" {
		cfg.KafkaBrokers = value
	}
	if value := os.Getenv("FAIL_DATA_RESULT_TOPIC"); value != "" {
		cfg.KafkaTopic = value
	}
	if value := os.Getenv("FAIL_DATA_THRESHOLD"); value != "" {
		_, err := fmt.Sscan(value, &cfg.Threshold)
		if err != nil {
			return config{}, fmt.Errorf("FAIL_DATA_THRESHOLD: %w", err)
		}
	}
	if cfg.EdgeID == "" || cfg.FailDirectory == "" || cfg.ArchiveDir == "" || cfg.Bucket == "" || cfg.GRPCTarget == "" || cfg.KafkaBrokers == "" || cfg.KafkaTopic == "" {
		return config{}, fmt.Errorf("edge_id, fail_directory, archive_directory, bucket, object_storage_grpc_target, kafka_bootstrap_servers and kafka_result_topic are required")
	}
	if cfg.ModelType != "detection" && cfg.ModelType != "recognizer" {
		return config{}, fmt.Errorf("model_type must be detection or recognizer")
	}
	if cfg.Threshold <= 0 {
		return config{}, fmt.Errorf("threshold must be positive")
	}
	if cfg.ScanSeconds <= 0 {
		cfg.ScanSeconds = 10
	}
	if cfg.ObjectPrefix == "" {
		cfg.ObjectPrefix = "edge-fail"
	}
	if cfg.KafkaClientID == "" {
		cfg.KafkaClientID = "fail-data-uploader"
	}
	return cfg, nil
}

func uploadAvailableBatch(ctx context.Context, cfg config) error {
	files, err := pendingFiles(cfg.FailDirectory)
	if err != nil {
		return err
	}
	if len(files) < cfg.Threshold {
		return nil
	}
	files = files[:cfg.Threshold]
	requestID := fmt.Sprintf("edge-%s-%d", cfg.EdgeID, time.Now().UTC().UnixNano())
	if err := os.MkdirAll(cfg.ArchiveDir, 0o755); err != nil {
		return err
	}
	archivePath := filepath.Join(cfg.ArchiveDir, requestID+".zip")
	checksum, err := createArchive(archivePath, requestID, cfg.EdgeID, files)
	if err != nil {
		return err
	}
	objectName := fmt.Sprintf("%s/%s/%s.zip", strings.Trim(cfg.ObjectPrefix, "/"), cfg.EdgeID, requestID)
	url, err := presignedUploadURL(ctx, cfg, requestID, objectName)
	if err != nil {
		return err
	}
	if err := putArchive(ctx, url, archivePath); err != nil {
		return err
	}
	// Publish before moving input files. If Kafka is unavailable, the original
	// frames remain eligible for retry instead of being silently stranded.
	if err := publishBatchUploaded(ctx, cfg, requestID, objectName, checksum, len(files)); err != nil {
		return err
	}
	// Moving files is deliberately last: an unsuccessful PUT leaves the input
	// batch intact, so a later scan can safely retry it.
	doneDir := filepath.Join(cfg.FailDirectory, "uploaded", requestID)
	if err := os.MkdirAll(doneDir, 0o755); err != nil {
		return err
	}
	for _, file := range files {
		if err := os.Rename(file, filepath.Join(doneDir, filepath.Base(file))); err != nil {
			return err
		}
	}
	fmt.Printf("edge_fail_batch_uploaded request_id=%s edge_id=%s bucket=%s object=%s checksum=%s files=%d\n", requestID, cfg.EdgeID, cfg.Bucket, objectName, checksum, len(files))
	return nil
}

func pendingFiles(directory string) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() && isImage(entry.Name()) {
			files = append(files, filepath.Join(directory, entry.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}

func isImage(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jpg", ".jpeg", ".png", ".bmp", ".webp":
		return true
	}
	return false
}

func createArchive(path, requestID, edgeID string, files []string) (string, error) {
	file, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	writer := io.MultiWriter(file, hash)
	zipWriter := zip.NewWriter(writer)
	names := make([]string, 0, len(files))
	for _, source := range files {
		name := filepath.Base(source)
		names = append(names, name)
		if err := addFile(zipWriter, source, name); err != nil {
			return "", err
		}
	}
	encoded, err := json.Marshal(manifest{RequestID: requestID, EdgeID: edgeID, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano), Files: names})
	if err != nil {
		return "", err
	}
	entry, err := zipWriter.Create("manifest.json")
	if err != nil {
		return "", err
	}
	if _, err = entry.Write(encoded); err != nil {
		return "", err
	}
	if err := zipWriter.Close(); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func addFile(writer *zip.Writer, source, name string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	entry, err := writer.Create(name)
	if err != nil {
		return err
	}
	_, err = io.Copy(entry, input)
	return err
}

func presignedUploadURL(ctx context.Context, cfg config, requestID, objectName string) (string, error) {
	options := []grpc.DialOption{}
	if cfg.TLS {
		options = append(options, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})))
	} else {
		options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	conn, err := grpc.DialContext(ctx, cfg.GRPCTarget, options...)
	if err != nil {
		return "", err
	}
	defer conn.Close()
	response, err := pb.NewObjectStorageServiceClient(conn).GenerateUploadUrl(ctx, &pb.GenerateUploadUrlRequest{RequestId: requestID, BucketName: cfg.Bucket, ObjectName: objectName})
	if err != nil {
		return "", err
	}
	if !response.Status || response.Url == "" {
		return "", fmt.Errorf("object storage returned no upload URL")
	}
	return response.Url, nil
}

func putArchive(ctx context.Context, url, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	request, err := http.NewRequestWithContext(ctx, http.MethodPut, url, file)
	if err != nil {
		return err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("presigned upload returned %s", response.Status)
	}
	return nil
}
func publishBatchUploaded(ctx context.Context, cfg config, requestID, objectName, checksum string, fileCount int) error {
	logger := platformlog.New("fail-data-uploader")
	producer, err := kafka.NewProducer(kafka.Config{BootstrapServers: cfg.KafkaBrokers, SecurityProtocol: "PLAINTEXT", ClientIDPrefix: cfg.KafkaClientID, Defaults: kafka.DefaultsConfig{Producer: kafka.DefaultProducerConfig()}}, logger)
	if err != nil {
		return err
	}
	defer producer.Close()
	payload, err := json.Marshal(map[string]any{"request_id": requestID, "event_type": "edge_fail_batch_uploaded", "payload": map[string]any{"edge_id": cfg.EdgeID, "model_type": cfg.ModelType, "bucket": cfg.Bucket, "object_name": objectName, "checksum_sha256": checksum, "file_count": fileCount, "uploaded_at": time.Now().UTC().Format(time.RFC3339Nano)}})
	if err != nil {
		return err
	}
	return producer.Publish(ctx, kafka.Message{Topic: cfg.KafkaTopic, Partition: 0, Offset: 0, Key: []byte(requestID), Value: payload, Timestamp: time.Now()})
}
func fail(err error) { fmt.Fprintln(os.Stderr, "fail-data-uploader:", err); os.Exit(1) }
