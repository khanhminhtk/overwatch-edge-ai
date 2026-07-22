package runner

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

type testLogger struct{}

func (testLogger) Info(string, ...any)  {}
func (testLogger) Error(string, ...any) {}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestRunMoveModelCopiesSelectedArtifacts(t *testing.T) {
	dir := t.TempDir()
	detectorSource := filepath.Join(dir, "detector.onnx")
	recognizerSource := filepath.Join(dir, "recognizer.onnx")
	for _, source := range []string{detectorSource, recognizerSource} {
		if err := os.WriteFile(source, []byte("model"), 0o600); err != nil {
			t.Fatalf("create model artifact: %v", err)
		}
	}

	runner, err := NewRunner(testLogger{}, "edge-1")
	if err != nil {
		t.Fatal(err)
	}
	config := domain.TritonConfig{
		Image:               "triton",
		ModelRepositoryPath: filepath.Join(dir, "repository"),
		IP:                  "127.0.0.1",
		HostPort:            8000,
		Model: domain.TritonModelConfig{
			Detector:   domain.TritonModelSpec{Name: "detector", Version: 1, ModelPathRaw: "detector.pt", ModelPathONNX: detectorSource, ModelPathTRT: "detector.plan"},
			Recognizer: domain.TritonModelSpec{Name: "recognizer", Version: 1, ModelPathRaw: "recognizer.pt", ModelPathONNX: recognizerSource, ModelPathTRT: "recognizer.plan"},
		},
	}

	if ok, err := runner.runMoveModel(context.Background(), config, "onnx", "1"); err != nil || !ok {
		t.Fatalf("run move model = %t, %v", ok, err)
	}
	for model, source := range map[string]string{"detector": detectorSource, "recognizer": recognizerSource} {
		if _, err := os.Stat(source); err != nil {
			t.Fatalf("source %q was not preserved: %v", source, err)
		}
		if _, err := os.Stat(filepath.Join(config.ModelRepositoryPath, model, "1", filepath.Base(source))); err != nil {
			t.Fatalf("moved artifact for %q: %v", model, err)
		}
	}
}

func TestRunMoveModelUsesDefaultModelFilename(t *testing.T) {
	dir := t.TempDir()
	repository := filepath.Join(dir, "repository")
	source := filepath.Join(dir, "recognizer.onnx")
	if err := os.WriteFile(source, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(repository, "recognizer", "config.pbtxt")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, []byte("default_model_filename: \"recognizer.best_cer.onnx\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	runner, err := NewRunner(testLogger{}, "edge-1")
	if err != nil {
		t.Fatal(err)
	}
	config := testTritonConfig(repository, "127.0.0.1", 8000)
	config.Model.Detector.ModelPathONNX = filepath.Join(dir, "detector.onnx")
	config.Model.Recognizer.ModelPathONNX = source
	if err := os.WriteFile(config.Model.Detector.ModelPathONNX, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if ok, err := runner.runMoveModel(context.Background(), config, "onnx", "2"); err != nil || !ok {
		t.Fatalf("run move model = %t, %v", ok, err)
	}
	if _, err := os.Stat(filepath.Join(repository, "recognizer", "2", "recognizer.best_cer.onnx")); err != nil {
		t.Fatalf("renamed recognizer artifact: %v", err)
	}
}

func TestUpdateVersionPolicyAndUnloadBacksUpConfigAndCallsTriton(t *testing.T) {
	var requests []string
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests = append(requests, request.Method+" "+request.URL.Path)
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(strings.NewReader("")), Request: request}, nil
	})}

	repository := filepath.Join(t.TempDir(), "repository")
	configPath := filepath.Join(repository, "detector", "config.pbtxt")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "name: \"detector\"\nversion_policy {\n  specific {\n    versions: [1]\n  }\n}\n"
	if err := os.WriteFile(configPath, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}

	runner, err := NewRunner(testLogger{}, "edge-1")
	if err != nil {
		t.Fatal(err)
	}
	runner.httpClient = client
	config := testTritonConfig(repository, "localhost", 8000)
	if err := runner.UpdateVersionPolicyAndUnload(context.Background(), config, "detector", "2"); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(configPath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("config backup was not removed: %v", err)
	}
	updated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	if !strings.Contains(string(updated), "versions: [2]") || strings.Contains(string(updated), "versions: [1]") {
		t.Fatalf("updated version policy = %q", updated)
	}
	wantRequests := []string{
		"POST /v2/repository/models/detector/unload",
		"POST /v2/repository/models/detector/load",
		"GET /v2/models/detector/versions/2/ready",
		"GET /v2/health/ready",
	}
	if strings.Join(requests, "\n") != strings.Join(wantRequests, "\n") {
		t.Fatalf("requests = %#v, want %#v", requests, wantRequests)
	}
}

func TestUpdateVersionPolicyAndUnloadRollsBackWhenHealthCheckFails(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		status := http.StatusOK
		if request.URL.Path == "/v2/health/ready" {
			status = http.StatusServiceUnavailable
		}
		return &http.Response{StatusCode: status, Status: http.StatusText(status), Body: io.NopCloser(strings.NewReader("unavailable")), Request: request}, nil
	})}

	repository := filepath.Join(t.TempDir(), "repository")
	configPath := filepath.Join(repository, "detector", "config.pbtxt")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "name: \"detector\"\nversion_policy {\n  specific {\n    versions: [1]\n  }\n}\n"
	if err := os.WriteFile(configPath, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}

	runner, err := NewRunner(testLogger{}, "edge-1")
	if err != nil {
		t.Fatal(err)
	}
	runner.httpClient = client
	contextWithTimeout, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if err := runner.UpdateVersionPolicyAndUnload(contextWithTimeout, testTritonConfig(repository, "localhost", 8000), "detector", "2"); err == nil {
		t.Fatal("expected health check failure")
	}

	restored, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read restored config: %v", err)
	}
	if string(restored) != original {
		t.Fatalf("restored config = %q, want original", restored)
	}
	if _, err := os.Stat(configPath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("config backup should have been restored: %v", err)
	}
}

func testTritonConfig(repository, host string, port int) domain.TritonConfig {
	return domain.TritonConfig{
		Image:               "triton",
		ModelRepositoryPath: repository,
		IP:                  host,
		HostPort:            port,
		Model: domain.TritonModelConfig{
			Detector:   domain.TritonModelSpec{Name: "detector", Version: 1, ModelPathRaw: "detector.pt", ModelPathONNX: "detector.onnx", ModelPathTRT: "detector.plan"},
			Recognizer: domain.TritonModelSpec{Name: "recognizer", Version: 1, ModelPathRaw: "recognizer.pt", ModelPathONNX: "recognizer.onnx", ModelPathTRT: "recognizer.plan"},
		},
	}
}
