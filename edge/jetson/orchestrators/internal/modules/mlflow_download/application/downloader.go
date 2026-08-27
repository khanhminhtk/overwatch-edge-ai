package mlflow_download

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

type Config struct{ TrackingURI, StagingDirectory, DefaultArtifactPath string }
type Result struct{ ModelVersion, OutputPath string }
type Downloader struct {
	config Config
	client *http.Client
}

func New(config Config) (*Downloader, error) {
	if strings.TrimSpace(config.TrackingURI) == "" || strings.TrimSpace(config.StagingDirectory) == "" {
		return nil, fmt.Errorf("tracking_uri and staging_directory are required")
	}
	if config.DefaultArtifactPath == "" {
		config.DefaultArtifactPath = "checkpoints/best.pt"
	}
	return &Downloader{config: config, client: &http.Client{}}, nil
}
func (d *Downloader) Download(ctx context.Context, modelName, requestedVersion string) (Result, error) {
	version, err := d.resolveVersion(ctx, modelName, requestedVersion)
	if err != nil {
		return Result{}, err
	}
	uri, err := d.downloadURI(ctx, modelName, version)
	if err != nil {
		return Result{}, err
	}
	artifact := d.config.DefaultArtifactPath
	output := filepath.Join(d.config.StagingDirectory, modelName, version, filepath.Base(artifact))
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		return Result{}, err
	}
	if err := d.fetch(ctx, strings.TrimRight(uri, "/")+"/"+strings.TrimLeft(artifact, "/"), output); err != nil {
		return Result{}, err
	}
	return Result{ModelVersion: version, OutputPath: output}, nil
}
func (d *Downloader) resolveVersion(ctx context.Context, name, version string) (string, error) {
	if strings.TrimSpace(version) != "" {
		return strings.TrimSpace(version), nil
	}
	endpoint := d.config.TrackingURI + "/api/2.0/mlflow/registered-models/alias?name=" + url.QueryEscape(name) + "&alias=champion"
	var response struct {
		ModelVersion struct {
			Version string `json:"version"`
		} `json:"model_version"`
	}
	if err := d.getJSON(ctx, endpoint, &response); err != nil {
		return "", fmt.Errorf("resolve champion alias: %w", err)
	}
	if response.ModelVersion.Version == "" {
		return "", fmt.Errorf("champion alias missing for %s", name)
	}
	return response.ModelVersion.Version, nil
}
func (d *Downloader) downloadURI(ctx context.Context, name, version string) (string, error) {
	endpoint := d.config.TrackingURI + "/api/2.0/mlflow/model-versions/get-download-uri?name=" + url.QueryEscape(name) + "&version=" + url.QueryEscape(version)
	var response struct {
		ArtifactURI string `json:"artifact_uri"`
	}
	if err := d.getJSON(ctx, endpoint, &response); err != nil {
		return "", err
	}
	if response.ArtifactURI == "" {
		return "", fmt.Errorf("MLflow returned no artifact_uri")
	}
	return response.ArtifactURI, nil
}
func (d *Downloader) getJSON(ctx context.Context, endpoint string, target any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	response, err := d.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("MLflow returned %s", response.Status)
	}
	return json.NewDecoder(response.Body).Decode(target)
}
func (d *Downloader) fetch(ctx context.Context, uri, output string) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, uri, nil)
	if err != nil {
		return err
	}
	response, err := d.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("artifact download returned %s", response.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(output), ".download-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = io.Copy(tmp, response.Body); err != nil {
		tmp.Close()
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(name, output)
}
