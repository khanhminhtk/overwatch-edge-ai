package runner

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"time"

	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

// UpdateVersionPolicyAndUnload preserves the previous Triton configuration as
// config.pbtxt.tmp, writes the requested specific version policy, unloads the
// model, and removes the backup only after Triton reports the selected version
// and server as ready.
func (r Runner) UpdateVersionPolicyAndUnload(ctx context.Context, config domain.TritonConfig, modelName, version string) error {
	if ctx == nil {
		return fmt.Errorf("unload context must not be nil")
	}
	if err := config.Validate(); err != nil {
		return fmt.Errorf("validate triton config: %w", err)
	}
	if modelName == "" {
		return fmt.Errorf("model name must not be empty")
	}
	parsedVersion, err := strconv.ParseInt(version, 10, 64)
	if err != nil {
		return fmt.Errorf("model version must be an integer: %w", err)
	}
	if parsedVersion <= 0 {
		return fmt.Errorf("model version must be positive")
	}

	configPath := filepath.Join(config.ModelRepositoryPath, modelName, "config.pbtxt")
	if err := updateVersionPolicy(configPath, version); err != nil {
		return err
	}

	if err := r.deployTritonModel(ctx, config, modelName, version); err != nil {
		if rollbackErr := rollbackVersionPolicy(configPath); rollbackErr != nil {
			return fmt.Errorf("%w; rollback config: %v", err, rollbackErr)
		}
		return err
	}
	if err := r.checkTritonHealth(ctx, config); err != nil {
		if rollbackErr := rollbackVersionPolicy(configPath); rollbackErr != nil {
			return fmt.Errorf("%w; rollback config: %v", err, rollbackErr)
		}
		return err
	}
	if err := os.Remove(configPath + ".tmp"); err != nil {
		return fmt.Errorf("remove triton config backup: %w", err)
	}
	return nil
}

func (r Runner) deployTritonModel(ctx context.Context, config domain.TritonConfig, modelName, version string) error {
	baseURL := fmt.Sprintf("http://%s", net.JoinHostPort(config.IP, strconv.Itoa(config.HostPort)))
	escapedModelName := url.PathEscape(modelName)
	if err := r.tritonRequest(ctx, http.MethodPost, baseURL+"/v2/repository/models/"+escapedModelName+"/unload"); err != nil {
		return fmt.Errorf("unload triton model %q: %w", modelName, err)
	}
	if err := r.tritonRequest(ctx, http.MethodPost, baseURL+"/v2/repository/models/"+escapedModelName+"/load"); err != nil {
		return fmt.Errorf("load triton model %q: %w", modelName, err)
	}
	if err := r.waitForTriton(ctx, baseURL+"/v2/models/"+escapedModelName+"/versions/"+version+"/ready"); err != nil {
		return fmt.Errorf("check triton model %q version %s readiness: %w", modelName, version, err)
	}
	return nil
}

func (r Runner) deployModels(ctx context.Context, config domain.TritonConfig, version string) (resultErr error) {
	configPaths := make([]string, 0, 2)
	shouldRollback := false
	defer func() {
		if resultErr == nil || !shouldRollback {
			return
		}
		for _, configPath := range configPaths {
			if err := rollbackVersionPolicy(configPath); err != nil {
				resultErr = fmt.Errorf("%w; rollback config %q: %v", resultErr, configPath, err)
			}
		}
		for _, model := range []domain.TritonModelSpec{config.Model.Detector, config.Model.Recognizer} {
			if err := r.reloadTritonModel(ctx, config, model.Name); err != nil {
				resultErr = fmt.Errorf("%w; rollback runtime model %q: %v", resultErr, model.Name, err)
			}
		}
	}()

	for _, model := range []domain.TritonModelSpec{config.Model.Detector, config.Model.Recognizer} {
		configPath := filepath.Join(config.ModelRepositoryPath, model.Name, "config.pbtxt")
		if err := updateVersionPolicy(configPath, version); err != nil {
			return fmt.Errorf("update model %q version policy: %w", model.Name, err)
		}
		configPaths = append(configPaths, configPath)
		shouldRollback = true
		if err := r.deployTritonModel(ctx, config, model.Name, version); err != nil {
			return fmt.Errorf("deploy model %q: %w", model.Name, err)
		}
	}
	if err := r.checkTritonHealth(ctx, config); err != nil {
		return err
	}
	shouldRollback = false
	for _, configPath := range configPaths {
		if err := os.Remove(configPath + ".tmp"); err != nil {
			return fmt.Errorf("remove triton config backup: %w", err)
		}
	}
	return nil
}

func (r Runner) reloadTritonModel(ctx context.Context, config domain.TritonConfig, modelName string) error {
	baseURL := fmt.Sprintf("http://%s", net.JoinHostPort(config.IP, strconv.Itoa(config.HostPort)))
	modelURL := baseURL + "/v2/repository/models/" + url.PathEscape(modelName)
	if err := r.tritonRequest(ctx, http.MethodPost, modelURL+"/unload"); err != nil {
		return err
	}
	return r.tritonRequest(ctx, http.MethodPost, modelURL+"/load")
}

func (r Runner) checkTritonHealth(ctx context.Context, config domain.TritonConfig) error {
	baseURL := fmt.Sprintf("http://%s", net.JoinHostPort(config.IP, strconv.Itoa(config.HostPort)))
	if err := r.waitForTriton(ctx, baseURL+"/v2/health/ready"); err != nil {
		return fmt.Errorf("check triton health readiness: %w", err)
	}
	return nil
}

func (r Runner) waitForTriton(ctx context.Context, endpoint string) error {
	timeout := r.policy.WithDefaults().ReadinessTimeout
	interval := r.policy.WithDefaults().PollInterval
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	var lastErr error
	for {
		if err := r.tritonRequest(ctx, http.MethodGet, endpoint); err == nil {
			return nil
		} else {
			lastErr = err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("timed out waiting for Triton: %w", lastErr)
		case <-ticker.C:
		}
	}
}
