package runner

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

var defaultModelFilenameStart = regexp.MustCompile(`(?m)^[\t ]*default_model_filename[\t ]*:[\t ]*"([^"]+)"`)

// runMoveModel copies the selected artifact type for each configured model
// into Triton's <repository>/<model>/<version> layout.
func (r Runner) runMoveModel(ctx context.Context, config domain.TritonConfig, typeModel, version string) (bool, error) {
	if ctx == nil {
		return false, fmt.Errorf("run context must not be nil")
	}
	if err := config.Validate(); err != nil {
		return false, fmt.Errorf("validate triton config: %w", err)
	}
	if version == "" {
		return false, fmt.Errorf("model version must not be empty")
	}

	models := []domain.TritonModelSpec{config.Model.Detector, config.Model.Recognizer}
	for _, model := range models {
		source, err := modelPath(model, typeModel)
		if err != nil {
			return false, err
		}
		destination := filepath.Join(config.ModelRepositoryPath, model.Name, version)
		if err := os.MkdirAll(destination, 0o755); err != nil {
			return false, fmt.Errorf("create model destination %q: %w", destination, err)
		}
		destinationFile, err := modelArtifactDestination(config.ModelRepositoryPath, model.Name, destination, source)
		if err != nil {
			return false, err
		}
		if _, err := os.Stat(destinationFile); err == nil {
			if err := sameFileDigest(source, destinationFile); err != nil {
				return false, fmt.Errorf("verify existing model artifact %q: %w", model.Name, err)
			}
			r.logger.Info("model artifact already present and verified", "server_id", r.serverID, "model", model.Name, "path", destinationFile)
			continue
		} else if !os.IsNotExist(err) {
			return false, fmt.Errorf("stat model destination %q: %w", destinationFile, err)
		}
		if _, err := os.Stat(source); err != nil {
			if !os.IsNotExist(err) {
				return false, fmt.Errorf("stat model %q: %w", source, err)
			}
			legacyDestination := filepath.Join(destination, filepath.Base(source))
			if legacyDestination != destinationFile {
				if _, legacyErr := os.Stat(legacyDestination); legacyErr == nil {
					if err := copyArtifactAtomically(legacyDestination, destinationFile); err != nil {
						return false, fmt.Errorf("rename model artifact %q: %w", model.Name, err)
					}
					continue
				}
			}
			return false, fmt.Errorf("model source %q and destination %q do not exist", source, destinationFile)
		}
		if err := copyArtifactAtomically(source, destinationFile); err != nil {
			return false, fmt.Errorf("move model %q: %w", model.Name, err)
		}
	}

	return true, nil
}

func copyArtifactAtomically(source, destination string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	info, err := input.Stat()
	if err != nil {
		return err
	}
	temporary := destination + ".staging"
	output, err := os.OpenFile(temporary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm())
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(temporary) }()
	if _, err := io.Copy(output, input); err != nil {
		output.Close()
		return err
	}
	if err := output.Sync(); err != nil {
		output.Close()
		return err
	}
	if err := output.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, destination)
}

func sameFileDigest(source, destination string) error {
	sourceDigest, err := fileDigest(source)
	if err != nil {
		return err
	}
	destinationDigest, err := fileDigest(destination)
	if err != nil {
		return err
	}
	if sourceDigest != destinationDigest {
		return fmt.Errorf("checksum differs")
	}
	return nil
}

func fileDigest(path string) ([32]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return [32]byte{}, err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return [32]byte{}, err
	}
	var result [32]byte
	copy(result[:], hash.Sum(nil))
	return result, nil
}

func modelArtifactDestination(repository, modelName, destination, source string) (string, error) {
	configPath := filepath.Join(repository, modelName, "config.pbtxt")
	contents, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return filepath.Join(destination, filepath.Base(source)), nil
		}
		return "", fmt.Errorf("read model config %q: %w", configPath, err)
	}
	match := defaultModelFilenameStart.FindStringSubmatch(string(contents))
	if len(match) != 2 || strings.TrimSpace(match[1]) == "" {
		return filepath.Join(destination, filepath.Base(source)), nil
	}
	return filepath.Join(destination, match[1]), nil
}

func modelPath(model domain.TritonModelSpec, typeModel string) (string, error) {
	switch typeModel {
	case "pt":
		return model.ModelPathRaw, nil
	case "onnx":
		return model.ModelPathONNX, nil
	case "trt":
		return model.ModelPathTRT, nil
	default:
		return "", fmt.Errorf("unknown model type %q", typeModel)
	}
}
