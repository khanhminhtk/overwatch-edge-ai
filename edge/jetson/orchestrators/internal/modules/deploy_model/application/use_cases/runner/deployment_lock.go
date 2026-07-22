package runner

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

type modelLocks []*os.File

func acquireModelLocks(repository string, models domain.TritonModelConfig) (modelLocks, error) {
	locks := modelLocks{}
	for _, model := range []domain.TritonModelSpec{models.Detector, models.Recognizer} {
		if err := os.MkdirAll(filepath.Join(repository, model.Name), 0o755); err != nil {
			locks.release()
			return nil, err
		}
		file, err := os.OpenFile(filepath.Join(repository, model.Name, ".deploy.lock"), os.O_CREATE|os.O_RDWR, 0o600)
		if err != nil {
			locks.release()
			return nil, err
		}
		if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
			file.Close()
			locks.release()
			return nil, fmt.Errorf("model %q is already being deployed: %w", model.Name, err)
		}
		locks = append(locks, file)
	}
	return locks, nil
}

func (locks modelLocks) release() {
	for _, file := range locks {
		_ = syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
		_ = file.Close()
	}
}
