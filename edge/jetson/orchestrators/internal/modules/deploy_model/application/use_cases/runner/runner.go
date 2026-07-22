package runner

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"time"

	"orchestrator/internal/modules/deploy_model/application/dto"
	"orchestrator/internal/modules/deploy_model/application/ports"
	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

type Logger interface {
	Info(string, ...any)
	Error(string, ...any)
}

type Runner struct {
	logger     Logger
	serverID   string
	httpClient *http.Client
	policy     domain.DeploymentPolicy
}

func NewRunner(logger Logger, serverID string) (Runner, error) {
	if logger == nil {
		return Runner{}, fmt.Errorf("logger must not be nil")
	}
	if serverID == "" {
		return Runner{}, fmt.Errorf("serverID must not be empty")
	}
	return Runner{logger: logger, serverID: serverID, httpClient: &http.Client{Timeout: 10 * time.Second}}, nil
}

var _ ports.DeploymentRunner = Runner{}

func (r Runner) Run(ctx context.Context, request dto.DeployRequest) error {
	if ctx == nil {
		return fmt.Errorf("run context must not be nil")
	}
	if err := request.Validate(); err != nil {
		return fmt.Errorf("validate deploy request: %w", err)
	}
	r.policy = request.Policy.WithDefaults()
	config, typeModel, version := request.Triton, string(request.Format), fmt.Sprintf("%d", request.Version)
	locks, err := acquireModelLocks(config.ModelRepositoryPath, config.Model)
	if err != nil {
		return err
	}
	defer locks.release()
	started := time.Now()
	r.logger.Info("deploy pipeline started", "server_id", r.serverID, "version", version)
	if ok, err := r.runMoveModel(ctx, config, typeModel, version); err != nil {
		return fmt.Errorf("move model artifacts: %w", err)
	} else if !ok {
		return fmt.Errorf("move model artifacts did not complete")
	}

	if err := r.deployModels(ctx, config, version); err != nil {
		return fmt.Errorf("deploy models: %w", err)
	}
	r.logger.Info("deploy pipeline completed", "server_id", r.serverID, "version", version, "duration_ms", time.Since(started).Milliseconds())
	return nil
}

func (r Runner) runCommand(ctx context.Context, command string, args ...string) error {
	if ctx == nil {
		return fmt.Errorf("run context must not be nil")
	}
	if command == "" {
		return fmt.Errorf("command must not be empty")
	}

	cmd := exec.CommandContext(ctx, command, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	r.logger.Info("running command", "server_id", r.serverID, "command", command, "args", args)
	if err := cmd.Run(); err != nil {
		r.logger.Error("command failed", "server_id", r.serverID, "command", command, "error", err)
		return fmt.Errorf("run %s: %w", command, err)
	}
	r.logger.Info("command completed", "server_id", r.serverID, "command", command)
	return nil
}
