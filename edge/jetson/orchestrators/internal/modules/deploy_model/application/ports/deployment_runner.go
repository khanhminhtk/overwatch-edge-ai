package ports

import (
	"context"

	"orchestrator/internal/modules/deploy_model/application/dto"
)

type DeploymentRunner interface {
	Run(context.Context, dto.DeployRequest) error
}
