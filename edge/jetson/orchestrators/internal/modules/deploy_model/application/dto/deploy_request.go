package dto

import (
	"fmt"

	domain "orchestrator/internal/modules/deploy_model/domain/value_objects"
)

type DeployRequest struct {
	Triton  domain.TritonConfig
	Format  domain.ModelFormat
	Version int
	Policy  domain.DeploymentPolicy
}

func (r DeployRequest) Validate() error {
	if err := r.Triton.Validate(); err != nil {
		return fmt.Errorf("triton: %w", err)
	}
	if err := r.Format.Validate(); err != nil {
		return err
	}
	if r.Version <= 0 {
		return fmt.Errorf("version must be positive")
	}
	return r.Policy.Validate()
}
