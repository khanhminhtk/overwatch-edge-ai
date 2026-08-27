package application

import (
	"context"

	"web_ui/internal/domain"
)

type InferenceFrameBroadcaster interface {
	Broadcast(context.Context, domain.InferenceFrame) error
}
