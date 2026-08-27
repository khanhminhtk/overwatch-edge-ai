package httpui

import (
	"context"
	"net/http"

	"web_ui/internal/domain"
)

type inferenceFramePublisher interface {
	Execute(context.Context, domain.InferenceFrame) error
}

type frameSubscriptionHub interface {
	Subscribe() (<-chan domain.InferenceFrame, func())
}

func NewRouter(staticHandler http.Handler, publisher inferenceFramePublisher, hub frameSubscriptionHub) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.Handle("GET /", resolveStaticHandler(staticHandler))
	mux.Handle("POST /ingest", newIngestHandler(publisher))
	mux.Handle("GET /ws", newWebsocketHandler(hub))

	return mux
}

func resolveStaticHandler(staticHandler http.Handler) http.Handler {
	if staticHandler != nil {
		return staticHandler
	}

	return http.FileServer(http.Dir("./static"))
}
