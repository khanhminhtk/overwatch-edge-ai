package dto

// RunRequest contains the daemon flags supplied by an inbound adapter.
type RunRequest struct {
	Display   bool
	Arguments []string
}
