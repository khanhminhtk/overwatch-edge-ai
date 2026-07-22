package runner

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

func (r Runner) tritonRequest(ctx context.Context, method, endpoint string) error {
	request, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	r.logger.Info("calling triton endpoint", "server_id", r.serverID, "method", method, "endpoint", endpoint)
	client := r.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	clientCopy := *client
	clientCopy.Timeout = r.policy.WithDefaults().HTTPTimeout
	response, err := clientCopy.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4<<10))
		return fmt.Errorf("unexpected status %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	return nil
}
