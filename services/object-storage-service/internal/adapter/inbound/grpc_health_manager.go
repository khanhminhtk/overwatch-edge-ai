package inbound

import (
	"sync"

	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type DependencyStatus struct {
	Name    string
	Ready   bool
	Message string
}

type HealthManager struct {
	mu          sync.RWMutex
	ready       bool
	deps        map[string]DependencyStatus
	grpcHealth  *health.Server
	serviceName string
}

func NewHealthManager(grpcHealth *health.Server, serviceName string) *HealthManager {
	h := &HealthManager{
		ready:       false,
		deps:        make(map[string]DependencyStatus),
		grpcHealth:  grpcHealth,
		serviceName: serviceName,
	}

	h.syncGRPCStatus()
	return h
}

func (h *HealthManager) SetReady(ready bool) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.ready = ready
	h.syncGRPCStatus()
}

func (h *HealthManager) SetDependency(name string, ready bool, message string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.deps[name] = DependencyStatus{
		Name:    name,
		Ready:   ready,
		Message: message,
	}
	h.syncGRPCStatus()
}

func (h *HealthManager) Snapshot() (bool, map[string]DependencyStatus) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	deps := make(map[string]DependencyStatus, len(h.deps))
	for k, v := range h.deps {
		deps[k] = v
	}
	return h.ready, deps
}

func (h *HealthManager) syncGRPCStatus() {
	if h.grpcHealth == nil {
		return
	}

	status := grpc_health_v1.HealthCheckResponse_NOT_SERVING
	if h.ready {
		allDepsReady := true
		for _, dep := range h.deps {
			if !dep.Ready {
				allDepsReady = false
				break
			}
		}
		if allDepsReady {
			status = grpc_health_v1.HealthCheckResponse_SERVING
		}
	}

	h.grpcHealth.SetServingStatus("", status)
	if h.serviceName != "" {
		h.grpcHealth.SetServingStatus(h.serviceName, status)
	}
}
