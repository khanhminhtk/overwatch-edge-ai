package minio

import "testing"

func TestNormalizeEndpointAddress_DefaultPort(t *testing.T) {
	addr, err := normalizeEndpointAddress("http://127.0.0.1")
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if addr != "127.0.0.1:9000" {
		t.Fatalf("expected 127.0.0.1:9000, got %s", addr)
	}
}
