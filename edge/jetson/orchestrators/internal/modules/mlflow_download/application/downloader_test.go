package mlflow_download

import "testing"

func TestNewRequiresTrackingAndStaging(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("expected validation error")
	}
}
