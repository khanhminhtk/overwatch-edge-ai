package domain

import "testing"

func TestNewExportRequestTrimsAndValidatesFields(t *testing.T) {
	request, err := NewExportRequest(" recognizer ", " 2 ")
	if err != nil {
		t.Fatal(err)
	}
	if request.ModelName() != "recognizer" || request.ModelVersion() != "2" {
		t.Fatalf("request = %#v", request)
	}
	if _, err := NewExportRequest("", "1"); err == nil {
		t.Fatal("expected empty model name error")
	}
}
