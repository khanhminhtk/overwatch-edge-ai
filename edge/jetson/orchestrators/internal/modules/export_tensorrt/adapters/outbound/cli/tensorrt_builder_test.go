package cli

import (
	"context"
	"strings"
	"testing"

	"orchestrator/internal/modules/export_tensorrt/domain"
)

func TestCommandBuildsDockerComposeInvocation(t *testing.T) {
	request, err := domain.NewExportRequest("recognizer", "2")
	if err != nil {
		t.Fatal(err)
	}
	builder := NewTensorRTBuilder()
	command, err := builder.Command(context.Background(), request, domain.Config{
		TritonDockerEnvFile:            "config/.env",
		TritonDockerComposeFile:        "compose/triton.yaml",
		TritonBuilderProfileFormat:     "builder-%s",
		TritonBuilderServiceNameFormat: "triton_builder_%s",
	}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(command.Args, " ")
	for _, expected := range []string{"docker compose", "--profile builder-recognizer", "run --rm --build triton_builder_recognizer"} {
		if !strings.Contains(got, expected) {
			t.Fatalf("command %q does not contain %q", got, expected)
		}
	}
	if !strings.Contains(strings.Join(command.Env, "\n"), "TRITON_MODEL_VERSION=2") {
		t.Fatal("command environment is missing the model version")
	}
}
