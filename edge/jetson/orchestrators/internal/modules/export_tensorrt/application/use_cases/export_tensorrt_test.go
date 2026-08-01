package usecases

import (
	"context"
	"testing"

	"orchestrator/internal/modules/export_tensorrt/application/dto"
	"orchestrator/internal/modules/export_tensorrt/domain"
)

func TestExecuteDelegatesValidatedRequestToBuilder(t *testing.T) {
	builder := &fakeBuilder{}
	useCase, err := NewExportTensorrtUseCase(builder, testConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	input := dto.ExportRequest{ModelName: "recognizer", ModelVersion: "2", RepositoryRoot: "/workspace"}
	if err := useCase.Execute(context.Background(), input); err != nil {
		t.Fatal(err)
	}
	if builder.request.ModelName() != "recognizer" || builder.request.ModelVersion() != "2" || builder.root != "/workspace" {
		t.Fatalf("builder received request=%#v root=%q", builder.request, builder.root)
	}
}

type fakeBuilder struct {
	request domain.ExportRequest
	root    string
}

func (b *fakeBuilder) Build(_ context.Context, request domain.ExportRequest, _ domain.Config, root string) error {
	b.request, b.root = request, root
	return nil
}

func testConfig() domain.Config {
	return domain.Config{TritonDockerEnvFile: ".env", TritonDockerComposeFile: "compose.yaml", TritonBuilderProfileFormat: "builder-%s", TritonBuilderServiceNameFormat: "builder-%s"}
}
