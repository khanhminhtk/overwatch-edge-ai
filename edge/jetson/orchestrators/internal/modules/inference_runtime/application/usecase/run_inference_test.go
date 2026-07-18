package usecase

import (
	"context"
	"testing"

	"orchestrator/internal/modules/inference_runtime/application/dto"
	"orchestrator/internal/modules/inference_runtime/domain"
)

func TestExecuteDelegatesValidatedInvocation(t *testing.T) {
	runner := &fakeRunner{}
	useCase, err := NewRunInference(runner)
	if err != nil {
		t.Fatal(err)
	}
	if err := useCase.Execute(context.Background(), dto.RunRequest{Display: true, Arguments: []string{"--camera", "0"}}); err != nil {
		t.Fatal(err)
	}
	if !runner.invocation.Display() || len(runner.invocation.Arguments()) != 2 {
		t.Fatalf("unexpected invocation: %#v", runner.invocation)
	}
}
func TestNewRunInferenceRejectsNilRunner(t *testing.T) {
	if _, err := NewRunInference(nil); err == nil {
		t.Fatal("expected error")
	}
}

type fakeRunner struct{ invocation domain.Invocation }

func (f *fakeRunner) Run(_ context.Context, invocation domain.Invocation) error {
	f.invocation = invocation
	return nil
}
