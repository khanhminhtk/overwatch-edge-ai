package domain

import "testing"

func TestInvocationCopiesArgumentsAndAddsDisplayFlag(t *testing.T) {
	invocation, err := NewInvocation(true, []string{"--camera", "0"})
	if err != nil {
		t.Fatal(err)
	}
	arguments := invocation.CLIArguments()
	if len(arguments) != 3 || arguments[0] != "--display" {
		t.Fatalf("unexpected arguments: %#v", arguments)
	}
	arguments[0] = "changed"
	if invocation.CLIArguments()[0] != "--display" {
		t.Fatal("invocation leaked mutable arguments")
	}
}
func TestInvocationRejectsEmptyArgument(t *testing.T) {
	if _, err := NewInvocation(false, []string{""}); err == nil {
		t.Fatal("expected validation error")
	}
}
