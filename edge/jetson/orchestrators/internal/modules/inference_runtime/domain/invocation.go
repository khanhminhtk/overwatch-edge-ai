package domain

import (
	"fmt"
	"strings"
)

// Invocation is an immutable request to execute the inference daemon.
type Invocation struct {
	display   bool
	arguments []string
}

func NewInvocation(display bool, arguments []string) (Invocation, error) {
	for _, argument := range arguments {
		if strings.TrimSpace(argument) == "" {
			return Invocation{}, fmt.Errorf("inference runtime argument must not be empty")
		}
	}
	return Invocation{display: display, arguments: append([]string(nil), arguments...)}, nil
}
func (i Invocation) Display() bool       { return i.display }
func (i Invocation) Arguments() []string { return append([]string(nil), i.arguments...) }
func (i Invocation) CLIArguments() []string {
	arguments := i.Arguments()
	if i.display {
		return append([]string{"--display"}, arguments...)
	}
	return arguments
}
