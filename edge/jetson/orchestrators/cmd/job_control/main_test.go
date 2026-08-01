package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveConfigDir(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(repoRoot, "edge", "jetson", "orchestrators", "config")

	for _, directory := range []string{repoRoot, filepath.Join(repoRoot, "edge", "jetson", "orchestrators")} {
		t.Run(directory, func(t *testing.T) {
			originalDir, err := os.Getwd()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = os.Chdir(originalDir) })
			if err := os.Chdir(directory); err != nil {
				t.Fatal(err)
			}

			got, err := resolveConfigDir()
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("resolveConfigDir() = %q, want %q", got, want)
			}
		})
	}
}
