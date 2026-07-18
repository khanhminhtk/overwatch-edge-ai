package cli

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"reflect"
	"testing"
	"time"

	"orchestrator/internal/modules/inference_runtime/domain"
)

func TestCommandUsesUVModuleAndWorkingDirectory(t *testing.T) {
	runner, err := NewRunner(Config{RuntimeDir: "/repo/edge/jetson/inference-runtime", UVPath: "uv", DaemonModule: "src.entrypoints.inference_daemon.main"})
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := domain.NewInvocation(true, []string{"--camera", "0"})
	if err != nil {
		t.Fatal(err)
	}
	command, err := runner.Command(context.Background(), invocation)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"uv", "run", "-m", "src.entrypoints.inference_daemon.main", "--display", "--camera", "0"}
	if !reflect.DeepEqual(command.Args, want) {
		t.Fatalf("got %#v, want %#v", command.Args, want)
	}
	if command.Dir != runner.RuntimeDir {
		t.Fatalf("got directory %q", command.Dir)
	}
}

func TestRestartStopsActiveRunBeforeStartingNewOne(t *testing.T) {
	t.Setenv("GO_WANT_INFERENCE_RUNTIME_HELPER", "1")
	runner, err := NewRunner(Config{RuntimeDir: t.TempDir(), UVPath: os.Args[0], DaemonModule: "helper"})
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := domain.NewInvocation(false, nil)
	if err != nil {
		t.Fatal(err)
	}
	firstDone := make(chan error, 1)
	go func() { firstDone <- runner.Run(context.Background(), invocation) }()
	waitForActive(t, runner, nil)
	restartContext, cancelRestart := context.WithCancel(context.Background())
	restarted := make(chan error, 1)
	go func() { restarted <- runner.Restart(restartContext, invocation) }()
	if err := <-firstDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("first run error = %v, want context canceled", err)
	}
	waitForActive(t, runner, nil)
	cancelRestart()
	if err := <-restarted; !errors.Is(err, context.Canceled) {
		t.Fatalf("restart error = %v, want context canceled", err)
	}
}

func TestMain(m *testing.M) {
	if os.Getenv("GO_WANT_INFERENCE_RUNTIME_HELPER") != "1" {
		os.Exit(m.Run())
	}
	signals := make(chan os.Signal, 1)
	// CommandContext sends SIGINT; exiting here models the daemon cleanup path.
	signal.Notify(signals, os.Interrupt)
	<-signals
	os.Exit(0)
}

func waitForActive(t *testing.T, runner *Runner, previous *activeRun) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		runner.mutex.Lock()
		active := runner.active
		runner.mutex.Unlock()
		if active != nil && active != previous {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("inference runtime did not become active")
}
