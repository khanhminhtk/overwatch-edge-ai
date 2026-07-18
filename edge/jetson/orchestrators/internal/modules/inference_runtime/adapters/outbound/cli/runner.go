package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"orchestrator/internal/modules/inference_runtime/domain"
)

type Logger interface{ Info(string, ...any) }

// Runner executes the existing Python inference runtime through uv.
type Runner struct {
	RuntimeDir   string
	UVPath       string
	DaemonModule string
	Stdout       io.Writer
	Stderr       io.Writer
	Stdin        io.Reader
	Logger       Logger

	mutex  sync.Mutex
	active *activeRun
}

type activeRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

var ErrAlreadyRunning = errors.New("inference runtime is already running")

// Config is supplied by the bootstrap instead of being hard-coded here.
type Config struct {
	RuntimeDir   string
	UVPath       string
	DaemonModule string
}

func NewRunner(config Config) (*Runner, error) {
	if config.RuntimeDir == "" {
		return nil, fmt.Errorf("inference runtime directory must not be empty")
	}
	if config.UVPath == "" {
		return nil, fmt.Errorf("inference runtime uv path must not be empty")
	}
	if config.DaemonModule == "" {
		return nil, fmt.Errorf("inference runtime module must not be empty")
	}
	return &Runner{RuntimeDir: config.RuntimeDir, UVPath: config.UVPath, DaemonModule: config.DaemonModule, Stdout: os.Stdout, Stderr: os.Stderr, Stdin: os.Stdin}, nil
}

// Command builds the uv command without invoking a shell.
func (r *Runner) Command(ctx context.Context, invocation domain.Invocation) (*exec.Cmd, error) {
	if r == nil || r.RuntimeDir == "" {
		return nil, fmt.Errorf("inference runtime directory must not be empty")
	}
	if ctx == nil {
		return nil, fmt.Errorf("command context must not be nil")
	}
	if r.UVPath == "" {
		return nil, fmt.Errorf("inference runtime uv path must not be empty")
	}
	if r.DaemonModule == "" {
		return nil, fmt.Errorf("inference runtime module must not be empty")
	}
	args := append([]string{"run", "-m", r.DaemonModule}, invocation.CLIArguments()...)
	command := exec.CommandContext(ctx, r.UVPath, args...)
	command.Dir = r.RuntimeDir
	command.Stdin, command.Stdout, command.Stderr = r.Stdin, r.Stdout, r.Stderr
	// Prefer a graceful SIGINT (the Python daemon releases video/Triton resources
	// in its finally block). If it does not exit, exec kills it after WaitDelay.
	command.Cancel = func() error {
		if command.Process == nil {
			return nil
		}
		return command.Process.Signal(os.Interrupt)
	}
	command.WaitDelay = 10 * time.Second
	return command, nil
}
func (r *Runner) Run(ctx context.Context, invocation domain.Invocation) error {
	command, run, runContext, err := r.start(ctx, invocation)
	if err != nil {
		return err
	}
	defer r.finish(run)
	if r.Logger != nil {
		r.Logger.Info("[INFERENCE_RUNTIME_STARTING]", "directory", r.RuntimeDir, "command", command.String())
	}
	if err := command.Run(); err != nil {
		if runContext.Err() != nil {
			return fmt.Errorf("run inference runtime: %w", runContext.Err())
		}
		return fmt.Errorf("run inference runtime: %w", err)
	}
	return nil
}

// Restart safely stops the active runtime, waits for it to exit, then starts a
// new invocation. Calling it while no runtime is active is equivalent to Run.
func (r *Runner) Restart(ctx context.Context, invocation domain.Invocation) error {
	if ctx == nil {
		return fmt.Errorf("restart context must not be nil")
	}
	if err := r.stop(ctx); err != nil {
		return err
	}
	return r.Run(ctx, invocation)
}

func (r *Runner) start(ctx context.Context, invocation domain.Invocation) (*exec.Cmd, *activeRun, context.Context, error) {
	if ctx == nil {
		return nil, nil, nil, fmt.Errorf("command context must not be nil")
	}
	runContext, cancel := context.WithCancel(ctx)
	command, err := r.Command(runContext, invocation)
	if err != nil {
		cancel()
		return nil, nil, nil, err
	}
	run := &activeRun{cancel: cancel, done: make(chan struct{})}
	r.mutex.Lock()
	if r.active != nil {
		r.mutex.Unlock()
		cancel()
		return nil, nil, nil, ErrAlreadyRunning
	}
	r.active = run
	r.mutex.Unlock()
	return command, run, runContext, nil
}

func (r *Runner) finish(run *activeRun) {
	run.cancel()
	r.mutex.Lock()
	if r.active == run {
		r.active = nil
	}
	close(run.done)
	r.mutex.Unlock()
}

func (r *Runner) stop(ctx context.Context) error {
	r.mutex.Lock()
	active := r.active
	r.mutex.Unlock()
	if active == nil {
		return nil
	}
	active.cancel()
	select {
	case <-active.done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("wait for inference runtime shutdown: %w", ctx.Err())
	}
}
