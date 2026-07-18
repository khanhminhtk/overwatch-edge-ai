# Inference runtime wrapper

This module launches the existing Python daemon through `uv`; it does not
duplicate the Triton/OpenCV inference implementation in Go. It follows a
dependency-inward layout:

```text
domain/                       Invocation value object and invariants
application/dto/              inbound request data
application/ports/            runtime execution port
application/usecase/          RunInference orchestration
adapters/outbound/cli/        uv/exec implementation of the port
```

```go
runner, err := cli.NewRunner(cli.Config{
    RuntimeDir: runtimeDir,
    UVPath: "uv",
    DaemonModule: "src.entrypoints.inference_daemon.main",
})
if err != nil { return err }
runInference, err := usecase.NewRunInference(runner)
if err != nil { return err }

// cd <repo>/edge/jetson/inference-runtime &&
// uv run -m src.entrypoints.inference_daemon.main --display
if err := runInference.Execute(ctx, dto.RunRequest{Display: true}); err != nil { return err }
```

Pass Python daemon flags through `dto.RunRequest.Arguments`, for example
`[]string{"--camera", "0"}`. Stdout, stderr, and stdin are forwarded to the
Go process by default.

The CLI adapter also provides `runner.Restart(ctx, invocation)`. It sends
`SIGINT` to the currently running daemon, waits for the process to release its
resources, then starts the new invocation. If graceful shutdown does not finish
within ten seconds, Go terminates the process as a fallback.
