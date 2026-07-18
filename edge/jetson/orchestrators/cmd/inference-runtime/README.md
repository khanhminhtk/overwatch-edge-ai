# Inference runtime command

From `edge/jetson/orchestrator`, run the existing Python daemon in display mode:

```bash
go run ./cmd/inference-runtime
```

It loads `config/.env` and `config/orchestrator.yaml` through
`internal/platform/config`, resolves the configured inference-runtime directory,
and executes the configured `uv run -m ...` command there.

Forward daemon options after `--`:

```bash
go run ./cmd/inference-runtime -- --camera 0
go run ./cmd/inference-runtime -- --image-path /path/to/image.jpg
go run ./cmd/inference-runtime -display=false -- --video-path /path/to/video.mp4
```

Use `-config-dir /path/to/config` or `-runtime-dir /custom/inference-runtime`
to override locations. Environment variables override the `.env` values.
