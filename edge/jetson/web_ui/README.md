# Web UI

This service hosts the local web dashboard for viewing inference frames and detection overlays. It starts with mock frames so the UI is immediately usable, then the latest real ingest frame takes over when `POST /ingest` receives one.

## Run

```bash
cd web_ui
go run ./cmd/web_ui
```

Open `http://localhost:8080`.

## API

- `POST /ingest` accepts a JSON frame payload with:
  - `frame_id`
  - `timestamp`
  - `image_payload`
  - `detections`
  - each detection includes `label`, `score`, and `bbox`
  - `bbox` keys: `x`, `y`, `width`, `height`
- `GET /ws` streams the latest frame updates to the browser.
- `GET /healthz` returns `200 OK` for health checks.

## Behavior

- Mock frames are shown locally until real ingest frames arrive.
- Once real frames are ingested, they replace the mock view in the latest frame panel.
- `timestamp` should be RFC3339Nano formatted, and `image_payload` should be a loadable image URL or data URI.
