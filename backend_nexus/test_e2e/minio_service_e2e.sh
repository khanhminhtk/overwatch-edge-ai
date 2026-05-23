#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-config/.env.test}"
if [[ ! -f "$ENV_FILE" ]]; then
  ENV_FILE="config/.env.test"
fi

SOURCE_FILE="${SOURCE_FILE:-checkpoint_recognizer-20260514T090327Z-3-001.zip}"
if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "[FAIL] source file not found: $SOURCE_FILE"
  exit 1
fi

for cmd in grpcurl curl go awk sed grep; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[FAIL] missing required command: $cmd"
    exit 1
  fi
done

if ! command -v jq >/dev/null 2>&1; then
  echo "[FAIL] missing required command: jq"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

GRPC_PORT="${MINIOSERVICE_GRPC_PORT:-50001}"
GRPC_ADDR="127.0.0.1:${GRPC_PORT}"
GRPC_CACERT="${GRPC_CACERT:-certs/server.crt}"
if [[ ! -f "$GRPC_CACERT" ]]; then
  echo "[FAIL] gRPC CA cert not found: $GRPC_CACERT"
  exit 1
fi
GRPCURL_ARGS=(-cacert "$GRPC_CACERT")
BUCKET="tmp-e2e-$(date +%s)-$RANDOM"
OBJECT_NAME="$(basename "$SOURCE_FILE")"
REQUEST_PREFIX="e2e-$(date +%s)"
TMP_DIR="$(mktemp -d)"
DOWN_FILE="$TMP_DIR/$OBJECT_NAME"
SERVICE_PID=""
SERVICE_CHILD_PID=""

cleanup() {
  set +e
  if [[ -n "$SERVICE_CHILD_PID" ]] && kill -0 "$SERVICE_CHILD_PID" >/dev/null 2>&1; then
    kill -TERM "$SERVICE_CHILD_PID" >/dev/null 2>&1
    wait "$SERVICE_CHILD_PID" >/dev/null 2>&1
  fi
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" >/dev/null 2>&1; then
    kill -TERM "$SERVICE_PID" >/dev/null 2>&1
    wait "$SERVICE_PID" >/dev/null 2>&1
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[INFO] starting minio_service on ${GRPC_ADDR}"
MINIOSERVICE_GRPC_PORT="$GRPC_PORT" GOCACHE=/tmp/go-build-cache go run ./cmd/minio_service >/tmp/minio_service_e2e.log 2>&1 &
SERVICE_PID=$!

wait_health_serving() {
  local tries=40
  for ((i=1; i<=tries; i++)); do
    if grpcurl "${GRPCURL_ARGS[@]}" -d '{}' "$GRPC_ADDR" grpc.health.v1.Health/Check >/tmp/e2e_health.json 2>/dev/null; then
      local st
      st="$(jq -r '.status // empty' /tmp/e2e_health.json)"
      if [[ "$st" == "SERVING" ]]; then
        echo "[INFO] health is SERVING"
        return 0
      fi
    fi
    sleep 0.5
  done
  echo "[FAIL] health is not SERVING"
  cat /tmp/minio_service_e2e.log || true
  return 1
}

wait_health_serving
SERVICE_CHILD_PID="$(pgrep -P "$SERVICE_PID" | head -n1 || true)"

echo "[INFO] requesting access token"
REFRESH_JSON="$(grpcurl "${GRPCURL_ARGS[@]}" -d "{\"request_id\":\"${REQUEST_PREFIX}-refresh\",\"refresh_token\":\"${MINIOSERVICE_GRPC_REFRESH_TOKEN}\"}" "$GRPC_ADDR" objectstorage.ObjectStorageService/RefreshAccessToken)"
ACCESS_TOKEN="$(echo "$REFRESH_JSON" | jq -r '.accessToken // empty')"
SUCCESS="$(echo "$REFRESH_JSON" | jq -r '.success // false')"
if [[ "$SUCCESS" != "true" || -z "$ACCESS_TOKEN" ]]; then
  echo "[FAIL] cannot get access token"
  echo "$REFRESH_JSON"
  exit 1
fi
AUTH_HEADER="authorization: Bearer ${ACCESS_TOKEN}"

echo "[INFO] creating bucket: $BUCKET"
CREATE_BUCKET_JSON="$(grpcurl "${GRPCURL_ARGS[@]}" -H "$AUTH_HEADER" -d "{\"request_id\":\"${REQUEST_PREFIX}-create-bucket\",\"bucket_name\":\"${BUCKET}\"}" "$GRPC_ADDR" objectstorage.ObjectStorageService/CreateBucket)"
if [[ "$(echo "$CREATE_BUCKET_JSON" | jq -r '.success // false')" != "true" ]]; then
  echo "[FAIL] create bucket failed"
  echo "$CREATE_BUCKET_JSON"
  exit 1
fi

echo "[INFO] generating upload url"
UPLOAD_JSON="$(grpcurl "${GRPCURL_ARGS[@]}" -H "$AUTH_HEADER" -d "{\"request_id\":\"${REQUEST_PREFIX}-upload-url\",\"bucket_name\":\"${BUCKET}\",\"object_name\":\"${OBJECT_NAME}\"}" "$GRPC_ADDR" objectstorage.ObjectStorageService/GenerateUploadUrl)"
UPLOAD_URL="$(echo "$UPLOAD_JSON" | jq -r '.url // empty')"
if [[ "$(echo "$UPLOAD_JSON" | jq -r '.status // false')" != "true" || -z "$UPLOAD_URL" ]]; then
  echo "[FAIL] generate upload url failed"
  echo "$UPLOAD_JSON"
  exit 1
fi

echo "[INFO] uploading object"
HTTP_CODE="$(curl -sS -o /tmp/e2e_upload.out -w '%{http_code}' -X PUT --data-binary "@$SOURCE_FILE" -H 'Content-Type: application/octet-stream' "$UPLOAD_URL")"
if [[ "$HTTP_CODE" -lt 200 || "$HTTP_CODE" -gt 299 ]]; then
  echo "[FAIL] upload failed, status=$HTTP_CODE"
  cat /tmp/e2e_upload.out || true
  exit 1
fi

echo "[INFO] generating download url"
DOWNLOAD_JSON="$(grpcurl "${GRPCURL_ARGS[@]}" -H "$AUTH_HEADER" -d "{\"request_id\":\"${REQUEST_PREFIX}-download-url\",\"bucket_name\":\"${BUCKET}\",\"object_name\":\"${OBJECT_NAME}\"}" "$GRPC_ADDR" objectstorage.ObjectStorageService/GenerateDownloadUrl)"
DOWNLOAD_URL="$(echo "$DOWNLOAD_JSON" | jq -r '.url // empty')"
if [[ "$(echo "$DOWNLOAD_JSON" | jq -r '.status // false')" != "true" || -z "$DOWNLOAD_URL" ]]; then
  echo "[FAIL] generate download url failed"
  echo "$DOWNLOAD_JSON"
  exit 1
fi

echo "[INFO] downloading object to tmp"
HTTP_CODE="$(curl -sS -o "$DOWN_FILE" -w '%{http_code}' "$DOWNLOAD_URL")"
if [[ "$HTTP_CODE" -lt 200 || "$HTTP_CODE" -gt 299 ]]; then
  echo "[FAIL] download failed, status=$HTTP_CODE"
  exit 1
fi

if ! cmp -s "$SOURCE_FILE" "$DOWN_FILE"; then
  echo "[FAIL] downloaded file mismatch"
  exit 1
fi

echo "[INFO] removing object via temporary go helper"
cat > /tmp/minio_remove_object_e2e.go << 'GOEOF'
package main

import (
  "fmt"
  "os"

  "backend_nexus/internal/infra"
  configInfra "backend_nexus/internal/infra/config"
  "backend_nexus/internal/utils"
)

func main() {
  cfg, err := utils.NewConfigLoader[configInfra.Minio]("config/minio_config.yaml", os.Getenv("ENV_FILE")).LoadConfig()
  if err != nil { panic(err) }
  c, err := infra.NewMinioClient(cfg.MinioServer)
  if err != nil { panic(err) }
  bucket := os.Getenv("E2E_BUCKET")
  object := os.Getenv("E2E_OBJECT")
  if err := c.RemoveObject(bucket, object); err != nil { panic(err) }
  fmt.Println("removed object")
}
GOEOF
ENV_FILE="$ENV_FILE" E2E_BUCKET="$BUCKET" E2E_OBJECT="$OBJECT_NAME" GOCACHE=/tmp/go-build-cache go run /tmp/minio_remove_object_e2e.go >/tmp/e2e_remove_object.log 2>&1

echo "[INFO] removing bucket"
REMOVE_BUCKET_JSON="$(grpcurl "${GRPCURL_ARGS[@]}" -H "$AUTH_HEADER" -d "{\"request_id\":\"${REQUEST_PREFIX}-remove-bucket\",\"bucket_name\":\"${BUCKET}\"}" "$GRPC_ADDR" objectstorage.ObjectStorageService/RemoveBucket)"
if [[ "$(echo "$REMOVE_BUCKET_JSON" | jq -r '.success // false')" != "true" ]]; then
  echo "[FAIL] remove bucket failed"
  echo "$REMOVE_BUCKET_JSON"
  exit 1
fi

echo "[INFO] stopping service"
if [[ -n "$SERVICE_CHILD_PID" ]] && kill -0 "$SERVICE_CHILD_PID" >/dev/null 2>&1; then
  kill -TERM "$SERVICE_CHILD_PID"
  wait "$SERVICE_CHILD_PID" >/dev/null 2>&1 || true
fi
kill -TERM "$SERVICE_PID" >/dev/null 2>&1 || true
wait "$SERVICE_PID" >/dev/null 2>&1 || true
SERVICE_CHILD_PID=""
SERVICE_PID=""

echo "[INFO] checking health should fail after stop"
if grpcurl "${GRPCURL_ARGS[@]}" -d '{}' "$GRPC_ADDR" grpc.health.v1.Health/Check >/tmp/e2e_after_stop_health.json 2>/tmp/e2e_after_stop_health.err; then
  echo "[FAIL] healthcheck unexpectedly succeeded after service stop"
  cat /tmp/e2e_after_stop_health.json || true
  exit 1
fi

echo "[PASS] minio e2e flow completed successfully"
