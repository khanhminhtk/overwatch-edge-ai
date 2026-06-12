#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Starting quickstart script in $ROOT_DIR"

ENV_FILE="${ENV_FILE:-config/.env}"
ENV_FILE_BACKEND_NEXUS="${ENV_FILE_BACKEND_NEXUS:-services/object-storage-service/config/.env}"

set -a
source "$ENV_FILE"
source "$ENV_FILE_BACKEND_NEXUS"
set +a

SERVICE_PID=""

cleanup() {
    local exit_code=$?

    if [[ -n "${SERVICE_PID:-}" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
        echo "Stopping minio_service process group: $SERVICE_PID..."

        kill -- "-$SERVICE_PID" 2>/dev/null || true

        for _ in {1..10}; do
            if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
                echo "minio_service stopped."
                exit "$exit_code"
            fi
            sleep 1
        done

        echo "Force killing minio_service process group: $SERVICE_PID..."
        kill -9 -- "-$SERVICE_PID" 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$ROOT_DIR/services/object-storage-service"

setsid bash -c 'go run cmd/minio_service/main.go' >minio_service.log 2>&1 &
SERVICE_PID=$!
echo "Started minio_service with PID $SERVICE_PID"

REQUEST_ID="request_id"
GRPC_HOST="${MINIOSERVICE_GRPC_HOST:-${IP_DARABASE_SERVICE_HOST:-127.0.0.1}}"
GRPC_PORT="${MINIOSERVICE_GRPC_PORT:-${IP_DARABASE_SERVICE_PORT:-50001}}"
GRPC_ADDR="${GRPC_HOST}:${GRPC_PORT}"

sleep 5

bucket_exists() {
    local bucket_name="$1"
    local request_id="$2"
    local auth_header="$3"
    local exists

    exists=$(grpcurl -cacert certs/server.crt \
      -H "$auth_header" \
      -d '{"request_id":"'"$request_id"'","bucket_name":"'"$bucket_name"'"}' \
      "$GRPC_ADDR" \
      objectstorage.ObjectStorageService/BucketExists | jq -r '.success')

    if [[ "$exists" == "true" ]]; then
      echo "Bucket '$bucket_name' exists."
      return 0
    fi

    echo "Bucket '$bucket_name' does not exist."
    return 1
}

access_token=$(grpcurl -cacert certs/server.crt \
  -d '{"request_id":"'"$REQUEST_ID"'","refresh_token":"'"$MINIOSERVICE_GRPC_REFRESH_TOKEN"'"}' \
  "$GRPC_ADDR" \
  objectstorage.ObjectStorageService/RefreshAccessToken | jq -r '.accessToken')

if [[ -z "$access_token" || "$access_token" == "null" ]]; then
    echo "Failed to get access token."
    exit 1
fi

AUTH_HEADER="authorization: Bearer ${access_token}"

for bucket in "$BUCKET_SAVE_ARTIFACTS" "$BUCKET_SAVE_DATA" "$BUCKET_SAVE_DATA_INFERENCE" "$BUCKET_SAVE_MLFLOW"; do
    if ! bucket_exists "$bucket" "$REQUEST_ID" "$AUTH_HEADER"; then
        echo "Bucket '$bucket' does not exist. Creating bucket..."

        grpcurl -cacert certs/server.crt \
          -H "$AUTH_HEADER" \
          -d '{"request_id":"'"$REQUEST_ID"'","bucket_name":"'"$bucket"'"}' \
          "$GRPC_ADDR" \
          objectstorage.ObjectStorageService/CreateBucket
    else
        echo "Bucket '$bucket' already exists."
    fi
done

echo "Quickstart completed successfully."