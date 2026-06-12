from __future__ import annotations

import argparse
import json
import uuid

from confluent_kafka import Producer


DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "model.lifecycle.export.recognizer.tensorrt"
DEFAULT_MODEL_NAME = "recognizer"
DEFAULT_MODEL_VERSION = "recognizer-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a test export_requested message to Kafka.")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request_id = args.request_id or f"manual-export-{uuid.uuid4().hex[:12]}"
    payload = {
        "event_type": "export_requested",
        "request_id": request_id,
        "payload": {
            "version": "1.0",
            "model_name": args.model_name,
            "model_version": args.model_version,
        },
    }

    producer = Producer(
        {
            "bootstrap.servers": args.bootstrap_servers,
            "client.id": "manual-export-tester",
            "security.protocol": "PLAINTEXT",
        }
    )
    producer.produce(
        topic=args.topic,
        key=request_id,
        value=json.dumps(payload).encode("utf-8"),
    )
    producer.flush()

    print(json.dumps({"topic": args.topic, "request_id": request_id, "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
