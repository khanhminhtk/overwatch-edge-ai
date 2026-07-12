from confluent_kafka import Producer as ConfluentKafkaProducer

from src.platform.logger import Logger
from src.platform.messaging.kafka.config import KafkaConfig
from src.platform.messaging.kafka.protocols import ConfluentProducer as ConfluentProducerProtocol


class KafkaProducerClient(ConfluentProducerProtocol):
    def __init__(
        self,
        config: KafkaConfig,
        logger: Logger,
    ):
        self._config = config
        self._logger = logger
        self._producer = ConfluentKafkaProducer(self._to_confluent_config(config))
        print(f"[PRODUCER_INITIALIZED] client_id={config.client_id_prefix}")
    
    @staticmethod
    def _to_confluent_config(config: KafkaConfig) -> dict[str, str | int | bool]:
        return {
            "bootstrap.servers": config.bootstrap_servers,
            "security.protocol": config.security_protocol,
            "client.id": config.client_id_prefix,
            "acks": config.producer_config.acks,
            "enable.idempotence": config.producer_config.enable_idempotence,
            "retries": config.producer_config.retries,
            "linger.ms": config.producer_config.linger_ms,
            "retry.backoff.ms": config.producer_config.retry_backoff_ms,
            "compression.type": config.producer_config.compression_type,
        }
    
    def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes | None,
        headers: list[tuple[str, bytes | None]] | None = None,
    ) -> None:
        try:
            self._producer.produce(
                topic=topic,
                key=key,
                value=value,
                headers=headers,
                on_delivery=self._delivery_report
            )
            self._producer.flush()
        except Exception as e:
            self._logger.error(f"[PRODUCER_ERROR] Failed to publish message to topic={topic}: {e}")
            raise

    def _delivery_report(self, err, msg) -> None:
        if err is not None:
            self._logger.error(f"[DELIVERY_REPORT_ERROR] Failed to deliver message to topic={msg.topic()}: {err}")
        else:
            self._logger.info(f"[DELIVERY_REPORT_SUCCESS] Message delivered to topic={msg.topic()} partition={msg.partition()} offset={msg.offset()}")

if __name__ == "__main__":
    import json
    import subprocess
    import time

    from src.platform.config import ConfigLoader

    pwd = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()

    config_loader = ConfigLoader.load(
        KafkaConfig,
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        section={
            "kafka": None,
            "kafka.defaults.consumer": "consumer_config",
            "kafka.defaults.producer": "producer_config",
        },
    )

    logger = Logger(name="KafkaProducerClient")
    producer = KafkaProducerClient(config=config_loader, logger=logger)

    # for event_type, model_name in [
    #     ("yolo_detector", "yolo_detector"),
    #     ("vit_ctc_deepseek", "vit_ctc_deepseek"),
    # ]:
    #     payload = {
    #         "request_id": f"test-{model_name}-{time.time_ns()}",
    #         "event_type": event_type,
    #         "payload": {
    #             "model_name": model_name,
    #             "model_version": "1",
    #             "git_commit": "abc123",
    #             "checkpoint_best_name": "best.pt",
    #             "checkpoint_last_name": "last.pt",
    #             "event": "tracking",
    #             "version": "1.0",
    #         },
    #     }
    #     producer.publish(
    #         topic="model.lifecycle.mlflow.tracking",
    #         key=model_name.encode("utf-8"),
    #         value=json.dumps(payload).encode("utf-8"),
    #         headers=[("content-type", b"application/json")],
    #     )
    #     print(f"Published tracking: event_type={event_type} request_id={payload['request_id']}")

    # print("\n--- Publishing export messages ---")
    # for event_type, model_name, checkpoint_best_name, checkpoint_last_name in [
    #     ("yolo_detector", "yolo_detector", "best.pt", "last.pt"),
    #     ("vit_ctc_deepseek", "vit_ctc_deepseek", "best_cer.pt", "last_checkpoint.pt"),
    # ]:
    #     payload = {
    #         "request_id": f"export-{model_name}-{time.time_ns()}",
    #         "event_type": event_type,
    #         "payload": {
    #             "model_name": model_name,
    #             "model_version": "1",
    #             "git_commit": "abc123",
    #             "checkpoint_best_name": checkpoint_best_name,
    #             "checkpoint_last_name": checkpoint_last_name,
    #             "event": "export_onnx",
    #             "version": "1.0",
    #         },
    #     }
    #     producer.publish(
    #         topic="model.lifecycle.export.onnx",
    #         key=model_name.encode("utf-8"),
    #         value=json.dumps(payload).encode("utf-8"),
    #         headers=[("content-type", b"application/json")],
    #     )
    #     print(f"Published export: event_type={event_type} request_id={payload['request_id']}")

    # print("\n--- Publishing training messages ---")
    # for event_type, dataset_version, mode in [
    #     ("train_requested_recognizer", "v1", "dev"),
    #     ("train_requested_detection", "v2", "dev"),
    # ]:
    #     training_payload = {"dataset_version": dataset_version, "mode": mode}
    #     payload = {
    #         "request_id": f"training-{time.time_ns()}",
    #         "event_type": event_type,
    #         "payload": training_payload,
    #     }
    #     producer.publish(
    #         topic="model.lifecycle.training",
    #         key=event_type.encode("utf-8"),
    #         value=json.dumps(payload).encode("utf-8"),
    #         headers=[("content-type", b"application/json")],
    #     )
    #     print(f"Published training: event_type={event_type} mode={mode} request_id={payload['request_id']}")

    print("\n--- Publishing dataset messages ---")
    for event_type, subset_percent in [
        ("dataset_requested_detection", 10.0),
        ("dataset_requested_recognizer", 15.0),
    ]:
        dataset_payload = {
            "source_data_path": f"/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data/{'detection' if 'detection' in event_type else 'recognizer'}",
            "subset_percent": subset_percent,
            "output_root": "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data",
            "dataset_version": "v1",
        }
        payload = {
            "request_id": f"dataset-{time.time_ns()}",
            "event_type": event_type,
            "payload": dataset_payload,
        }
        producer.publish(
            topic="model.lifecycle.dataset",
            key=event_type.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
            headers=[("content-type", b"application/json")],
        )
        print(f"Published dataset: event_type={event_type} subset_percent={subset_percent} request_id={payload['request_id']}")

    # print("\n--- Publishing download messages ---")
    # for event_type, model_name, checkpoint_best_name, checkpoint_last_name, download_dir in [
    #     ("yolo_detector", "yolo_detector", "best.pt", "last.pt", "data/download")
    #     # ("vit_ctc_deepseek", "vit_ctc_deepseek", "best_cer.pt", "last_checkpoint.pt", "data/download"),
    # ]:
    #     payload = {
    #         "request_id": f"download-{model_name}-{time.time_ns()}",
    #         "event_type": event_type,
    #         "payload": {
    #             "model_name": model_name,
    #             "model_version": "",
    #             "git_commit": "abc123",
    #             "checkpoint_best_name": checkpoint_best_name,
    #             "checkpoint_last_name": checkpoint_last_name,
    #             "output_path": f"{download_dir}/{model_name}/{checkpoint_best_name}",
    #             "event": "download_requested",
    #             "version": "1.0",
    #         },
    #     }
    #     producer.publish(
    #         topic="model.lifecycle.mlflow.download",
    #         key=model_name.encode("utf-8"),
    #         value=json.dumps(payload).encode("utf-8"),
    #         headers=[("content-type", b"application/json")],
    #     )
    #     print(f"Published download: event_type={event_type} request_id={payload['request_id']}")
