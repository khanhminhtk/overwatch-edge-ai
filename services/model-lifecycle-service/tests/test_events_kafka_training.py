import unittest

from src.applications.dtos.events.kafka_base import BaseKafkaEvent
from src.applications.dtos.events.events_kafka_training import (
    BaseTrainingEvent,
    KafkaTrainingEvent,
    PayloadTrainingJob,
    TrainRequestedEvent,
    TrainingCompletedEvent,
    TrainingFailedEvent,
)


class KafkaTrainingEventDTOTest(unittest.TestCase):
    def test_wraps_train_requested_payload_with_schema_aligned_fields(self) -> None:
        payload = TrainRequestedEvent(
            request_id="req-1",
            payload=PayloadTrainingJob(
                model_name="detector-a",
                dataset_version="dataset-v3",
                train_config_uri="s3://configs/train.yaml",
                target_server_id="edge-01",
            ),
        )

        event = KafkaTrainingEvent(
            request_id="req-1",
            topic="training.jobs",
            partition_id=3,
            message_offset=15,
            consumer_group="training-orchestrator",
            message_key="req-1",
            event_type=payload.event_type,
            schema_name="training_event",
            schema_version="1.0",
            payload=payload.payload.model_dump(mode="json"),
            produced_at=payload.created_at,
        )

        self.assertEqual(event.status, "RECEIVED")
        self.assertEqual(event.topic, "training.jobs")
        self.assertEqual(event.partition_id, 3)
        self.assertEqual(event.message_offset, 15)
        self.assertEqual(event.payload["model_name"], "detector-a")
        self.assertEqual(event.payload["dataset_version"], "dataset-v3")
        self.assertIsNone(event.error_message)
        self.assertIsNotNone(event.consumed_at)
        self.assertIsNotNone(event.created_at)
        self.assertIsNotNone(event.updated_at)

    def test_wraps_training_completed_payload_without_losing_business_fields(self) -> None:
        payload = TrainingCompletedEvent(
            request_id="req-2",
            model_version="model-v9",
            artifact_uri="s3://models/detector-b/model-v9",
            payload=PayloadTrainingJob(
                model_name="detector-b",
                dataset_version="dataset-v4",
                mlflow_run_id="run-123",
            ),
        )

        event = KafkaTrainingEvent(
            request_id="req-2",
            topic="training.results",
            partition_id=1,
            message_offset=27,
            consumer_group="deployment-handler",
            event_type=payload.event_type,
            payload=payload.payload.model_dump(mode="json"),
            status="PROCESSED",
            processed_at=payload.created_at,
        )

        self.assertEqual(event.status, "PROCESSED")
        self.assertEqual(event.payload["model_name"], "detector-b")
        self.assertEqual(event.payload["dataset_version"], "dataset-v4")
        self.assertEqual(event.payload["mlflow_run_id"], "run-123")
        self.assertEqual(event.event_type, "training_completed")

    def test_kafka_training_event_inherits_shared_kafka_envelope(self) -> None:
        event = KafkaTrainingEvent(
            request_id="req-3",
            topic="training.jobs",
            partition_id=0,
            message_offset=99,
            consumer_group="training-orchestrator",
            event_type="train_requested",
        )

        self.assertIsInstance(event, BaseKafkaEvent)
        self.assertEqual(event.status, "RECEIVED")
        self.assertIsNone(event.payload)

    def test_training_failed_event_inherits_shared_business_fields(self) -> None:
        payload = PayloadTrainingJob(
            model_name="detector-c",
            dataset_version="dataset-v5",
        )
        event = TrainingFailedEvent(
            request_id="req-4",
            error_message="boom",
            payload=payload,
        )

        self.assertIsInstance(event, BaseTrainingEvent)
        self.assertEqual(event.payload, payload)
        self.assertIsNotNone(event.created_at)


if __name__ == "__main__":
    unittest.main()
