import unittest
from unittest.mock import Mock

from src.applications.dtos.events.events_kafka_training import PayloadTrainingJob, TrainRequestedEvent
from src.applications.use_cases.training_job_repository import TrainingJobRepository
from src.infra.queries.kafka_event_queries import (
    CLAIM_NEXT_PENDING_TRAINING_JOB,
    INSERT_KAFKA_EVENT,
    SELECT_KAFKA_EVENTS_BY_REQUEST_ID,
    UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID,
    UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID,
)


class _RecordingSQLHandler:
    def __init__(self, existing_events=None):
        self.existing_events = existing_events or []
        self.calls = []

    def execute_query(self, query: str, params=None):
        self.calls.append((query, params))
        if query == SELECT_KAFKA_EVENTS_BY_REQUEST_ID:
            return self.existing_events
        return None


class TrainingJobRepositoryTest(unittest.TestCase):
    def test_create_job_if_not_exists_inserts_schema_aligned_kafka_event(self) -> None:
        sql_handler = _RecordingSQLHandler()
        logger = Mock()
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=logger)
        event = TrainRequestedEvent(
            request_id="req-1",
            payload=PayloadTrainingJob(
                model_name="detector-a",
                dataset_version="dataset-v1",
                train_config_uri="s3://cfgs/train.yaml",
                target_server_id="edge-01",
            ),
        )

        repository.create_job_if_not_exists(
            event=event,
            topic="training.jobs",
            partition_id=5,
            message_offset=42,
            consumer_group="training-orchestrator",
        )

        self.assertEqual(len(sql_handler.calls), 2)
        select_query, select_params = sql_handler.calls[0]
        self.assertEqual(select_query, SELECT_KAFKA_EVENTS_BY_REQUEST_ID)
        self.assertEqual(select_params, {"request_id": "req-1"})

        insert_query, insert_params = sql_handler.calls[1]
        self.assertEqual(insert_query, INSERT_KAFKA_EVENT)
        self.assertEqual(insert_params["request_id"], "req-1")
        self.assertEqual(insert_params["topic"], "training.jobs")
        self.assertEqual(insert_params["partition_id"], 5)
        self.assertEqual(insert_params["message_offset"], 42)
        self.assertEqual(insert_params["consumer_group"], "training-orchestrator")
        self.assertEqual(insert_params["message_key"], "detector-a")
        self.assertEqual(insert_params["event_type"], "train_requested")
        self.assertEqual(insert_params["schema_name"], "training_event")
        self.assertEqual(insert_params["schema_version"], "1.0")
        self.assertEqual(insert_params["status"], "RECEIVED")
        self.assertEqual(insert_params["payload"]["model_name"], "detector-a")
        self.assertEqual(insert_params["payload"]["dataset_version"], "dataset-v1")
        self.assertEqual(insert_params["payload"]["train_config_uri"], "s3://cfgs/train.yaml")
        self.assertEqual(insert_params["produced_at"], event.created_at)

    def test_create_job_if_not_exists_accepts_dict_payload_from_validated_event(self) -> None:
        sql_handler = _RecordingSQLHandler()
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=Mock())
        event = TrainRequestedEvent.model_validate(
            {
                "event_type": "train_requested",
                "request_id": "req-dict",
                "payload": {
                    "version": "1.0",
                    "model_name": "detector-a",
                    "dataset_version": "dataset-v7",
                    "train_config_uri": "s3://cfgs/train.yaml",
                    "target_server_id": "edge-01",
                },
            }
        )

        repository.create_job_if_not_exists(
            event=event,
            topic="training.jobs",
            partition_id=6,
            message_offset=43,
            consumer_group="training-orchestrator",
        )

        insert_query, insert_params = sql_handler.calls[1]
        self.assertEqual(insert_query, INSERT_KAFKA_EVENT)
        self.assertEqual(insert_params["payload"]["model_name"], "detector-a")
        self.assertEqual(insert_params["payload"]["dataset_version"], "dataset-v7")
        self.assertEqual(insert_params["schema_version"], "1.0")

    def test_create_job_if_not_exists_skips_insert_when_request_id_exists(self) -> None:
        sql_handler = _RecordingSQLHandler(existing_events=[{"id": 1}])
        logger = Mock()
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=logger)
        event = TrainRequestedEvent(
            request_id="req-2",
            payload=PayloadTrainingJob(
                model_name="detector-b",
                dataset_version="dataset-v2",
            ),
        )

        repository.create_job_if_not_exists(
            event=event,
            topic="training.jobs",
            partition_id=1,
            message_offset=2,
            consumer_group="training-orchestrator",
        )

        self.assertEqual(len(sql_handler.calls), 1)
        logger.info.assert_called_once()

    def test_claim_next_pending_job_returns_first_row(self) -> None:
        expected_row = (
            7,
            "req-claim",
            "detector-c",
            "dataset-v9",
            "s3://cfg/train.yaml",
            "edge-02",
            "PROCESSING",
        )
        sql_handler = _RecordingSQLHandler()
        sql_handler.execute_query = Mock(return_value=[expected_row])
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=Mock())

        result = repository.claim_next_pending_job(server_id="edge-02")

        self.assertEqual(
            result,
            {
                "id": 7,
                "request_id": "req-claim",
                "model_name": "detector-c",
                "dataset_version": "dataset-v9",
                "train_config_uri": "s3://cfg/train.yaml",
                "target_server_id": "edge-02",
                "status": "PROCESSING",
            },
        )
        sql_handler.execute_query.assert_called_once_with(
            CLAIM_NEXT_PENDING_TRAINING_JOB,
            params={"server_id": "edge-02"},
        )

    def test_make_job_successful_updates_processed_status_by_request_id(self) -> None:
        sql_handler = _RecordingSQLHandler()
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=Mock())

        repository.make_job_successful("req-success")

        self.assertEqual(
            sql_handler.calls[0],
            (
                UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID,
                {"request_id": "req-success"},
            ),
        )

    def test_make_job_failed_updates_failed_status_and_error_by_request_id(self) -> None:
        sql_handler = _RecordingSQLHandler()
        repository = TrainingJobRepository(sql_handler=sql_handler, logger=Mock())

        repository.make_job_failed("req-failed", "boom")

        self.assertEqual(
            sql_handler.calls[0],
            (
                UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID,
                {"request_id": "req-failed", "error_message": "boom"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
