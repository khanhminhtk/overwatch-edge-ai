import unittest
from unittest.mock import Mock

from src.applications.dtos.events.events_kafka_export import ExportRequestedEvent, PayloadExportJob
from src.applications.use_cases.export_job_repository import ExportJobRepository
from src.infra.queries.kafka_event_queries import (
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


class ExportJobRepositoryTest(unittest.TestCase):
    def test_create_job_if_not_exists_inserts_export_schema_event(self) -> None:
        sql_handler = _RecordingSQLHandler()
        logger = Mock()
        repository = ExportJobRepository(sql_handler=sql_handler, logger=logger)
        event = ExportRequestedEvent(
            request_id="req-1",
            payload=PayloadExportJob(
                model_name="recognizer",
                model_version="recognizer-v1",
            ),
        )

        repository.create_job_if_not_exists(
            event=event,
            topic="export.jobs",
            partition_id=2,
            message_offset=7,
            consumer_group="export-orchestrator",
        )

        self.assertEqual(len(sql_handler.calls), 2)
        insert_query, insert_params = sql_handler.calls[1]
        self.assertEqual(insert_query, INSERT_KAFKA_EVENT)
        self.assertEqual(insert_params["schema_name"], "export_event")
        self.assertEqual(insert_params["event_type"], "export_requested")
        self.assertEqual(insert_params["message_key"], "recognizer")
        self.assertEqual(insert_params["payload"]["model_version"], "recognizer-v1")

    def test_create_job_if_not_exists_accepts_dict_payload_from_validated_event(self) -> None:
        sql_handler = _RecordingSQLHandler()
        repository = ExportJobRepository(sql_handler=sql_handler, logger=Mock())
        event = ExportRequestedEvent.model_validate(
            {
                "event_type": "export_requested",
                "request_id": "req-2",
                "payload": {
                    "version": "1.0",
                    "model_name": "recognizer",
                    "model_version": "recognizer-v2",
                },
            }
        )

        repository.create_job_if_not_exists(
            event=event,
            topic="export.jobs",
            partition_id=3,
            message_offset=8,
            consumer_group="export-orchestrator",
        )

        insert_query, insert_params = sql_handler.calls[1]
        self.assertEqual(insert_query, INSERT_KAFKA_EVENT)
        self.assertEqual(insert_params["payload"]["model_name"], "recognizer")
        self.assertEqual(insert_params["payload"]["model_version"], "recognizer-v2")
        self.assertEqual(insert_params["schema_version"], "1.0")

    def test_make_job_status_updates_use_shared_queries(self) -> None:
        sql_handler = _RecordingSQLHandler()
        repository = ExportJobRepository(sql_handler=sql_handler, logger=Mock())

        repository.make_job_successful("req-success")
        repository.make_job_failed("req-failed", "boom")

        self.assertEqual(
            sql_handler.calls,
            [
                (
                    UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID,
                    {"request_id": "req-success"},
                ),
                (
                    UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID,
                    {"request_id": "req-failed", "error_message": "boom"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
