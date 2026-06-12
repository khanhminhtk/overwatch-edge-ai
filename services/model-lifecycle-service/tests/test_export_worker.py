import subprocess
import unittest
from unittest.mock import Mock

from src.applications.use_cases.export_worker import ExportWorker


class ExportWorkerTest(unittest.TestCase):
    def test_process_next_job_marks_success_and_publishes_completed_event(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        worker = ExportWorker(
            job_repository=repository,
            producer=producer,
            logger=logger,
            server_id="edge-01",
            event_topic="export.events",
            training_mode="local",
        )
        worker._get_next_job = Mock(
            return_value={
                "request_id": "req-1",
                "model_name": "recognizer",
                "model_version": "recognizer-v1",
            }
        )
        worker._run_job = Mock(
            return_value={
                "model_version": "recognizer-v1",
                "artifact_uri": "s3://artifacts/recognizer/recognizer-v1",
            }
        )

        processed = worker.process_next_job()

        self.assertTrue(processed)
        repository.make_job_successful.assert_called_once_with("req-1")
        producer.publish.assert_called_once()
        published = producer.publish.call_args.kwargs["value"]
        self.assertEqual(published["event_type"], "export_completed")
        self.assertEqual(published["request_id"], "req-1")
        self.assertEqual(published["model_version"], "recognizer-v1")

    def test_process_next_job_marks_failure_and_publishes_failed_event(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        worker = ExportWorker(
            job_repository=repository,
            producer=producer,
            logger=logger,
            server_id="edge-01",
            event_topic="export.events",
            training_mode="local",
        )
        worker._get_next_job = Mock(
            return_value={
                "request_id": "req-2",
                "model_name": "detector",
                "model_version": "detector-v2",
            }
        )
        worker._run_job = Mock(side_effect=RuntimeError("export failed"))

        processed = worker.process_next_job()

        self.assertTrue(processed)
        repository.make_job_failed.assert_called_once_with("req-2", "export failed")
        producer.publish.assert_called_once()
        published = producer.publish.call_args.kwargs["value"]
        self.assertEqual(published["event_type"], "export_failed")
        self.assertEqual(published["error_message"], "export failed")

    def test_run_job_executes_expected_script_for_detector(self) -> None:
        worker = ExportWorker(
            job_repository=Mock(),
            producer=Mock(),
            logger=Mock(),
            server_id="edge-01",
            event_topic="export.events",
            training_mode="docker",
        )
        run = Mock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
        worker._subprocess_run = run

        result = worker._run_job(
            {
                "request_id": "req-3",
                "model_name": "detector",
                "model_version": "detector-v3",
            }
        )

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("ml/training/scripts/export_yolo_tensorrt.sh"))
        self.assertEqual(command[2:], ["--mode", "docker"])
        self.assertEqual(result["model_version"], "detector-v3")

    def test_get_next_job_uses_export_requested_query(self) -> None:
        repository = Mock()
        repository.claim_next_pending_job.return_value = {"request_id": "req-4"}
        worker = ExportWorker(
            job_repository=repository,
            producer=Mock(),
            logger=Mock(),
            server_id="edge-09",
            event_topic="export.events",
            training_mode="local",
        )

        result = worker._get_next_job()

        self.assertEqual(result, {"request_id": "req-4"})
        repository.claim_next_pending_job.assert_called_once()
        self.assertEqual(repository.claim_next_pending_job.call_args.kwargs["server_id"], "edge-09")
        query = str(repository.claim_next_pending_job.call_args.kwargs["query"])
        self.assertIn("export_requested", query)


if __name__ == "__main__":
    unittest.main()
