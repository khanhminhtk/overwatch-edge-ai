import subprocess
import unittest
from unittest.mock import Mock

from src.applications.use_cases.training_worker import TrainingWorker


class TrainingWorkerTest(unittest.TestCase):
    def test_process_next_job_marks_success_and_publishes_completed_event(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        worker = TrainingWorker(
            job_repository=repository,
            producer=producer,
            logger=logger,
            server_id="edge-01",
            event_topic="training.events",
            training_mode="local",
        )
        worker._get_next_job = Mock(
            return_value={
                "request_id": "req-1",
                "model_name": "recognizer",
                "dataset_version": "dataset-v1",
                "train_config_uri": "s3://cfg/train.yaml",
                "target_server_id": "edge-01",
            }
        )
        worker._run_job = Mock(
            return_value={
                "model_version": "recognizer-abcd1234",
                "artifact_uri": "s3://artifacts/recognizer/recognizer-abcd1234",
                "mlflow_run_id": "run-1",
            }
        )

        processed = worker.process_next_job()

        self.assertTrue(processed)
        repository.make_job_successful.assert_called_once_with("req-1")
        producer.publish.assert_called_once()
        published = producer.publish.call_args.kwargs["value"]
        self.assertEqual(published["event_type"], "training_completed")
        self.assertEqual(published["request_id"], "req-1")
        self.assertEqual(published["model_version"], "recognizer-abcd1234")

    def test_process_next_job_marks_failure_and_publishes_failed_event(self) -> None:
        repository = Mock()
        producer = Mock()
        logger = Mock()
        worker = TrainingWorker(
            job_repository=repository,
            producer=producer,
            logger=logger,
            server_id="edge-01",
            event_topic="training.events",
            training_mode="local",
        )
        worker._get_next_job = Mock(
            return_value={
                "request_id": "req-2",
                "model_name": "detector",
                "dataset_version": "dataset-v2",
                "train_config_uri": None,
                "target_server_id": "edge-01",
            }
        )
        worker._run_job = Mock(side_effect=RuntimeError("train failed"))

        processed = worker.process_next_job()

        self.assertTrue(processed)
        repository.make_job_failed.assert_called_once_with("req-2", "train failed")
        producer.publish.assert_called_once()
        published = producer.publish.call_args.kwargs["value"]
        self.assertEqual(published["event_type"], "training_failed")
        self.assertEqual(published["error_message"], "train failed")

    def test_process_next_job_returns_false_when_no_job_is_available(self) -> None:
        repository = Mock()
        worker = TrainingWorker(
            job_repository=repository,
            producer=Mock(),
            logger=Mock(),
            server_id="edge-01",
            event_topic="training.events",
            training_mode="local",
        )
        worker._get_next_job = Mock(return_value=None)

        processed = worker.process_next_job()

        self.assertFalse(processed)

    def test_run_job_executes_expected_script_for_recognizer(self) -> None:
        worker = TrainingWorker(
            job_repository=Mock(),
            producer=Mock(),
            logger=Mock(),
            server_id="edge-01",
            event_topic="training.events",
            training_mode="local",
        )
        run = Mock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
        worker._subprocess_run = run

        result = worker._run_job(
            {
                "request_id": "req-3",
                "model_name": "recognizer",
                "dataset_version": "dataset-v3",
            }
        )

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("ml/training/scripts/train_recognizer.sh"))
        self.assertEqual(command[2:], ["--mode", "local"])
        self.assertTrue(result["artifact_uri"].startswith("s3://artifacts/recognizer/"))

    def test_get_next_job_uses_train_requested_query(self) -> None:
        repository = Mock()
        repository.claim_next_pending_job.return_value = {"request_id": "req-4"}
        worker = TrainingWorker(
            job_repository=repository,
            producer=Mock(),
            logger=Mock(),
            server_id="edge-05",
            event_topic="training.events",
            training_mode="local",
        )

        result = worker._get_next_job()

        self.assertEqual(result, {"request_id": "req-4"})
        repository.claim_next_pending_job.assert_called_once()
        self.assertEqual(repository.claim_next_pending_job.call_args.kwargs["server_id"], "edge-05")
        query = str(repository.claim_next_pending_job.call_args.kwargs["query"])
        self.assertIn("train_requested", query)


if __name__ == "__main__":
    unittest.main()
