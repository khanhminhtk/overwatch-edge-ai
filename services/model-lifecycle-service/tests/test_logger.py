import io
import logging
import unittest

from src.utils.logger import Logger, LoggerConfig


class LoggerTest(unittest.TestCase):
    def tearDown(self) -> None:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(logging.NOTSET)

    def test_logger_instance_keeps_expected_name(self) -> None:
        logger = Logger("model_lifecycle_orchestrator.service")

        self.assertEqual(logger.name, "model_lifecycle_orchestrator.service")

    def test_configure_replaces_existing_handlers_without_duplicates(self) -> None:
        first_stream = io.StringIO()
        second_stream = io.StringIO()

        Logger.configure(LoggerConfig(stream=first_stream))
        Logger.configure(LoggerConfig(stream=second_stream))

        root_logger = logging.getLogger()

        self.assertEqual(len(root_logger.handlers), 1)
        self.assertIs(root_logger.handlers[0].stream, second_stream)

    def test_logger_writes_message_with_expected_level_and_format(self) -> None:
        stream = io.StringIO()
        Logger.configure(LoggerConfig(level="INFO", fmt="%(levelname)s|%(name)s|%(message)s", stream=stream))

        logger = Logger("orchestrator.worker")
        logger.debug("ignored")
        logger.info("started")

        self.assertEqual(stream.getvalue().strip(), "INFO|orchestrator.worker|started")

    def test_logger_joins_multiple_message_parts_without_percent_formatting(self) -> None:
        stream = io.StringIO()
        Logger.configure(LoggerConfig(level="INFO", fmt="%(message)s", stream=stream))

        logger = Logger("orchestrator.worker")
        logger.info("[JOB_CREATED]", "request_id=req-1", "model_name=recognizer")

        self.assertEqual(stream.getvalue().strip(), "[JOB_CREATED] request_id=req-1 model_name=recognizer")


if __name__ == "__main__":
    unittest.main()
