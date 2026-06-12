import unittest

from psycopg import sql

from src.infra.queries.kafka_event_queries import CLAIM_NEXT_PENDING_JOB, CLAIM_NEXT_PENDING_TRAINING_JOB


class KafkaEventQueriesTest(unittest.TestCase):
    def test_claim_next_pending_training_job_query_contains_expected_locking_and_filters(self) -> None:
        self.assertIn("WITH candidate AS (", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("FROM kafka_events e", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("e.event_type = 'train_requested'", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("e.status = 'RECEIVED'", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("e.payload->>'target_server_id'", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("%(server_id)s", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("running.payload->>'model_name' = e.payload->>'model_name'", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("running.status = 'PROCESSING'", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("FOR UPDATE SKIP LOCKED", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("SET status = 'PROCESSING',", CLAIM_NEXT_PENDING_TRAINING_JOB)
        self.assertIn("e.payload->>'model_name' AS model_name", CLAIM_NEXT_PENDING_TRAINING_JOB)

    def test_claim_next_pending_job_template_formats_event_type_once(self) -> None:
        query = CLAIM_NEXT_PENDING_JOB.format(
            event_type=sql.Literal("export_requested"),
            additional_conditions=sql.SQL(""),
            returning=sql.SQL("e.id"),
        )

        rendered = str(query)
        self.assertEqual(rendered.count("Literal('export_requested')"), 2)
        self.assertNotIn("''export_requested''", rendered)


if __name__ == "__main__":
    unittest.main()
