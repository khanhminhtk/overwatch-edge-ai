from psycopg import sql

INSERT_KAFKA_EVENT = """
INSERT INTO kafka_events (
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    produced_at
)
VALUES (
    %(request_id)s,
    %(topic)s,
    %(partition_id)s,
    %(message_offset)s,
    %(consumer_group)s,
    %(message_key)s,
    %(event_type)s,
    %(schema_name)s,
    %(schema_version)s,
    %(payload)s,
    %(status)s,
    %(produced_at)s
)
ON CONFLICT (
    topic,
    partition_id,
    message_offset,
    consumer_group
)
DO NOTHING
RETURNING id;
"""


SELECT_KAFKA_EVENT_BY_ID = """
SELECT
    id,
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    error_message,
    produced_at,
    consumed_at,
    processed_at,
    created_at,
    updated_at
FROM kafka_events
WHERE request_id = %(request_id)s;
"""


SELECT_KAFKA_EVENTS_BY_REQUEST_ID = """
SELECT
    id,
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    error_message,
    produced_at,
    consumed_at,
    processed_at,
    created_at,
    updated_at
FROM kafka_events
WHERE request_id = %(request_id)s
ORDER BY created_at ASC;
"""


SELECT_KAFKA_EVENT_BY_MESSAGE_IDENTITY = """
SELECT
    id,
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    error_message,
    produced_at,
    consumed_at,
    processed_at,
    created_at,
    updated_at
FROM kafka_events
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s;
"""




SELECT_KAFKA_EVENTS_BY_STATUS = """
SELECT
    id,
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    error_message,
    produced_at,
    consumed_at,
    processed_at,
    created_at,
    updated_at
FROM kafka_events
WHERE consumer_group = %(consumer_group)s
  AND status = %(status)s
ORDER BY created_at ASC
LIMIT %(limit)s;
"""


SELECT_KAFKA_EVENTS_BY_EVENT_TYPE = """
SELECT
    id,
    request_id,
    topic,
    partition_id,
    message_offset,
    consumer_group,
    message_key,
    event_type,
    schema_name,
    schema_version,
    payload,
    status,
    error_message,
    produced_at,
    consumed_at,
    processed_at,
    created_at,
    updated_at
FROM kafka_events
WHERE topic = %(topic)s
  AND event_type = %(event_type)s
ORDER BY created_at DESC
LIMIT %(limit)s;
"""


MARK_KAFKA_EVENT_PROCESSING = """
UPDATE kafka_events
SET
    status = 'PROCESSING',
    error_message = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


MARK_KAFKA_EVENT_PROCESSED = """
UPDATE kafka_events
SET
    status = 'PROCESSED',
    error_message = NULL,
    processed_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


MARK_KAFKA_EVENT_FAILED = """
UPDATE kafka_events
SET
    status = 'FAILED',
    error_message = %(error_message)s,
    processed_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


UPDATE_KAFKA_EVENT_STATUS = """
UPDATE kafka_events
SET
    status = %(status)s,
    error_message = %(error_message)s,
    processed_at = %(processed_at)s,
    updated_at = CURRENT_TIMESTAMP
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


UPDATE_KAFKA_EVENT_PROCESSED_BY_REQUEST_ID = """
UPDATE kafka_events
SET
    status = 'PROCESSED',
    error_message = NULL,
    processed_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE request_id = %(request_id)s
RETURNING id;
"""


UPDATE_KAFKA_EVENT_FAILED_BY_REQUEST_ID = """
UPDATE kafka_events
SET
    status = 'FAILED',
    error_message = %(error_message)s,
    processed_at = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP
WHERE request_id = %(request_id)s
RETURNING id;
"""


DELETE_KAFKA_EVENT_BY_REQUEST_ID = """
DELETE FROM kafka_events
WHERE request_id = %(request_id)s;
"""


COUNT_KAFKA_EVENTS_BY_REQUEST_ID = """
SELECT COUNT(*) AS total
FROM kafka_events
WHERE request_id = %(request_id)s;
"""


CLAIM_NEXT_PENDING_TRAINING_JOB = """
WITH candidate AS (
    SELECT e.id
    FROM kafka_events e
    WHERE e.event_type = 'train_requested'
      AND e.status = 'RECEIVED'
      AND (
          COALESCE(e.payload->>'target_server_id', '') = ''
          OR e.payload->>'target_server_id' = %(server_id)s
      )
      AND NOT EXISTS (
          SELECT 1
          FROM kafka_events running
          WHERE running.event_type = 'train_requested'
            AND running.status = 'PROCESSING'
            AND running.payload->>'model_name' = e.payload->>'model_name'
      )
    ORDER BY e.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE kafka_events e
SET status = 'PROCESSING',
    updated_at = CURRENT_TIMESTAMP
FROM candidate
WHERE e.id = candidate.id
RETURNING
    e.id,
    e.request_id,
    e.payload->>'model_name' AS model_name,
    e.payload->>'dataset_version' AS dataset_version,
    e.payload->>'train_config_uri' AS train_config_uri,
    e.payload->>'target_server_id' AS target_server_id,
    e.status;
"""


CLAIM_NEXT_PENDING_EXPORT_JOB = """
WITH candidate AS (
    SELECT e.id
    FROM kafka_events e
    WHERE e.event_type = 'export_requested'
      AND e.status = 'RECEIVED'
    ORDER BY e.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE kafka_events e
SET status = 'PROCESSING',
    updated_at = CURRENT_TIMESTAMP
FROM candidate
WHERE e.id = candidate.id
RETURNING
    e.id,
    e.request_id,
    e.payload->>'model_name' AS model_name,
    e.payload->>'model_version' AS model_version,
    e.status;
"""

CLAIM_NEXT_PENDING_JOB = sql.SQL("""
WITH candidate AS (
    SELECT e.id
    FROM kafka_events e
    WHERE e.event_type = {event_type}
      AND e.status = 'RECEIVED'
      AND (
          COALESCE(e.payload->>'target_server_id', '') = ''
          OR e.payload->>'target_server_id' = %(server_id)s
      )
      AND NOT EXISTS (
          SELECT 1
          FROM kafka_events running
          WHERE running.event_type = {event_type}
            AND running.status = 'PROCESSING'
            {additional_conditions}
      )
    ORDER BY e.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE kafka_events e
SET status = 'PROCESSING',
    updated_at = CURRENT_TIMESTAMP
FROM candidate
WHERE e.id = candidate.id
RETURNING
    {returning};
""")
