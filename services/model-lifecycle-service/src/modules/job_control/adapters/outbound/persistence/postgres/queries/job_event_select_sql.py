from __future__ import annotations


_BASE_JOB_EVENT_SELECT = """
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
"""


SELECT_JOB_EVENTS_BY_REQUEST_ID = (
    _BASE_JOB_EVENT_SELECT
    + """
WHERE request_id = %(request_id)s
ORDER BY created_at ASC;
"""
)


SELECT_JOB_EVENT_BY_MESSAGE_IDENTITY = (
    _BASE_JOB_EVENT_SELECT
    + """
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s;
"""
)


SELECT_JOB_EVENTS_BY_STATUS = (
    _BASE_JOB_EVENT_SELECT
    + """
WHERE consumer_group = %(consumer_group)s
  AND status = %(status)s
ORDER BY created_at ASC
LIMIT %(limit)s;
"""
)


SELECT_JOB_EVENTS_BY_EVENT_TYPE = (
    _BASE_JOB_EVENT_SELECT
    + """
WHERE topic = %(topic)s
  AND event_type = %(event_type)s
ORDER BY created_at DESC
LIMIT %(limit)s;
"""
)
