from __future__ import annotations


INSERT_JOB_EVENT = """
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
ON CONFLICT
DO NOTHING
RETURNING id;
"""
