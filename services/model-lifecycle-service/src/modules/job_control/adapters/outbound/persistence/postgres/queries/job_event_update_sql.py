from __future__ import annotations

HCM_NOW_SQL = "(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh')"


MARK_JOB_EVENT_PROCESSING = """
UPDATE kafka_events
SET
    status = 'PROCESSING',
    error_message = NULL,
    updated_at = """ + HCM_NOW_SQL + """
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


MARK_JOB_EVENT_PROCESSED = """
UPDATE kafka_events
SET
    status = 'PROCESSED',
    error_message = NULL,
    processed_at = """ + HCM_NOW_SQL + """,
    updated_at = """ + HCM_NOW_SQL + """
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


MARK_JOB_EVENT_FAILED = """
UPDATE kafka_events
SET
    status = 'FAILED',
    error_message = %(error_message)s,
    processed_at = """ + HCM_NOW_SQL + """,
    updated_at = """ + HCM_NOW_SQL + """
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


UPDATE_JOB_EVENT_STATUS = """
UPDATE kafka_events
SET
    status = %(status)s,
    error_message = %(error_message)s,
    processed_at = %(processed_at)s,
    updated_at = """ + HCM_NOW_SQL + """
WHERE topic = %(topic)s
  AND partition_id = %(partition_id)s
  AND message_offset = %(message_offset)s
  AND consumer_group = %(consumer_group)s
RETURNING id;
"""


UPDATE_JOB_EVENT_PROCESSED_BY_REQUEST_ID = """
UPDATE kafka_events
SET
    status = 'PROCESSED',
    error_message = NULL,
    processed_at = """ + HCM_NOW_SQL + """,
    updated_at = """ + HCM_NOW_SQL + """
WHERE request_id = %(request_id)s
RETURNING id;
"""


UPDATE_JOB_EVENT_FAILED_BY_REQUEST_ID = """
UPDATE kafka_events
SET
    status = 'FAILED',
    error_message = %(error_message)s,
    processed_at = """ + HCM_NOW_SQL + """,
    updated_at = """ + HCM_NOW_SQL + """
WHERE request_id = %(request_id)s
RETURNING id;
"""
