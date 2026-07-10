from __future__ import annotations

from psycopg import sql

HCM_NOW_SQL = "(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh')"


CLAIM_NEXT_PENDING_JOB = sql.SQL("""
WITH candidate AS (
    SELECT e.id
    FROM kafka_events e
    WHERE e.event_type = {event_type}
        AND (
            e.status = 'RECEIVED'
            OR (e.status = 'PROCESSING' AND e.updated_at < %(stale_cutoff)s)
        )
        AND (
            COALESCE(e.payload->>'target_server_id', '') = ''
            OR e.payload->>'target_server_id' = %(server_id)s
        )
        AND NOT EXISTS (
            SELECT 1
            FROM kafka_events running
            WHERE running.event_type = {event_type}
            AND running.status = 'PROCESSING'
            AND running.updated_at >= %(stale_cutoff)s
            {additional_conditions}
        )
    ORDER BY e.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE kafka_events e
SET status = 'PROCESSING',
    updated_at = """ + HCM_NOW_SQL + """
FROM candidate
WHERE e.id = candidate.id
RETURNING
    {returning};
""")
