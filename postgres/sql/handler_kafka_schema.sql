CREATE TABLE IF NOT EXISTS kafka_events (
    id BIGSERIAL PRIMARY KEY,

    request_id VARCHAR(100) NOT NULL,

    topic VARCHAR(255) NOT NULL,
    partition_id INT NOT NULL,
    message_offset BIGINT NOT NULL,

    consumer_group VARCHAR(255) NOT NULL,
    message_key VARCHAR(255),

    event_type VARCHAR(255) NOT NULL,
    schema_name VARCHAR(255),
    schema_version VARCHAR(50),

    payload JSONB NULL,

    status VARCHAR(30) NOT NULL DEFAULT 'RECEIVED',
    error_message TEXT,

    produced_at TIMESTAMP NULL,
    consumed_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    processed_at TIMESTAMP NULL,

    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh'),

    CONSTRAINT uq_kafka_message UNIQUE (
        topic,
        partition_id,
        message_offset,
        consumer_group
    ),

    CONSTRAINT chk_kafka_status CHECK (
        status IN ('RECEIVED', 'PROCESSING', 'PROCESSED', 'FAILED')
    )
);

CREATE INDEX IF NOT EXISTS idx_kafka_events_request_id
ON kafka_events (request_id);

CREATE INDEX IF NOT EXISTS idx_kafka_events_topic_event_type
ON kafka_events (topic, event_type);

CREATE INDEX IF NOT EXISTS idx_kafka_events_consumer_group_status
ON kafka_events (consumer_group, status);

CREATE INDEX IF NOT EXISTS idx_kafka_events_created_at
ON kafka_events (created_at);
