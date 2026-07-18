package kafka

import (
	"encoding/base64"
	"fmt"
	"strings"
	"time"
)

type Header struct {
	Key   string
	Value []byte
}
type MessageIdentity struct {
	Topic     string
	Partition int
	Offset    int64
}
type Message struct {
	Topic       string
	Partition   int
	Offset      int64
	Key         []byte
	Value       []byte
	Headers     []Header
	Timestamp   time.Time
	LeaderEpoch *int32
}

func (m Message) Validate() error {
	if strings.TrimSpace(m.Topic) == "" {
		return fmt.Errorf("Kafka topic must not be empty")
	}
	if m.Partition < 0 {
		return fmt.Errorf("Kafka partition must be non-negative: %d", m.Partition)
	}
	if m.Offset < 0 {
		return fmt.Errorf("Kafka offset must be non-negative: %d", m.Offset)
	}
	return nil
}
func (m Message) Identity() MessageIdentity {
	return MessageIdentity{Topic: m.Topic, Partition: m.Partition, Offset: m.Offset}
}
func (m Message) NextOffset() int64   { return m.Offset + 1 }
func (m Message) ValueSizeBytes() int { return len(m.Value) }
func (m Message) HeaderValues(name string) [][]byte {
	var values [][]byte
	for _, header := range m.Headers {
		if header.Key == name {
			values = append(values, header.Value)
		}
	}
	return values
}
func (m Message) LastHeader(name string) []byte {
	for index := len(m.Headers) - 1; index >= 0; index-- {
		if m.Headers[index].Key == name {
			return m.Headers[index].Value
		}
	}
	return nil
}
func (m Message) Metadata() map[string]any {
	encode := func(value []byte) any {
		if value == nil {
			return nil
		}
		return base64.StdEncoding.EncodeToString(value)
	}
	headers := make([]map[string]any, len(m.Headers))
	for index, header := range m.Headers {
		headers[index] = map[string]any{"key": header.Key, "value": encode(header.Value)}
	}
	return map[string]any{"topic": m.Topic, "partition": m.Partition, "offset": m.Offset, "key": encode(m.Key), "value": encode(m.Value), "headers": headers, "timestamp": m.Timestamp, "leader_epoch": m.LeaderEpoch}
}

type ProcessingError struct {
	Message Message
	Cause   error
}

func (e *ProcessingError) Error() string {
	return fmt.Sprintf("failed to process Kafka message topic=%s partition=%d offset=%d: %v", e.Message.Topic, e.Message.Partition, e.Message.Offset, e.Cause)
}
func (e *ProcessingError) Unwrap() error { return e.Cause }
