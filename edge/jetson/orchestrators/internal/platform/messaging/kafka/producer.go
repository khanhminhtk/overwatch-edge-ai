package kafka

import (
	"context"
	"fmt"

	segment "github.com/segmentio/kafka-go"
)

type Logger interface {
	Info(string, ...any)
	Error(string, ...any)
}
type writer interface {
	WriteMessages(context.Context, ...segment.Message) error
	Close() error
}

// Producer publishes Kafka messages synchronously; a successful return means
// the configured writer acknowledgement has been received.
type Producer struct {
	writer writer
	logger Logger
}

func NewProducer(config Config, logger Logger) (*Producer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	writer := &segment.Writer{Addr: segment.TCP(config.Brokers()...), Balancer: &segment.LeastBytes{}, RequiredAcks: requiredAcks(config.ProducerConfig.Acks), Async: false, BatchTimeout: config.ProducerConfig.batchTimeout(), MaxAttempts: config.ProducerConfig.Retries + 1, Transport: &segment.Transport{ClientID: config.ClientIDPrefix, TLS: config.tlsConfig()}}
	return &Producer{writer: writer, logger: logger}, nil
}
func (p *Producer) Publish(ctx context.Context, message Message) error {
	if err := message.Validate(); err != nil {
		return err
	}
	native := segment.Message{Topic: message.Topic, Key: message.Key, Value: message.Value, Time: message.Timestamp, Headers: toNativeHeaders(message.Headers)}
	if err := p.writer.WriteMessages(ctx, native); err != nil {
		if p.logger != nil {
			p.logger.Error("[KAFKA_PRODUCER_ERROR]", "topic", message.Topic, "error", err)
		}
		return fmt.Errorf("publish Kafka message to %q: %w", message.Topic, err)
	}
	if p.logger != nil {
		p.logger.Info("[KAFKA_MESSAGE_PUBLISHED]", "topic", message.Topic)
	}
	return nil
}
func (p *Producer) Close() error { return p.writer.Close() }
func requiredAcks(acks string) segment.RequiredAcks {
	switch acks {
	case "0":
		return segment.RequireNone
	case "1":
		return segment.RequireOne
	default:
		return segment.RequireAll
	}
}
func toNativeHeaders(headers []Header) []segment.Header {
	result := make([]segment.Header, len(headers))
	for index, header := range headers {
		result[index] = segment.Header{Key: header.Key, Value: header.Value}
	}
	return result
}
