package kafka

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	segment "github.com/segmentio/kafka-go"
)

type Handler func(context.Context, Message) error
type reader interface {
	FetchMessage(context.Context) (segment.Message, error)
	CommitMessages(context.Context, ...segment.Message) error
	Close() error
}

// Consumer fetches one message at a time and commits only after its handler
// succeeds. Cancel the Run context or call Stop to end it.
type Consumer struct {
	reader  reader
	handler Handler
	logger  Logger
	cancel  context.CancelFunc
	mutex   sync.Mutex
	running bool
	closed  bool
}

func NewConsumer(config Config, groupID string, topics []string, handler Handler, logger Logger) (*Consumer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if groupID == "" {
		return nil, fmt.Errorf("Kafka group ID must not be empty")
	}
	if len(topics) == 0 {
		return nil, fmt.Errorf("at least one Kafka topic is required")
	}
	if handler == nil {
		return nil, fmt.Errorf("Kafka handler must not be nil")
	}
	reader := segment.NewReader(segment.ReaderConfig{Brokers: config.Brokers(), GroupID: groupID, GroupTopics: topics, Dialer: &segment.Dialer{ClientID: config.ClientIDPrefix, TLS: config.tlsConfig()}, MinBytes: 1, MaxBytes: 10e6, StartOffset: config.ConsumerConfig.startOffset(), SessionTimeout: time.Duration(config.ConsumerConfig.SessionTimeoutMS) * time.Millisecond, MaxWait: time.Second})
	return &Consumer{reader: reader, handler: handler, logger: logger}, nil
}
func (c *Consumer) Run(ctx context.Context) error {
	c.mutex.Lock()
	if c.running {
		c.mutex.Unlock()
		return fmt.Errorf("Kafka consumer is already running")
	}
	if c.closed {
		c.mutex.Unlock()
		return fmt.Errorf("Kafka consumer is closed")
	}
	ctx, c.cancel = context.WithCancel(ctx)
	c.running = true
	c.mutex.Unlock()
	defer func() { c.mutex.Lock(); c.running = false; c.closed = true; c.mutex.Unlock(); _ = c.reader.Close() }()
	if c.logger != nil {
		c.logger.Info("[KAFKA_CONSUMER_STARTED]")
	}
	for {
		native, err := c.reader.FetchMessage(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			}
			return fmt.Errorf("fetch Kafka message: %w", err)
		}
		message := fromNativeMessage(native)
		if err := c.handler(ctx, message); err != nil {
			return &ProcessingError{Message: message, Cause: err}
		}
		if err := c.reader.CommitMessages(ctx, native); err != nil {
			return fmt.Errorf("commit Kafka message: %w", err)
		}
	}
}
func (c *Consumer) Stop() {
	c.mutex.Lock()
	cancel := c.cancel
	c.mutex.Unlock()
	if cancel != nil {
		cancel()
	}
}
func fromNativeMessage(message segment.Message) Message {
	headers := make([]Header, len(message.Headers))
	for index, header := range message.Headers {
		headers[index] = Header{Key: header.Key, Value: header.Value}
	}
	return Message{Topic: message.Topic, Partition: message.Partition, Offset: message.Offset, Key: message.Key, Value: message.Value, Headers: headers, Timestamp: message.Time}
}
