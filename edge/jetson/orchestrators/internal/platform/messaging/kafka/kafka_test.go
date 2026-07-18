package kafka

import (
	"context"
	"errors"
	"testing"

	segment "github.com/segmentio/kafka-go"
)

func TestMessageOperations(t *testing.T) {
	message := Message{Topic: "jobs", Partition: 1, Offset: 4, Value: []byte("body"), Headers: []Header{{Key: "kind", Value: []byte("a")}, {Key: "kind", Value: []byte("b")}}}
	if err := message.Validate(); err != nil {
		t.Fatal(err)
	}
	if message.NextOffset() != 5 || message.ValueSizeBytes() != 4 {
		t.Fatal("unexpected message properties")
	}
	if string(message.LastHeader("kind")) != "b" || len(message.HeaderValues("kind")) != 2 {
		t.Fatal("unexpected headers")
	}
}
func TestJobEventTypes(t *testing.T) {
	values := (JobConfig{EventType: "a, b, ,c"}).EventTypes()
	if len(values) != 3 || values[1] != "b" {
		t.Fatalf("unexpected event types: %#v", values)
	}
}
func TestConsumerCommitsOnlyAfterSuccessfulHandler(t *testing.T) {
	native := segment.Message{Topic: "jobs", Partition: 0, Offset: 1, Value: []byte("v")}
	reader := &fakeReader{message: native}
	consumer := &Consumer{reader: reader, handler: func(context.Context, Message) error { return errors.New("invalid") }}
	err := consumer.Run(context.Background())
	var processing *ProcessingError
	if !errors.As(err, &processing) {
		t.Fatalf("got %v", err)
	}
	if reader.commits != 0 {
		t.Fatal("message was committed after a handler error")
	}
}

type fakeReader struct {
	message segment.Message
	fetched bool
	commits int
}

func (r *fakeReader) FetchMessage(context.Context) (segment.Message, error) {
	if r.fetched {
		return segment.Message{}, context.Canceled
	}
	r.fetched = true
	return r.message, nil
}
func (r *fakeReader) CommitMessages(context.Context, ...segment.Message) error {
	r.commits++
	return nil
}
func (r *fakeReader) Close() error { return nil }
