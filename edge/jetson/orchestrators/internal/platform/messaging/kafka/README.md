# Kafka messaging

This package is the Go counterpart of the model-lifecycle Kafka platform. It
uses the pure-Go `segmentio/kafka-go` client, so it does not require Confluent's
native client library.

```go
producer, err := kafka.NewProducer(cfg, logger)
if err != nil { return err }
defer producer.Close()

err = producer.Publish(ctx, kafka.Message{
	Topic: "model.lifecycle.training", Key: []byte("train"), Value: payload,
})
```

`Consumer.Run` calls the handler with a context and commits a message only
after the handler returns nil. Handler failures are returned as
`*ProcessingError`; cancel the run context or call `Stop` to shut down.
