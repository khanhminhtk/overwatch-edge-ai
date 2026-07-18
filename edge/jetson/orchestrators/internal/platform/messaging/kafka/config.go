package kafka

import (
	"crypto/tls"
	"fmt"
	"strings"
	"time"
)

type ConsumerConfig struct {
	AutoOffsetReset   string `yaml:"auto_offset_reset"`
	EnableAutoCommit  bool   `yaml:"enable_auto_commit"`
	SessionTimeoutMS  int    `yaml:"session_timeout_ms"`
	MaxPollIntervalMS int    `yaml:"max_poll_interval_ms"`
}

func DefaultConsumerConfig() ConsumerConfig {
	return ConsumerConfig{AutoOffsetReset: "latest", SessionTimeoutMS: 45000, MaxPollIntervalMS: 300000}
}

type ProducerConfig struct {
	Acks              string `yaml:"acks"`
	EnableIdempotence bool   `yaml:"enable_idempotence"`
	Retries           int    `yaml:"retries"`
	LingerMS          int    `yaml:"linger_ms"`
	RetryBackoffMS    int    `yaml:"retry_backoff_ms"`
	CompressionType   string `yaml:"compression_type"`
}

func DefaultProducerConfig() ProducerConfig {
	return ProducerConfig{Acks: "all", EnableIdempotence: true, Retries: 3, LingerMS: 5, RetryBackoffMS: 1000, CompressionType: "snappy"}
}

type JobConfig struct {
	Topic     string          `yaml:"topic"`
	GroupID   string          `yaml:"group_id"`
	EventType string          `yaml:"event_type"`
	Consumer  *ConsumerConfig `yaml:"consumer"`
	Producer  *ProducerConfig `yaml:"producer"`
}

func (j JobConfig) EventTypes() []string {
	values := strings.Split(j.EventType, ",")
	result := make([]string, 0, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			result = append(result, value)
		}
	}
	return result
}

type Config struct {
	BootstrapServers string               `yaml:"bootstrap_servers"`
	SecurityProtocol string               `yaml:"security_protocol"`
	ClientIDPrefix   string               `yaml:"client_id_prefix"`
	ConsumerConfig   ConsumerConfig       `yaml:"consumer_config"`
	ProducerConfig   ProducerConfig       `yaml:"producer_config"`
	Jobs             map[string]JobConfig `yaml:"jobs"`
}

func (c Config) Brokers() []string {
	var brokers []string
	for _, broker := range strings.Split(c.BootstrapServers, ",") {
		if broker = strings.TrimSpace(broker); broker != "" {
			brokers = append(brokers, broker)
		}
	}
	return brokers
}
func (c Config) Validate() error {
	if len(c.Brokers()) == 0 {
		return fmt.Errorf("Kafka bootstrap servers must not be empty")
	}
	if strings.TrimSpace(c.ClientIDPrefix) == "" {
		return fmt.Errorf("Kafka client ID prefix must not be empty")
	}
	switch strings.ToUpper(strings.TrimSpace(c.SecurityProtocol)) {
	case "PLAINTEXT", "SSL":
		return nil
	case "SASL_PLAINTEXT", "SASL_SSL":
		return fmt.Errorf("Kafka security protocol %q requires SASL credentials, which are not configured", c.SecurityProtocol)
	default:
		return fmt.Errorf("unsupported Kafka security protocol %q", c.SecurityProtocol)
	}
}

func (c Config) tlsConfig() *tls.Config {
	if strings.EqualFold(c.SecurityProtocol, "SSL") {
		return &tls.Config{MinVersion: tls.VersionTLS12}
	}
	return nil
}
func (c ConsumerConfig) startOffset() int64 {
	if c.AutoOffsetReset == "earliest" {
		return -2
	}
	return -1
}
func (c ProducerConfig) batchTimeout() time.Duration {
	return time.Duration(c.LingerMS) * time.Millisecond
}
