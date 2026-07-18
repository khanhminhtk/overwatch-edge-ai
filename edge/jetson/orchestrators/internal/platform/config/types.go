package config

// ConfigSource supplies structured configuration. Implementations must return a
// mapping with string keys.
type ConfigSource interface {
	Load() (map[string]any, error)
}

// EnvSource supplies environment values used when resolving placeholders.
type EnvSource interface {
	Load() (map[string]string, error)
}
