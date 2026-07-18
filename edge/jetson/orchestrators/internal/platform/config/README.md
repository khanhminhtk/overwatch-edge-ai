# Config

`config` is the Go counterpart to the Python config loader: source composition,
ordered YAML and `.env` loading, deep merge, placeholder resolution, and typed
struct decoding.

```go
type AppConfig struct {
	Server struct {
		Host string `yaml:"host"`
		Port int    `yaml:"port"`
	} `yaml:"server"`
}

cfg, err := config.Load[AppConfig](config.Options{
	YAMLFiles:    []string{"config/base.yaml", "config/edge.yaml"},
	EnvFiles:     []string{".env"},
	IncludeOSEnv: true,
	Env:          map[string]string{"PORT": "8080"},
})
```

Sources merge in order. Nested maps merge recursively; later scalar values and
lists replace earlier values. Environment sources likewise merge in order.
Placeholders support `${NAME}` and `${NAME:-default}`. Use `Provider` directly
when an application needs explicit custom sources or a selected `section`.
