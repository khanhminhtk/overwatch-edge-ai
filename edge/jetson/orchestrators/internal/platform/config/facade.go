package config

// Options is the convenient source-based configuration setup for Load.
// Explicit sources take precedence over the corresponding file options.
type Options struct {
	YAMLFiles     []string
	EnvFiles      []string
	Env           map[string]string
	IncludeOSEnv  bool
	Section       string
	ConfigSources []ConfigSource
	EnvSources    []EnvSource
}

// Load reads configuration into T. Process environment is included only when
// IncludeOSEnv is true; this avoids hidden global input by default.
func Load[T any](options Options) (T, error) {
	var target T
	configSources := options.ConfigSources
	if len(configSources) == 0 && len(options.YAMLFiles) > 0 {
		configSources = []ConfigSource{YAMLConfigSource{Paths: options.YAMLFiles}}
	}
	envSources := options.EnvSources
	if len(envSources) == 0 {
		if len(options.EnvFiles) > 0 {
			envSources = append(envSources, DotenvEnvSource{Paths: options.EnvFiles})
		}
		if options.IncludeOSEnv {
			envSources = append(envSources, EnvironmentEnvSource{})
		}
		if options.Env != nil {
			envSources = append(envSources, EnvironmentEnvSource{Values: options.Env})
		}
	}
	provider := Provider{ConfigSources: configSources, EnvSources: envSources}
	return target, provider.Require(&target, options.Section)
}
