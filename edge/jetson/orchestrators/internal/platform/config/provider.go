package config

import (
	"fmt"
	"strings"
)

// Provider composes config and environment sources. It is safe to reuse, but
// it intentionally reloads sources on each call so callers see current inputs.
type Provider struct {
	ConfigSources []ConfigSource
	EnvSources    []EnvSource
}

func (p Provider) Raw(section string) (map[string]any, error) {
	merged := map[string]any{}
	for _, source := range p.ConfigSources {
		loaded, err := source.Load()
		if err != nil {
			return nil, err
		}
		if loaded == nil {
			return nil, fmt.Errorf("config source %T returned nil", source)
		}
		merged = DeepMerge(merged, loaded).(map[string]any)
	}
	if section == "" {
		return merged, nil
	}
	current := any(merged)
	for _, part := range strings.Split(section, ".") {
		mapping, ok := current.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("config section %q must be a mapping", section)
		}
		var found bool
		current, found = mapping[part]
		if !found {
			return nil, fmt.Errorf("missing config section %q", section)
		}
	}
	selected, ok := current.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("config section %q must be a mapping", section)
	}
	return cloneValue(selected).(map[string]any), nil
}

// Require resolves placeholders and unmarshals the selected config into target,
// which must be a non-nil pointer to a struct, map, or other YAML-decodable value.
func (p Provider) Require(target any, section string) error {
	raw, err := p.Raw(section)
	if err != nil {
		return err
	}
	env, err := p.loadEnv()
	if err != nil {
		return err
	}
	resolved, err := ResolvePlaceholders(raw, env)
	if err != nil {
		return err
	}
	return Decode(resolved, target)
}

func (p Provider) loadEnv() (map[string]string, error) {
	merged := map[string]string{}
	for _, source := range p.EnvSources {
		loaded, err := source.Load()
		if err != nil {
			return nil, err
		}
		for key, value := range loaded {
			merged[key] = value
		}
	}
	return merged, nil
}

// DeepMerge combines mappings recursively. For all other types, override
// replaces base, including lists.
func DeepMerge(base, override any) any {
	baseMap, baseOK := base.(map[string]any)
	overrideMap, overrideOK := override.(map[string]any)
	if !baseOK || !overrideOK {
		return cloneValue(override)
	}
	merged := make(map[string]any, len(baseMap)+len(overrideMap))
	for key, value := range baseMap {
		merged[key] = cloneValue(value)
	}
	for key, value := range overrideMap {
		if existing, ok := merged[key]; ok {
			merged[key] = DeepMerge(existing, value)
		} else {
			merged[key] = cloneValue(value)
		}
	}
	return merged
}

func cloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		copy := make(map[string]any, len(typed))
		for key, item := range typed {
			copy[key] = cloneValue(item)
		}
		return copy
	case []any:
		copy := make([]any, len(typed))
		for index, item := range typed {
			copy[index] = cloneValue(item)
		}
		return copy
	default:
		return value
	}
}

func cloneStrings(values map[string]string) map[string]string {
	copy := make(map[string]string, len(values))
	for key, value := range values {
		copy[key] = value
	}
	return copy
}
