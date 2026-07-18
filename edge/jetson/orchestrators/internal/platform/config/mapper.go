package config

import (
	"encoding"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

var placeholderPattern = regexp.MustCompile(`\$\{([^}:]+)(:-([^}]*))?\}`)

// ResolvePlaceholders recursively expands ${NAME} and ${NAME:-default} using
// env. A missing required value returns an error that identifies its location.
func ResolvePlaceholders(value any, env map[string]string) (any, error) {
	return resolve(value, env, "")
}

func resolve(value any, env map[string]string, path string) (any, error) {
	switch typed := value.(type) {
	case map[string]any:
		resolved := make(map[string]any, len(typed))
		for key, item := range typed {
			itemPath := key
			if path != "" {
				itemPath = path + "." + key
			}
			result, err := resolve(item, env, itemPath)
			if err != nil {
				return nil, err
			}
			resolved[key] = result
		}
		return resolved, nil
	case []any:
		resolved := make([]any, len(typed))
		for index, item := range typed {
			result, err := resolve(item, env, fmt.Sprintf("%s[%d]", path, index))
			if err != nil {
				return nil, err
			}
			resolved[index] = result
		}
		return resolved, nil
	case string:
		var resolveErr error
		result := placeholderPattern.ReplaceAllStringFunc(typed, func(match string) string {
			parts := placeholderPattern.FindStringSubmatch(match)
			if replacement, ok := env[parts[1]]; ok {
				return replacement
			}
			if parts[3] != "" || len(parts) > 2 && parts[2] != "" {
				return parts[3]
			}
			resolveErr = fmt.Errorf("missing environment variable %q at %s", parts[1], displayPath(path))
			return match
		})
		if resolveErr != nil {
			return nil, resolveErr
		}
		return result, nil
	default:
		return value, nil
	}
}

func displayPath(path string) string {
	if path == "" {
		return "<root>"
	}
	return path
}

// Decode maps resolved configuration into target. Unlike yaml.Unmarshal, it
// deliberately coerces string environment values into scalar field types.
func Decode(value any, target any) error {
	destination := reflect.ValueOf(target)
	if destination.Kind() != reflect.Pointer || destination.IsNil() {
		return fmt.Errorf("config target must be a non-nil pointer, got %T", target)
	}
	return decodeValue(value, destination.Elem(), "")
}

var textUnmarshalerType = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()

func decodeValue(value any, destination reflect.Value, path string) error {
	if !destination.CanSet() {
		return fmt.Errorf("config field %s cannot be set", displayPath(path))
	}
	if destination.Kind() == reflect.Pointer {
		if value == nil {
			destination.SetZero()
			return nil
		}
		destination.Set(reflect.New(destination.Type().Elem()))
		return decodeValue(value, destination.Elem(), path)
	}
	if destination.CanAddr() && destination.Addr().Type().Implements(textUnmarshalerType) {
		text, ok := value.(string)
		if !ok {
			return typeError(path, destination.Type(), value)
		}
		if err := destination.Addr().Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(text)); err != nil {
			return fmt.Errorf("decode config field %s: %w", displayPath(path), err)
		}
		return nil
	}
	switch destination.Kind() {
	case reflect.Struct:
		mapping, ok := value.(map[string]any)
		if !ok {
			return typeError(path, destination.Type(), value)
		}
		typeOfDestination := destination.Type()
		for index := 0; index < destination.NumField(); index++ {
			field := typeOfDestination.Field(index)
			if field.PkgPath != "" {
				continue
			}
			name := strings.Split(field.Tag.Get("yaml"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = strings.ToLower(field.Name)
			}
			item, found := mapping[name]
			if !found {
				continue
			}
			fieldPath := name
			if path != "" {
				fieldPath = path + "." + name
			}
			if err := decodeValue(item, destination.Field(index), fieldPath); err != nil {
				return err
			}
		}
		return nil
	case reflect.Slice:
		items, ok := value.([]any)
		if !ok {
			return typeError(path, destination.Type(), value)
		}
		result := reflect.MakeSlice(destination.Type(), len(items), len(items))
		for index, item := range items {
			if err := decodeValue(item, result.Index(index), fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
		}
		destination.Set(result)
		return nil
	case reflect.Map:
		mapping, ok := value.(map[string]any)
		if !ok {
			return typeError(path, destination.Type(), value)
		}
		if destination.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("config map %s must use string keys", displayPath(path))
		}
		result := reflect.MakeMapWithSize(destination.Type(), len(mapping))
		for key, item := range mapping {
			entry := reflect.New(destination.Type().Elem()).Elem()
			if err := decodeValue(item, entry, path+"."+key); err != nil {
				return err
			}
			result.SetMapIndex(reflect.ValueOf(key).Convert(destination.Type().Key()), entry)
		}
		destination.Set(result)
		return nil
	case reflect.Interface:
		destination.Set(reflect.ValueOf(value))
		return nil
	case reflect.String:
		text, ok := value.(string)
		if !ok {
			return typeError(path, destination.Type(), value)
		}
		destination.SetString(text)
		return nil
	case reflect.Bool:
		if boolean, ok := value.(bool); ok {
			destination.SetBool(boolean)
			return nil
		}
		if text, ok := value.(string); ok {
			boolean, err := strconv.ParseBool(strings.ToLower(strings.TrimSpace(text)))
			if err == nil {
				destination.SetBool(boolean)
				return nil
			}
			switch strings.ToLower(strings.TrimSpace(text)) {
			case "yes", "on":
				destination.SetBool(true)
				return nil
			case "no", "off":
				destination.SetBool(false)
				return nil
			}
		}
		return typeError(path, destination.Type(), value)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		integer, err := int64Value(value)
		if err != nil {
			return typeError(path, destination.Type(), value)
		}
		destination.SetInt(integer)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		integer, err := uint64Value(value)
		if err != nil {
			return typeError(path, destination.Type(), value)
		}
		destination.SetUint(integer)
		return nil
	case reflect.Float32, reflect.Float64:
		floating, err := float64Value(value)
		if err != nil {
			return typeError(path, destination.Type(), value)
		}
		destination.SetFloat(floating)
		return nil
	default:
		return fmt.Errorf("unsupported config type %s at %s", destination.Type(), displayPath(path))
	}
}

func int64Value(value any) (int64, error) {
	switch typed := value.(type) {
	case int:
		return int64(typed), nil
	case int64:
		return typed, nil
	case string:
		return strconv.ParseInt(strings.TrimSpace(typed), 10, 64)
	default:
		return 0, fmt.Errorf("not an integer")
	}
}
func uint64Value(value any) (uint64, error) {
	switch typed := value.(type) {
	case int:
		if typed >= 0 {
			return uint64(typed), nil
		}
	case int64:
		if typed >= 0 {
			return uint64(typed), nil
		}
	case string:
		return strconv.ParseUint(strings.TrimSpace(typed), 10, 64)
	}
	return 0, fmt.Errorf("not an unsigned integer")
}
func float64Value(value any) (float64, error) {
	switch typed := value.(type) {
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	case float64:
		return typed, nil
	case string:
		return strconv.ParseFloat(strings.TrimSpace(typed), 64)
	default:
		return 0, fmt.Errorf("not a number")
	}
}
func typeError(path string, expected reflect.Type, actual any) error {
	return fmt.Errorf("invalid config type at %s: expected %s, got %T", displayPath(path), expected, actual)
}
