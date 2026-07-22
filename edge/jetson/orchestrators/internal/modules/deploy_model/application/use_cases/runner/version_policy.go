package runner

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

var versionPolicyStart = regexp.MustCompile(`(?m)^[\t ]*version_policy[\t ]*\{`)

func updateVersionPolicy(configPath, version string) error {
	info, err := os.Stat(configPath)
	if err != nil {
		return fmt.Errorf("stat triton config %q: %w", configPath, err)
	}
	original, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("read triton config %q: %w", configPath, err)
	}
	updated, err := replaceVersionPolicy(string(original), version)
	if err != nil {
		return fmt.Errorf("update version policy in %q: %w", configPath, err)
	}

	backupPath := configPath + ".tmp"
	if _, err := os.Stat(backupPath); err == nil {
		return fmt.Errorf("triton config backup already exists: %s", backupPath)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("stat triton config backup %q: %w", backupPath, err)
	}
	if err := os.Rename(configPath, backupPath); err != nil {
		return fmt.Errorf("backup triton config to %q: %w", backupPath, err)
	}
	temporaryPath := configPath + ".new"
	if err := os.WriteFile(temporaryPath, []byte(updated), info.Mode().Perm()); err != nil {
		return fmt.Errorf("write updated triton config %q: %w", configPath, err)
	}
	if err := os.Rename(temporaryPath, configPath); err != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("activate updated triton config %q: %w", configPath, err)
	}
	return nil
}

func rollbackVersionPolicy(configPath string) error {
	backupPath := configPath + ".tmp"
	if _, err := os.Stat(backupPath); err != nil {
		return fmt.Errorf("stat triton config backup %q: %w", backupPath, err)
	}
	if err := os.Rename(backupPath, configPath); err != nil {
		return fmt.Errorf("restore triton config from %q: %w", backupPath, err)
	}
	return nil
}

func replaceVersionPolicy(config, version string) (string, error) {
	match := versionPolicyStart.FindStringIndex(config)
	policy := fmt.Sprintf("version_policy {\n  specific {\n    versions: [%s]\n  }\n}\n", version)
	if match == nil {
		return strings.TrimRight(config, "\n") + "\n\n" + policy, nil
	}

	openBrace := strings.Index(config[match[0]:match[1]], "{") + match[0]
	depth := 0
	for index := openBrace; index < len(config); index++ {
		switch config[index] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return config[:match[0]] + policy + config[index+1:], nil
			}
		}
	}
	return "", fmt.Errorf("unterminated version_policy block")
}
