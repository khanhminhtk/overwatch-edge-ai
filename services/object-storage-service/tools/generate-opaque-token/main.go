package main

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
)

func GenerateOpaqueToken(bytesLen int) (string, error) {
	if bytesLen <= 0 {
		return "", errors.New("bytesLen must be > 0")
	}

	buf := make([]byte, bytesLen)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("rand.Read: %w", err)
	}
	return hex.EncodeToString(buf), nil
}

func main() {
	token, err := GenerateOpaqueToken(64)
	if err != nil {
		fmt.Printf("Error generating opaque token: %v\n", err)
		return
	}
	fmt.Printf("Generated opaque token: %s\n", token)
}