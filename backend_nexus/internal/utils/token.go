package utils

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)



type AccessTokenClaims struct {
	Type string `json:"type"`
	jwt.RegisteredClaims
}

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

func GenerateAccessToken(secret string, expiresInSeconds int64, serviceName string) (string, int64, error) {
	trimmedSecret := strings.TrimSpace(secret)
	if trimmedSecret == "" {
		return "", 0, errors.New("jwt secret is required")
	}
	if expiresInSeconds <= 0 {
		return "", 0, errors.New("expiresInSeconds must be > 0")
	}

	now := time.Now().UTC()
	expiresAt := now.Add(time.Duration(expiresInSeconds) * time.Second)

	claims := AccessTokenClaims{
		Type: tokenTypeAccess,
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    serviceName,
			Subject:   serviceName,
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(expiresAt),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString([]byte(trimmedSecret))
	if err != nil {
		return "", 0, fmt.Errorf("sign jwt: %w", err)
	}

	return signed, expiresInSeconds, nil
}

func ParseAndValidateAccessToken(accessToken, secret string) (*AccessTokenClaims, error) {
	trimmedToken := strings.TrimSpace(accessToken)
	trimmedSecret := strings.TrimSpace(secret)
	if trimmedToken == "" {
		return nil, errors.New("access token is required")
	}
	if trimmedSecret == "" {
		return nil, errors.New("jwt secret is required")
	}

	claims := &AccessTokenClaims{}
	parsed, err := jwt.ParseWithClaims(trimmedToken, claims, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(trimmedSecret), nil
	})
	if err != nil {
		return nil, err
	}
	if !parsed.Valid {
		return nil, errors.New("invalid access token")
	}
	if claims.Type != tokenTypeAccess {
		return nil, errors.New("invalid access token type")
	}

	return claims, nil
}

func ValidateStaticRefreshToken(providedToken, expectedToken string) error {
	p := strings.TrimSpace(providedToken)
	e := strings.TrimSpace(expectedToken)

	if p == "" {
		return errors.New("refresh token is required")
	}
	if e == "" {
		return errors.New("server refresh token is not configured")
	}

	pHash := sha256.Sum256([]byte(p))
	eHash := sha256.Sum256([]byte(e))
	if !hmac.Equal(pHash[:], eHash[:]) {
		return errors.New("refresh token is invalid")
	}
	return nil
}
