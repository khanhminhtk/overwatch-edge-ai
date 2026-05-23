package utils

import (
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestGenerateOpaqueToken(t *testing.T) {
	token, err := GenerateOpaqueToken(32)
	if err != nil {
		t.Fatalf("GenerateOpaqueToken() error = %v", err)
	}
	if len(token) != 64 {
		t.Fatalf("expected token length 64, got %d", len(token))
	}
	if _, err := GenerateOpaqueToken(0); err == nil {
		t.Fatal("expected error when bytesLen <= 0")
	}
}

func TestGenerateAndParseAccessToken_Success(t *testing.T) {
	const (
		secret      = "test-secret"
		serviceName = "minio_service"
		ttl         = int64(900)
	)

	token, expiresIn, err := GenerateAccessToken(secret, ttl, serviceName)
	if err != nil {
		t.Fatalf("GenerateAccessToken() error = %v", err)
	}
	if token == "" {
		t.Fatal("expected non-empty token")
	}
	if expiresIn != ttl {
		t.Fatalf("expected expiresIn %d, got %d", ttl, expiresIn)
	}

	claims, err := ParseAndValidateAccessToken(token, secret)
	if err != nil {
		t.Fatalf("ParseAndValidateAccessToken() error = %v", err)
	}
	if claims.Type != tokenTypeAccess {
		t.Fatalf("expected type %q, got %q", tokenTypeAccess, claims.Type)
	}
	if claims.Issuer != serviceName || claims.Subject != serviceName {
		t.Fatalf("unexpected issuer/subject: issuer=%q subject=%q", claims.Issuer, claims.Subject)
	}
}

func TestGenerateAccessToken_ValidationErrors(t *testing.T) {
	if _, _, err := GenerateAccessToken("", 900, "svc"); err == nil {
		t.Fatal("expected error when secret is empty")
	}
	if _, _, err := GenerateAccessToken("secret", 0, "svc"); err == nil {
		t.Fatal("expected error when ttl <= 0")
	}
}

func TestParseAndValidateAccessToken_Errors(t *testing.T) {
	const secret = "test-secret"

	if _, err := ParseAndValidateAccessToken("", secret); err == nil {
		t.Fatal("expected error when token is empty")
	}
	if _, err := ParseAndValidateAccessToken("abc", ""); err == nil {
		t.Fatal("expected error when secret is empty")
	}

	validToken, _, err := GenerateAccessToken(secret, 900, "svc")
	if err != nil {
		t.Fatalf("GenerateAccessToken() error = %v", err)
	}
	if _, err := ParseAndValidateAccessToken(validToken, "wrong-secret"); err == nil {
		t.Fatal("expected error when secret is wrong")
	}

	expiredClaims := AccessTokenClaims{
		Type: tokenTypeAccess,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().UTC().Add(-1 * time.Minute)),
		},
	}
	expiredJWT := jwt.NewWithClaims(jwt.SigningMethodHS256, expiredClaims)
	expiredToken, err := expiredJWT.SignedString([]byte(secret))
	if err != nil {
		t.Fatalf("sign expired token failed: %v", err)
	}
	if _, err := ParseAndValidateAccessToken(expiredToken, secret); err == nil {
		t.Fatal("expected error for expired token")
	}

	wrongTypeClaims := AccessTokenClaims{
		Type: tokenTypeRefresh,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().UTC().Add(5 * time.Minute)),
		},
	}
	wrongTypeJWT := jwt.NewWithClaims(jwt.SigningMethodHS256, wrongTypeClaims)
	wrongTypeToken, err := wrongTypeJWT.SignedString([]byte(secret))
	if err != nil {
		t.Fatalf("sign wrong type token failed: %v", err)
	}
	if _, err := ParseAndValidateAccessToken(wrongTypeToken, secret); err == nil || !strings.Contains(err.Error(), "type") {
		t.Fatalf("expected token type error, got: %v", err)
	}
}

func TestValidateStaticRefreshToken(t *testing.T) {
	if err := ValidateStaticRefreshToken("refresh-abc", "refresh-abc"); err != nil {
		t.Fatalf("expected valid token, got error: %v", err)
	}
	if err := ValidateStaticRefreshToken(" refresh-abc ", "refresh-abc"); err != nil {
		t.Fatalf("expected valid token with spaces, got error: %v", err)
	}
	if err := ValidateStaticRefreshToken("", "refresh-abc"); err == nil {
		t.Fatal("expected error when provided token is empty")
	}
	if err := ValidateStaticRefreshToken("refresh-abc", ""); err == nil {
		t.Fatal("expected error when expected token is empty")
	}
	if err := ValidateStaticRefreshToken("refresh-abc", "refresh-def"); err == nil {
		t.Fatal("expected error when tokens mismatch")
	}
}
