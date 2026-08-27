package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"web_ui/internal/application"
	"web_ui/internal/infrastructure/broadcast"
	"web_ui/internal/infrastructure/mockstream"
	httpui "web_ui/internal/interfaces/http"
)

const mockPublishInterval = 1200 * time.Millisecond

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	hub := broadcast.NewHub()
	publisher := application.NewPublishInferenceFrame(hub)
	generator := mockstream.NewGenerator(time.Now)

	go publishMockFrames(ctx, publisher, generator, mockPublishInterval)

	staticDir, err := resolveStaticDir()
	if err != nil {
		log.Fatalf("web_ui static dir resolution failed: %v", err)
	}

	staticHandler := http.FileServer(http.Dir(staticDir))
	router := httpui.NewRouter(staticHandler, publisher, hub)

	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	go func() {
		<-ctx.Done()

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Printf("web_ui shutdown failed: %v", err)
		}
	}()

	log.Printf("web_ui listening on %s", server.Addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("web_ui server failed: %v", err)
	}
}

func publishMockFrames(ctx context.Context, publisher *application.PublishInferenceFrame, generator *mockstream.Generator, interval time.Duration) {
	publishOnce := func() {
		if current, ok := publisher.Current(); ok && !mockstream.IsMockFrame(current) {
			return
		}

		frame, err := generator.Next()
		if err != nil {
			log.Printf("mock stream generation failed: %v", err)
			return
		}

		if err := publisher.Execute(ctx, frame); err != nil {
			log.Printf("mock stream publish failed: %v", err)
		}
	}

	publishOnce()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			publishOnce()
		}
	}
}

func resolveStaticDir() (string, error) {
	candidates := make([]string, 0, 5)

	if executablePath, err := os.Executable(); err == nil {
		executableDir := filepath.Dir(executablePath)
		candidates = append(candidates,
			filepath.Join(executableDir, "static"),
			filepath.Join(executableDir, "..", "..", "static"),
		)
	}

	if _, sourceFile, _, ok := runtime.Caller(0); ok {
		candidates = append(candidates, filepath.Join(filepath.Dir(sourceFile), "..", "..", "static"))
	}

	candidates = append(candidates, "./static", "./web_ui/static")

	for _, candidate := range candidates {
		resolved, err := filepath.Abs(candidate)
		if err != nil {
			continue
		}

		info, err := os.Stat(resolved)
		if err == nil && info.IsDir() {
			return resolved, nil
		}
	}

	return "", fmt.Errorf("static directory not found in candidates: %v", candidates)
}
