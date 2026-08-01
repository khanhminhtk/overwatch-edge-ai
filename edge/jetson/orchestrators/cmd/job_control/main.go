package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	started := time.Now()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	configDir, err := resolveConfigDir()
	if err != nil {
		fail(err)
	}
	app, err := newApp(ctx, configDir)
	if err != nil {
		fail(err)
	}
	defer app.Close()

	app.Logger.Info("[cmd.job_control.main] - Job-control bootstrap completed; starting workers.",
		"config_dir", app.ConfigDir,
		"consumer_count", len(app.Consumers),
		"worker_count", len(app.Workers),
		"bootstrap_duration_ms", time.Since(started).Milliseconds(),
	)
	if err := run(ctx, app.Consumers, app.Workers, app.Logger); err != nil {
		app.Logger.Error("[cmd.job_control.main] - Job-control stopped because a worker failed.", "error", err, "uptime_ms", time.Since(started).Milliseconds())
		fail(err)
	}
	app.Logger.Info("[cmd.job_control.main] - Job-control stopped gracefully.", "uptime_ms", time.Since(started).Milliseconds())
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "job-control:", err)
	os.Exit(1)
}
