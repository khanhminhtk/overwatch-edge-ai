package main

import (
	"context"
	"errors"
	"sync"

	"orchestrator/internal/modules/job_control/adapters/outbound/scheduler"
	platformlog "orchestrator/internal/platform/log"
	kafkaPkg "orchestrator/internal/platform/messaging/kafka"
)

// run starts all inbound consumers and polling workers, then stops every
// component after the first non-cancellation error or a shutdown signal.
func run(parent context.Context, consumers []*kafkaPkg.Consumer, workers []*scheduler.PollingRunner, logger *platformlog.Logger) error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	var waitGroup sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	report := func(err error) {
		if err == nil || errors.Is(err, context.Canceled) {
			return
		}
		errOnce.Do(func() {
			firstErr = err
			cancel()
		})
	}
	for _, consumer := range consumers {
		waitGroup.Add(1)
		go func(consumer *kafkaPkg.Consumer) { defer waitGroup.Done(); report(consumer.Run(ctx)) }(consumer)
	}
	for _, worker := range workers {
		waitGroup.Add(1)
		go func(worker *scheduler.PollingRunner) { defer waitGroup.Done(); report(worker.Run(ctx)) }(worker)
	}

	<-ctx.Done()
	for _, consumer := range consumers {
		consumer.Stop()
	}
	waitGroup.Wait()
	if firstErr != nil {
		return firstErr
	}
	logger.Info("[cmd.job_control.run] - Shutdown signal received.", "reason", parent.Err())
	return nil
}
