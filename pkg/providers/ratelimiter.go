package providers

import (
	"context"
	"time"

	"golang.org/x/time/rate"
)

// RateLimitedProvider wraps an LLMProvider with per-model RPM and concurrency limits.
// Both limits are optional — nil limiter or nil semaphore means unlimited.
// This is a transparent decorator: it does not affect CooldownTracker or FallbackChain logic.
type RateLimitedProvider struct {
	inner         LLMProvider
	rpmLimiter    *rate.Limiter // token bucket for Requests Per Minute; nil = unlimited
	semaphore     chan struct{}  // buffered channel acting as counting semaphore; nil = unlimited
}

// NewRateLimitedProvider wraps provider with RPM and/or concurrency limits.
//   - rpm <= 0 means no per-minute rate limit.
//   - maxConcurrent <= 0 means no concurrency limit.
//
// If both are 0/negative, the original provider is returned as-is (no wrapping overhead).
func NewRateLimitedProvider(provider LLMProvider, rpm, maxConcurrent int) LLMProvider {
	if rpm <= 0 && maxConcurrent <= 0 {
		return provider
	}

	p := &RateLimitedProvider{inner: provider}

	if rpm > 0 {
		// Burst = rpm allows the full quota to be consumed instantly at start
		// (matches standard token-bucket semantics for API rate limits).
		p.rpmLimiter = rate.NewLimiter(rate.Every(time.Minute/time.Duration(rpm)), rpm)
	}

	if maxConcurrent > 0 {
		p.semaphore = make(chan struct{}, maxConcurrent)
	}

	return p
}

// Chat implements LLMProvider.Chat with rate limiting applied before the inner call.
// Order: concurrency acquire → RPM wait → inner Chat → concurrency release.
func (p *RateLimitedProvider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	// 1. Acquire concurrency slot (blocks if max concurrent requests already in flight).
	if p.semaphore != nil {
		select {
		case p.semaphore <- struct{}{}:
			defer func() { <-p.semaphore }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// 2. Wait for RPM token bucket allowance.
	if p.rpmLimiter != nil {
		if err := p.rpmLimiter.Wait(ctx); err != nil {
			return nil, err
		}
	}

	// 3. Delegate to the underlying provider.
	return p.inner.Chat(ctx, messages, tools, model, options)
}

// GetDefaultModel delegates to the inner provider.
func (p *RateLimitedProvider) GetDefaultModel() string {
	return p.inner.GetDefaultModel()
}
