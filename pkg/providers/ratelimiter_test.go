package providers

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockProvider is a minimal LLMProvider for testing.
type mockProvider struct {
	defaultModel string
	// onChat is called on each Chat() invocation; returns the configured response/error.
	onChat func(ctx context.Context) (*LLMResponse, error)
}

func (m *mockProvider) Chat(ctx context.Context, _ []Message, _ []ToolDefinition, _ string, _ map[string]any) (*LLMResponse, error) {
	if m.onChat != nil {
		return m.onChat(ctx)
	}
	return &LLMResponse{Content: "ok", FinishReason: "stop"}, nil
}

func (m *mockProvider) GetDefaultModel() string { return m.defaultModel }

func okProvider() *mockProvider {
	return &mockProvider{defaultModel: "test-model"}
}

// --- No-op passthrough ---

func TestNewRateLimitedProvider_NoLimits_ReturnsOriginal(t *testing.T) {
	inner := okProvider()
	wrapped := NewRateLimitedProvider(inner, 0, 0)
	// With both limits off, NewRateLimitedProvider must return the inner provider directly.
	if wrapped != LLMProvider(inner) {
		t.Errorf("expected original provider to be returned unchanged, got %T", wrapped)
	}
}

func TestNewRateLimitedProvider_NegativeLimits_ReturnsOriginal(t *testing.T) {
	inner := okProvider()
	wrapped := NewRateLimitedProvider(inner, -1, -10)
	if wrapped != LLMProvider(inner) {
		t.Errorf("expected original provider to be returned unchanged, got %T", wrapped)
	}
}

// --- GetDefaultModel delegation ---

func TestRateLimitedProvider_GetDefaultModel(t *testing.T) {
	inner := &mockProvider{defaultModel: "my-model"}
	wrapped := NewRateLimitedProvider(inner, 10, 0)
	if got := wrapped.GetDefaultModel(); got != "my-model" {
		t.Errorf("GetDefaultModel() = %q, want %q", got, "my-model")
	}
}

// --- Concurrency limit ---

func TestRateLimitedProvider_MaxConcurrent_LimitsParallelism(t *testing.T) {
	const maxConcurrent = 3
	const goroutines = 10

	var (
		inFlight atomic.Int64
		exceeded atomic.Bool
		peak     atomic.Int64
		mu       sync.Mutex
		wg       sync.WaitGroup
	)

	inner := &mockProvider{
		onChat: func(ctx context.Context) (*LLMResponse, error) {
			cur := inFlight.Add(1)
			defer inFlight.Add(-1)

			// Record peak concurrency
			mu.Lock()
			if cur > peak.Load() {
				peak.Store(cur)
			}
			mu.Unlock()

			if cur > maxConcurrent {
				exceeded.Store(true)
			}
			// Simulate a slow call so goroutines pile up
			time.Sleep(20 * time.Millisecond)
			return &LLMResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}

	provider := NewRateLimitedProvider(inner, 0, maxConcurrent)

	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			_, _ = provider.Chat(context.Background(), nil, nil, "m", nil)
		}()
	}
	wg.Wait()

	if exceeded.Load() {
		t.Errorf("concurrency exceeded max_concurrent=%d (peak=%d)", maxConcurrent, peak.Load())
	}
	if peak.Load() == 0 {
		t.Error("expected at least one chat call to execute")
	}
}

// --- ctx cancel while waiting for concurrency slot ---

func TestRateLimitedProvider_MaxConcurrent_CtxCancelWhileWaiting(t *testing.T) {
	const maxConcurrent = 1

	// hold is closed to release the blocker goroutine at the end of the test.
	hold := make(chan struct{})

	inner := &mockProvider{
		onChat: func(ctx context.Context) (*LLMResponse, error) {
			<-hold // hold the slot until we release it
			return &LLMResponse{Content: "ok"}, nil
		},
	}

	provider := NewRateLimitedProvider(inner, 0, maxConcurrent)

	// Start a goroutine that holds the single slot.
	var blockerReady sync.WaitGroup
	blockerReady.Add(1)
	go func() {
		// Signal that we're about to occupy the slot, then call Chat.
		blockerReady.Done()
		_, _ = provider.Chat(context.Background(), nil, nil, "m", nil)
	}()
	blockerReady.Wait()
	// Give the blocker goroutine time to actually enter Chat and acquire the semaphore.
	time.Sleep(10 * time.Millisecond)

	// Now try to call Chat with a context that we cancel immediately.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	_, err := provider.Chat(ctx, nil, nil, "m", nil)

	// Unblock the first goroutine.
	close(hold)

	if err == nil {
		t.Fatal("expected context error, got nil")
	}
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

// --- RPM limit applies ---

func TestRateLimitedProvider_RPM_LimitsRate(t *testing.T) {
	// rpm=2 means at most 2 requests per minute, i.e. one token every 30 s.
	// With an initial burst of 2 the first two calls should be immediate.
	// Subsequent calls would need to wait 30 s — we verify via context cancellation.
	const rpm = 2

	callCount := atomic.Int64{}
	inner := &mockProvider{
		onChat: func(ctx context.Context) (*LLMResponse, error) {
			callCount.Add(1)
			return &LLMResponse{Content: "ok"}, nil
		},
	}

	provider := NewRateLimitedProvider(inner, rpm, 0)

	// First two calls should succeed quickly (burst allowance).
	for i := range 2 {
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		_, err := provider.Chat(ctx, nil, nil, "m", nil)
		cancel()
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i+1, err)
		}
	}

	// Third call should be blocked by the limiter beyond our short timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := provider.Chat(ctx, nil, nil, "m", nil)
	if err == nil {
		t.Fatal("3rd call should have been blocked by RPM limiter but returned nil error")
	}

	if callCount.Load() != 2 {
		t.Errorf("expected exactly 2 successful calls, got %d", callCount.Load())
	}
}

// --- Combined RPM + concurrency ---

func TestRateLimitedProvider_BothLimits(t *testing.T) {
	inner := okProvider()
	provider := NewRateLimitedProvider(inner, 60, 2)

	// Should succeed for a couple of immediate calls (burst covers RPM, concurrency not exhausted).
	for range 2 {
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		_, err := provider.Chat(ctx, nil, nil, "m", nil)
		cancel()
		if err != nil {
			t.Fatalf("unexpected error with combined limits: %v", err)
		}
	}
}
