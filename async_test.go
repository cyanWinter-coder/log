package log

import (
	"errors"
	"sync"
	"testing"
	"time"
)

// collectLogger records every event it receives.
type collectLogger struct {
	mu     sync.Mutex
	events [][]interface{}
	calls  int
	hook   func(call int) error // called before recording; may return an error or panic
}

func (c *collectLogger) Log(keyvals ...interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	if c.hook != nil {
		if err := c.hook(c.calls); err != nil {
			return err
		}
	}
	kvs := make([]interface{}, len(keyvals))
	copy(kvs, keyvals)
	c.events = append(c.events, kvs)
	return nil
}

func (c *collectLogger) snapshot() [][]interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([][]interface{}, len(c.events))
	copy(out, c.events)
	return out
}

// gatedLogger blocks inside Log until released, letting tests control when
// the downstream makes progress.
type gatedLogger struct {
	entered chan struct{} // receives each time Log is entered
	release chan struct{} // Log returns after receiving from release
}

func newGatedLogger() *gatedLogger {
	return &gatedLogger{
		entered: make(chan struct{}, 1),
		release: make(chan struct{}),
	}
}

func (g *gatedLogger) Log(keyvals ...interface{}) error {
	g.entered <- struct{}{}
	<-g.release
	return nil
}

func fastConfig() AsyncLoggerConfig {
	return AsyncLoggerConfig{
		QueueSize:     256,
		BatchSize:     4,
		FlushInterval: 5 * time.Millisecond,
		RetryBackoff:  time.Millisecond,
		CloseTimeout:  5 * time.Second,
	}
}

func TestAsyncLoggerDeliversEventsInOrder(t *testing.T) {
	downstream := &collectLogger{}
	cfg := fastConfig()
	cfg.BatchSize = 7 // cross batch boundaries
	l := NewAsyncLogger(downstream, cfg)

	const n = 100
	for i := 0; i < n; i++ {
		if err := l.Log("seq", i); err != nil {
			t.Fatalf("Log %d: %v", i, err)
		}
	}
	if err := l.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	events := downstream.snapshot()
	if len(events) != n {
		t.Fatalf("want %d events, got %d", n, len(events))
	}
	for i, ev := range events {
		if len(ev) != 2 || ev[0] != "seq" || ev[1] != i {
			t.Fatalf("event %d out of order or malformed: %v", i, ev)
		}
	}
	if got := l.Written(); got != n {
		t.Errorf("Written = %d, want %d", got, n)
	}
	if got := l.Dropped(); got != 0 {
		t.Errorf("Dropped = %d, want 0", got)
	}
	if got := l.Failed(); got != 0 {
		t.Errorf("Failed = %d, want 0", got)
	}
}

func TestAsyncLoggerDropsWhenQueueFull(t *testing.T) {
	gate := newGatedLogger()
	cfg := fastConfig()
	cfg.QueueSize = 1
	cfg.BatchSize = 1
	cfg.BlockTimeout = 0 // drop immediately
	l := NewAsyncLogger(gate, cfg)
	defer func() {
		close(gate.release)
		l.Close()
	}()

	if err := l.Log("n", 1); err != nil {
		t.Fatalf("Log 1: %v", err)
	}
	<-gate.entered // run is now blocked inside downstream Log

	if err := l.Log("n", 2); err != nil {
		t.Fatalf("Log 2: %v", err) // sits in the single queue slot
	}
	if err := l.Log("n", 3); !errors.Is(err, ErrAsyncQueueFull) {
		t.Fatalf("Log 3 error = %v, want ErrAsyncQueueFull", err)
	}
	if got := l.Dropped(); got != 1 {
		t.Fatalf("Dropped = %d, want 1", got)
	}
}

func TestAsyncLoggerBlockTimeoutExpires(t *testing.T) {
	gate := newGatedLogger()
	cfg := fastConfig()
	cfg.QueueSize = 1
	cfg.BatchSize = 1
	cfg.BlockTimeout = 50 * time.Millisecond
	l := NewAsyncLogger(gate, cfg)
	defer func() {
		close(gate.release)
		l.Close()
	}()

	l.Log("n", 1)
	<-gate.entered
	l.Log("n", 2) // fills the queue

	start := time.Now()
	if err := l.Log("n", 3); !errors.Is(err, ErrAsyncQueueFull) {
		t.Fatalf("Log error = %v, want ErrAsyncQueueFull", err)
	}
	if elapsed := time.Since(start); elapsed < cfg.BlockTimeout {
		t.Fatalf("Log returned after %v, want it to block for at least %v", elapsed, cfg.BlockTimeout)
	}
	if got := l.Dropped(); got != 1 {
		t.Fatalf("Dropped = %d, want 1", got)
	}
}

func TestAsyncLoggerBlockTimeoutSucceedsWhenSpaceFrees(t *testing.T) {
	gate := newGatedLogger()
	cfg := fastConfig()
	cfg.QueueSize = 1
	cfg.BatchSize = 1
	cfg.BlockTimeout = 2 * time.Second
	l := NewAsyncLogger(gate, cfg)
	defer func() {
		close(gate.release)
		l.Close()
	}()

	l.Log("n", 1)
	<-gate.entered
	l.Log("n", 2) // fills the queue

	errc := make(chan error, 1)
	go func() { errc <- l.Log("n", 3) }()

	// Let the producer block on the full queue, then make progress.
	time.Sleep(20 * time.Millisecond)
	gate.release <- struct{}{} // event 1 completes
	<-gate.entered             // event 2 entered downstream, queue slot freed

	select {
	case err := <-errc:
		if err != nil {
			t.Fatalf("blocked Log returned %v, want nil", err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked Log did not return after queue space freed")
	}
	if got := l.Dropped(); got != 0 {
		t.Fatalf("Dropped = %d, want 0", got)
	}
}

func TestAsyncLoggerRetriesFailedEvent(t *testing.T) {
	boom := errors.New("boom")
	downstream := &collectLogger{}
	downstream.hook = func(call int) error {
		if call == 1 {
			return boom // fail the first attempt only
		}
		return nil
	}
	cfg := fastConfig()
	cfg.BatchSize = 1
	cfg.MaxRetries = 3
	l := NewAsyncLogger(downstream, cfg)
	defer l.Close()

	if err := l.Log("k", "v"); err != nil {
		t.Fatalf("Log: %v", err)
	}
	l.Flush()

	if got := l.Written(); got != 1 {
		t.Errorf("Written = %d, want 1", got)
	}
	if got := l.Failed(); got != 0 {
		t.Errorf("Failed = %d, want 0", got)
	}
	downstream.mu.Lock()
	calls := downstream.calls
	downstream.mu.Unlock()
	if calls != 2 {
		t.Errorf("downstream calls = %d, want 2 (1 failure + 1 retry)", calls)
	}
}

func TestAsyncLoggerCountsPermanentFailuresOnce(t *testing.T) {
	boom := errors.New("boom")
	downstream := &collectLogger{}
	downstream.hook = func(call int) error { return boom }
	cfg := fastConfig()
	cfg.BatchSize = 2
	cfg.MaxRetries = 2
	l := NewAsyncLogger(downstream, cfg)
	defer l.Close()

	const n = 5
	for i := 0; i < n; i++ {
		if err := l.Log("n", i); err != nil {
			t.Fatalf("Log %d: %v", i, err)
		}
	}
	l.Flush()

	if got := l.Written(); got != 0 {
		t.Errorf("Written = %d, want 0", got)
	}
	if got := l.Failed(); got != n {
		t.Errorf("Failed = %d, want %d (each event counted exactly once)", got, n)
	}
	downstream.mu.Lock()
	calls := downstream.calls
	downstream.mu.Unlock()
	if want := n * (1 + cfg.MaxRetries); calls != want {
		t.Errorf("downstream calls = %d, want %d", calls, want)
	}
}

func TestAsyncLoggerSurvivesDownstreamPanic(t *testing.T) {
	downstream := &collectLogger{}
	downstream.hook = func(call int) error {
		if call == 1 {
			panic("downstream exploded")
		}
		return nil
	}
	cfg := fastConfig()
	cfg.BatchSize = 1
	cfg.MaxRetries = 0
	l := NewAsyncLogger(downstream, cfg)
	defer l.Close()

	l.Log("n", 1) // will panic downstream
	l.Log("n", 2) // must still be delivered after the panic

	done := make(chan struct{})
	go func() {
		l.Flush()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Flush did not return after downstream panic; waiters were not woken")
	}

	if got := l.Failed(); got != 1 {
		t.Errorf("Failed = %d, want 1", got)
	}
	if got := l.Written(); got != 1 {
		t.Errorf("Written = %d, want 1", got)
	}
	events := downstream.snapshot()
	if len(events) != 1 || events[0][1] != 2 {
		t.Errorf("downstream events = %v, want the second event only", events)
	}
}

func TestAsyncLoggerCloseFlushesAndRejectsNewWrites(t *testing.T) {
	downstream := &collectLogger{}
	cfg := fastConfig()
	cfg.FlushInterval = time.Hour // ensure Close, not the timer, flushes
	l := NewAsyncLogger(downstream, cfg)

	const n = 10
	for i := 0; i < n; i++ {
		if err := l.Log("n", i); err != nil {
			t.Fatalf("Log %d: %v", i, err)
		}
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := l.Written(); got != n {
		t.Errorf("Written = %d, want %d after Close", got, n)
	}
	if err := l.Log("n", n+1); !errors.Is(err, ErrAsyncLoggerClosed) {
		t.Errorf("Log after Close = %v, want ErrAsyncLoggerClosed", err)
	}
	// Counters must not move after Close.
	if got := l.Written(); got != n {
		t.Errorf("Written = %d after rejected Log, want %d", got, n)
	}
	if got := l.Dropped(); got != 0 {
		t.Errorf("Dropped = %d after rejected Log, want 0", got)
	}
}

func TestAsyncLoggerCloseAndFlushAreConcurrentIdempotent(t *testing.T) {
	downstream := &collectLogger{}
	l := NewAsyncLogger(downstream, fastConfig())

	const producers = 4
	const perProducer = 50
	var wg sync.WaitGroup
	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < perProducer; i++ {
				l.Log("p", p, "i", i)
			}
		}(p)
	}
	wg.Wait()

	// Hammer Close and Flush concurrently; all calls must return without
	// panic, and every accepted event must be written exactly once.
	const callers = 8
	errs := make(chan error, 2*callers)
	for i := 0; i < callers; i++ {
		go func() { errs <- l.Close() }()
		go func() { errs <- l.Flush() }()
	}
	for i := 0; i < 2*callers; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent Close/Flush: %v", err)
		}
	}
	if got := l.Written(); got != producers*perProducer {
		t.Errorf("Written = %d, want %d", got, producers*perProducer)
	}
	if got := len(downstream.snapshot()); got != producers*perProducer {
		t.Errorf("downstream events = %d, want %d", got, producers*perProducer)
	}
}

func TestAsyncLoggerCloseTimeout(t *testing.T) {
	gate := newGatedLogger()
	cfg := fastConfig()
	cfg.QueueSize = 1
	cfg.BatchSize = 1
	cfg.CloseTimeout = 50 * time.Millisecond
	l := NewAsyncLogger(gate, cfg)

	l.Log("n", 1)
	<-gate.entered // downstream now blocked

	start := time.Now()
	if err := l.Close(); !errors.Is(err, ErrAsyncCloseTimeout) {
		t.Fatalf("Close = %v, want ErrAsyncCloseTimeout", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("Close took %v, want it to give up near %v", elapsed, cfg.CloseTimeout)
	}

	// Unblock the downstream; the background goroutine must converge and a
	// repeated Close must succeed.
	close(gate.release)
	if err := l.Close(); err != nil {
		t.Fatalf("second Close = %v, want nil", err)
	}
	if got := l.Written(); got != 1 {
		t.Fatalf("Written = %d, want 1", got)
	}
}

func TestAsyncLoggerFlushWaitsForAcceptedEvents(t *testing.T) {
	downstream := &collectLogger{}
	slow := LoggerFunc(func(keyvals ...interface{}) error {
		time.Sleep(20 * time.Millisecond)
		return downstream.Log(keyvals...)
	})
	cfg := fastConfig()
	cfg.BatchSize = 1
	l := NewAsyncLogger(slow, cfg)
	defer l.Close()

	const n = 3
	for i := 0; i < n; i++ {
		l.Log("n", i)
	}
	start := time.Now()
	l.Flush()
	if elapsed := time.Since(start); elapsed < n*20*time.Millisecond {
		t.Fatalf("Flush returned after %v, before the accepted events were written", elapsed)
	}
	if got := l.Written(); got != n {
		t.Fatalf("Written = %d, want %d when Flush returned", got, n)
	}
}

func TestAsyncLoggerWithContextIsImmutableAndConcurrentSafe(t *testing.T) {
	downstream := &collectLogger{}
	l := NewAsyncLogger(downstream, fastConfig())
	defer l.Close()

	base := With(l, "app", "test")
	l1 := With(base, "req", "1")
	l2 := With(base, "req", "2")

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			logger := l1
			if g%2 == 0 {
				logger = l2
			}
			for i := 0; i < 50; i++ {
				logger.Log("i", i)
			}
		}(g)
	}
	wg.Wait()
	l.Flush()

	for _, ev := range downstream.snapshot() {
		// Expect exactly: app=test req=<1|2> i=<n>
		if len(ev) != 6 {
			t.Fatalf("event has %d keyvals, want 6: %v", len(ev), ev)
		}
		if ev[0] != "app" || ev[1] != "test" {
			t.Fatalf("event missing base context: %v", ev)
		}
		if ev[2] != "req" || (ev[3] != "1" && ev[3] != "2") {
			t.Fatalf("event has corrupted context: %v", ev)
		}
		if ev[4] != "i" {
			t.Fatalf("event missing call keyvals: %v", ev)
		}
	}
}

type closeTrackingLogger struct {
	closed bool
}

func (c *closeTrackingLogger) Log(keyvals ...interface{}) error { return nil }
func (c *closeTrackingLogger) Close() error                     { c.closed = true; return nil }

func TestAsyncLoggerCloseDoesNotCloseWrappedLogger(t *testing.T) {
	downstream := &closeTrackingLogger{}
	l := NewAsyncLogger(downstream, fastConfig())
	l.Log("k", "v")
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if downstream.closed {
		t.Error("Close closed the wrapped logger; ownership must stay with the caller")
	}
}

func TestAsyncLoggerStressAccountingInvariant(t *testing.T) {
	boom := errors.New("boom")
	downstream := &collectLogger{}
	// Fail roughly every 7th downstream call; retries absorb most failures.
	downstream.hook = func(call int) error {
		if call%7 == 0 {
			return boom
		}
		return nil
	}
	cfg := AsyncLoggerConfig{
		QueueSize:     16,
		BatchSize:     5,
		FlushInterval: time.Millisecond,
		BlockTimeout:  time.Millisecond,
		MaxRetries:    1,
		RetryBackoff:  time.Millisecond,
		CloseTimeout:  30 * time.Second,
	}
	l := NewAsyncLogger(downstream, cfg)

	const producers = 8
	const perProducer = 200
	var accepted, dropped, closed uint64
	var mu sync.Mutex
	var wg sync.WaitGroup
	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			var a, d, c uint64
			for i := 0; i < perProducer; i++ {
				switch err := l.Log("p", p, "i", i); {
				case err == nil:
					a++
				case errors.Is(err, ErrAsyncQueueFull):
					d++
				case errors.Is(err, ErrAsyncLoggerClosed):
					c++
				default:
					t.Errorf("unexpected Log error: %v", err)
				}
			}
			mu.Lock()
			accepted += a
			dropped += d
			closed += c
			mu.Unlock()
		}(p)
	}
	wg.Wait()
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if got := l.Dropped(); got != dropped {
		t.Errorf("Dropped = %d, want %d (counted by producers)", got, dropped)
	}
	if total := l.Written() + l.Failed(); total != accepted {
		t.Errorf("Written+Failed = %d, want %d accepted events (written=%d failed=%d)",
			total, accepted, l.Written(), l.Failed())
	}
	if closed != 0 {
		t.Errorf("%d events rejected as closed before Close was called", closed)
	}
}
