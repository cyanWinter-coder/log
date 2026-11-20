package log

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ErrAsyncLoggerClosed is returned by Log when the AsyncLogger has been
// closed. Events rejected this way are not counted as dropped.
var ErrAsyncLoggerClosed = errors.New("async logger: closed")

// ErrAsyncQueueFull is returned by Log when the bounded queue is full and the
// configured BlockTimeout elapsed (or is zero). The event is dropped and the
// drop is reflected in the counter returned by Dropped.
var ErrAsyncQueueFull = errors.New("async logger: queue full, event dropped")

// ErrAsyncCloseTimeout is returned by Close when the queued events could not
// be flushed to the wrapped logger within CloseTimeout. The background
// goroutine keeps draining in the background; callers may Close again.
var ErrAsyncCloseTimeout = errors.New("async logger: close timed out before queue drained")

// AsyncLoggerConfig configures an AsyncLogger. See NewAsyncLogger for the
// defaults applied to zero or negative values.
type AsyncLoggerConfig struct {
	// QueueSize bounds the number of log events buffered for the background
	// writer. Log rejects events once the queue is full. Default 1024.
	QueueSize int

	// BatchSize is the maximum number of queued events written to the
	// wrapped logger in one batch. Default 128, capped at QueueSize.
	BatchSize int

	// FlushInterval is the maximum time a non-empty batch waits for more
	// events before being written to the wrapped logger. Default 1s.
	FlushInterval time.Duration

	// BlockTimeout controls how Log behaves when the queue is full. Zero
	// means drop the event immediately; a positive value blocks the caller
	// for up to that duration waiting for queue space before dropping.
	// Default zero (drop immediately).
	BlockTimeout time.Duration

	// MaxRetries is the number of times a failed event write is retried
	// before the event is counted as failed. Default zero (no retries).
	MaxRetries int

	// RetryBackoff is the delay between retries of a failed event write.
	// Default 10ms.
	RetryBackoff time.Duration

	// CloseTimeout bounds how long Close waits for queued events to be
	// flushed to the wrapped logger. Default 5s.
	CloseTimeout time.Duration
}

func (cfg *AsyncLoggerConfig) setDefaults() {
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 1024
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 128
	}
	if cfg.BatchSize > cfg.QueueSize {
		cfg.BatchSize = cfg.QueueSize
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = time.Second
	}
	if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
	}
	if cfg.RetryBackoff <= 0 {
		cfg.RetryBackoff = 10 * time.Millisecond
	}
	if cfg.CloseTimeout <= 0 {
		cfg.CloseTimeout = 5 * time.Second
	}
}

// asyncItem is one slot in the AsyncLogger queue: either a log event or a
// flush marker. Because the queue is FIFO and has a single consumer, a flush
// marker is processed only after every event enqueued before it.
type asyncItem struct {
	keyvals []interface{}
	flush   chan struct{}
}

// AsyncLogger wraps a Logger with a bounded queue and a single background
// goroutine that writes queued events to the wrapped logger in batches. It
// decouples callers from downstream latency: Log returns as soon as the event
// is queued, and slow or failing downstream loggers do not block callers
// beyond the configured BlockTimeout.
//
// Events accepted by Log are eventually written to the wrapped logger or
// counted as failed; they are never silently lost. Events rejected by Log are
// reported via the returned error and, when dropped due to a full queue, via
// Dropped. For every call to Log exactly one of the following holds: the
// event was written (Written), the event exhausted its retries (Failed), the
// event was dropped at admission (Dropped), or the call was rejected because
// the logger is closed.
//
// Events from a single goroutine are written to the wrapped logger in the
// order they were accepted, and events are never reordered within the queue.
// A retried event is retried in place, so retries do not reorder events
// either.
//
// AsyncLogger is safe for concurrent use. Use With, WithPrefix, or WithSuffix
// to attach context as with any other Logger; the context remains immutable
// and is evaluated by the calling goroutine before the event is queued.
//
// Close must be called to release the background goroutine. Close does not
// close the wrapped logger or any underlying writer: ownership stays with the
// caller.
type AsyncLogger struct {
	logger Logger
	cfg    AsyncLoggerConfig

	queue   chan asyncItem
	closeCh chan struct{}
	done    chan struct{}

	// mu guards closed. Producers hold it (read) across the closed check and
	// the queue send, so once Close sets closed (write) no new event can
	// enter the queue and the drain in run sees every accepted event.
	mu     sync.RWMutex
	closed bool

	closeOnce sync.Once

	dropped uint64 // events rejected because the queue was full
	written uint64 // events successfully written to the wrapped logger
	failed  uint64 // events that exhausted retries
}

// NewAsyncLogger returns an AsyncLogger that asynchronously writes to logger,
// buffering up to cfg.QueueSize events. The wrapped logger is only called
// from a single background goroutine, so it need not be safe for concurrent
// use. Call Close to flush buffered events and stop the background goroutine.
func NewAsyncLogger(logger Logger, cfg AsyncLoggerConfig) *AsyncLogger {
	cfg.setDefaults()
	l := &AsyncLogger{
		logger:  logger,
		cfg:     cfg,
		queue:   make(chan asyncItem, cfg.QueueSize),
		closeCh: make(chan struct{}),
		done:    make(chan struct{}),
	}
	go l.run()
	return l
}

// Log enqueues keyvals for asynchronous writing. The keyvals are copied, so
// the caller may reuse the slice after Log returns.
//
// If the queue has space, Log returns nil immediately. If the queue is full,
// Log blocks for up to the configured BlockTimeout waiting for space; if no
// space becomes available it drops the event, counts the drop (see Dropped),
// and returns ErrAsyncQueueFull. If the logger has been closed, Log returns
// ErrAsyncLoggerClosed.
func (l *AsyncLogger) Log(keyvals ...interface{}) error {
	// The queue retains the keyvals beyond this call, so the slice must be
	// copied per the Logger interface contract.
	kvs := make([]interface{}, len(keyvals))
	copy(kvs, keyvals)

	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.closed {
		return ErrAsyncLoggerClosed
	}
	item := asyncItem{keyvals: kvs}
	if l.cfg.BlockTimeout <= 0 {
		select {
		case l.queue <- item:
			return nil
		default:
			atomic.AddUint64(&l.dropped, 1)
			return ErrAsyncQueueFull
		}
	}
	timer := time.NewTimer(l.cfg.BlockTimeout)
	defer timer.Stop()
	select {
	case l.queue <- item:
		return nil
	case <-timer.C:
		atomic.AddUint64(&l.dropped, 1)
		return ErrAsyncQueueFull
	}
}

// Flush blocks until every event accepted before the call has been written to
// the wrapped logger (or counted as failed). Flush is safe to call
// concurrently with Log, Close, and other Flush calls. After Close has
// completed, Flush returns immediately.
func (l *AsyncLogger) Flush() error {
	ack := make(chan struct{})
	l.mu.RLock()
	if l.closed {
		// Close is draining or has drained the queue; wait for it.
		l.mu.RUnlock()
		<-l.done
		return nil
	}
	// The marker is ordered behind every previously accepted event. The
	// send cannot block forever: closed is false, so run is still consuming.
	l.queue <- asyncItem{flush: ack}
	l.mu.RUnlock()
	select {
	case <-ack:
	case <-l.done:
	}
	return nil
}

// Close stops accepting new events and flushes queued events to the wrapped
// logger, waiting at most the configured CloseTimeout. It returns
// ErrAsyncCloseTimeout if the queue did not drain in time; the background
// goroutine keeps draining and a later Close waits again. Close is idempotent
// and safe to call concurrently with Log and Flush. Close does not close the
// wrapped logger or any underlying writer.
func (l *AsyncLogger) Close() error {
	l.closeOnce.Do(func() {
		l.mu.Lock()
		l.closed = true
		l.mu.Unlock()
		close(l.closeCh)
	})
	timer := time.NewTimer(l.cfg.CloseTimeout)
	defer timer.Stop()
	select {
	case <-l.done:
		return nil
	case <-timer.C:
		return ErrAsyncCloseTimeout
	}
}

// Dropped returns the number of events rejected by Log because the queue was
// full and the BlockTimeout elapsed (or was zero).
func (l *AsyncLogger) Dropped() uint64 {
	return atomic.LoadUint64(&l.dropped)
}

// Written returns the number of events successfully written to the wrapped
// logger. An event is counted at most once, however often it was retried.
func (l *AsyncLogger) Written() uint64 {
	return atomic.LoadUint64(&l.written)
}

// Failed returns the number of events that could not be written to the
// wrapped logger within the configured retries. An event is counted at most
// once, and an event is never counted as both written and failed.
func (l *AsyncLogger) Failed() uint64 {
	return atomic.LoadUint64(&l.failed)
}

// run is the single consumer of the queue. It returns only after closeCh is
// closed and the queue is drained; the deferred close of done releases Close
// and Flush waiters on every exit path.
func (l *AsyncLogger) run() {
	defer close(l.done)

	var batch [][]interface{}
	var timer *time.Timer
	var timerC <-chan time.Time

	startTimer := func() {
		if timer == nil {
			timer = time.NewTimer(l.cfg.FlushInterval)
		} else {
			timer.Reset(l.cfg.FlushInterval)
		}
		timerC = timer.C
	}
	stopTimer := func() {
		if timer != nil {
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		}
		timerC = nil
	}
	flush := func() {
		stopTimer()
		if len(batch) == 0 {
			return
		}
		l.writeBatch(batch)
		batch = batch[:0]
	}

	for {
		select {
		case item := <-l.queue:
			if item.flush != nil {
				flush()
				close(item.flush)
				continue
			}
			if len(batch) == 0 {
				startTimer()
			}
			batch = append(batch, item.keyvals)
			if len(batch) >= l.cfg.BatchSize {
				flush()
			}
		case <-timerC:
			flush()
		case <-l.closeCh:
			// No new events can be enqueued once closeCh is closed, so
			// drain what is left and exit.
			for {
				select {
				case item := <-l.queue:
					if item.flush != nil {
						flush()
						close(item.flush)
						continue
					}
					batch = append(batch, item.keyvals)
					if len(batch) >= l.cfg.BatchSize {
						flush()
					}
				default:
					flush()
					return
				}
			}
		}
	}
}

// writeBatch writes each event in order. A failed event is retried in place
// before moving on, so failures never reorder the stream.
func (l *AsyncLogger) writeBatch(batch [][]interface{}) {
	for _, kvs := range batch {
		l.writeEvent(kvs)
	}
}

// writeEvent writes one event, retrying failures up to MaxRetries times, and
// records the final outcome exactly once: written on success, failed after
// the last retry.
func (l *AsyncLogger) writeEvent(kvs []interface{}) {
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			time.Sleep(l.cfg.RetryBackoff)
		}
		if err := l.safeLog(kvs); err == nil {
			atomic.AddUint64(&l.written, 1)
			return
		}
		if attempt >= l.cfg.MaxRetries {
			atomic.AddUint64(&l.failed, 1)
			return
		}
	}
}

// safeLog calls the wrapped logger, converting a panic into an error so the
// background goroutine survives a misbehaving downstream and Flush and Close
// waiters are always woken.
func (l *AsyncLogger) safeLog(kvs []interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("async logger: panic in wrapped logger: %v", r)
		}
	}()
	return l.logger.Log(kvs...)
}
