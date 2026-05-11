// UDP receiver and bounded fan-out to every active WebTransport session.
// Each subscriber gets its own bounded channel; the sender drops the OLDEST
// queued packet when the channel is full.  Drop-on-overrun is critical for
// a live media relay: blocking the UDP loop because one session is slow
// would corrupt every other subscriber's stream.
package main

import (
	"context"
	"log"
	"net"
	"sync"
	"sync/atomic"

	"golang.org/x/net/ipv4"
)

type fanout struct {
	mu          sync.Mutex
	subscribers []*subscriber
	queueDepth  int

	packetsIn atomic.Uint64
	bytesIn   atomic.Uint64
	dropsOut  atomic.Uint64
}

type subscriber struct {
	ch      chan []byte
	dropped atomic.Uint64
}

func newFanout(queueDepth int) *fanout {
	return &fanout{queueDepth: queueDepth}
}

func (f *fanout) subscribe() *subscriber {
	f.mu.Lock()
	defer f.mu.Unlock()
	s := &subscriber{ch: make(chan []byte, f.queueDepth)}
	f.subscribers = append(f.subscribers, s)
	return s
}

func (f *fanout) unsubscribe(s *subscriber) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, x := range f.subscribers {
		if x == s {
			f.subscribers = append(f.subscribers[:i], f.subscribers[i+1:]...)
			break
		}
	}
	close(s.ch)
}

func (f *fanout) subscriberCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.subscribers)
}

func (f *fanout) broadcast(pkt []byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, s := range f.subscribers {
		select {
		case s.ch <- pkt:
		default:
			// Channel full.  Pop the oldest (non-blocking), then push the new.
			// If the pop also fails (rare race), increment drop counter and skip.
			select {
			case <-s.ch:
				s.dropped.Add(1)
				f.dropsOut.Add(1)
				select {
				case s.ch <- pkt:
				default:
					s.dropped.Add(1)
					f.dropsOut.Add(1)
				}
			default:
				s.dropped.Add(1)
				f.dropsOut.Add(1)
			}
		}
	}
}

// runUDPIn binds and drains a UDP socket into the fanout until ctx is done.
// Uses recvmmsg (Linux) to batch up to 64 packets per syscall.
func (f *fanout) runUDPIn(ctx context.Context, listen *net.UDPAddr) error {
	conn, err := net.ListenUDP("udp4", listen)
	if err != nil {
		return err
	}
	defer conn.Close()

	if err := conn.SetReadBuffer(8 << 20); err != nil {
		log.Printf("udp: SetReadBuffer: %v (kernel may clamp; check net.core.rmem_max)", err)
	}
	log.Printf("udp listening %s", listen)

	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	const batch = 64
	pc := ipv4.NewPacketConn(conn)
	msgs := make([]ipv4.Message, batch)
	for i := range msgs {
		msgs[i].Buffers = [][]byte{make([]byte, 2048)}
	}

	for {
		n, err := pc.ReadBatch(msgs, 0)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		for i := 0; i < n; i++ {
			pktLen := msgs[i].N
			f.packetsIn.Add(1)
			f.bytesIn.Add(uint64(pktLen))
			pkt := make([]byte, pktLen)
			copy(pkt, msgs[i].Buffers[0][:pktLen])
			f.broadcast(pkt)
		}
	}
}
