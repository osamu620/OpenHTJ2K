// Per-WebTransport-session handler.  Opens one unidirectional stream and
// forwards every UDP datagram it sees on its subscriber channel,
// length-prefixed: [u16BE len][packet bytes].  Streams (not datagrams)
// because Chromium negotiates a max WebTransport datagram size of ~1170 B,
// below typical RFC 9828 RTP packet sizes (~1400 B).
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"log"
	"sync/atomic"

	"github.com/quic-go/webtransport-go"
)

// monotonic id counter, just for log readability.
var sessionSeq atomic.Uint64

func runSession(ctx context.Context, sess *webtransport.Session, fan *fanout) {
	id := sessionSeq.Add(1)
	log.Printf("session accepted id=%d remote=%s", id, sess.RemoteAddr())

	stream, err := sess.OpenUniStreamSync(ctx)
	if err != nil {
		log.Printf("session %d: open uni stream: %v", id, err)
		return
	}
	defer stream.Close()

	sub := fan.subscribe()
	defer fan.unsubscribe(sub)

	var forwarded uint64
	hdr := make([]byte, 2)

	for {
		select {
		case <-ctx.Done():
			return
		case <-sess.Context().Done():
			log.Printf("session %d ended; forwarded=%d dropped=%d", id, forwarded, sub.dropped.Load())
			return
		case pkt, ok := <-sub.ch:
			if !ok {
				return
			}
			if len(pkt) > 0xFFFF {
				log.Printf("session %d: oversized packet %d, dropping", id, len(pkt))
				continue
			}
			binary.BigEndian.PutUint16(hdr, uint16(len(pkt)))
			if _, err := stream.Write(hdr); err != nil {
				if !isClosedErr(err) {
					log.Printf("session %d: write hdr: %v", id, err)
				}
				return
			}
			if _, err := stream.Write(pkt); err != nil {
				if !isClosedErr(err) {
					log.Printf("session %d: write body: %v", id, err)
				}
				return
			}
			// Local-only counter — this goroutine is the sole writer and
			// reader, so no atomic dance is needed.  (Cross-session totals
			// live on `fanout.stats` which uses atomics.)
			forwarded++
			if forwarded == 1 || forwarded%1000 == 0 {
				log.Printf("session %d forwarded=%d", id, forwarded)
			}
		}
	}
}

func isClosedErr(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrClosedPipe) ||
		errors.Is(err, context.Canceled)
}
