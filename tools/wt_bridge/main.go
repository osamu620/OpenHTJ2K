// wt_bridge — UDP -> WebTransport relay for OpenHTJ2K live HTJ2K streams.
//
// Receives RFC 9828 RTP packets on a UDP socket and forwards each as one
// length-prefixed message on a unidirectional WebTransport stream to every
// connected viewer.  No HTJ2K parsing — opaque pass-through.
package main

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/http3"
	"github.com/quic-go/webtransport-go"
)

func main() {
	var (
		listenUDP   = flag.String("listen-udp", "0.0.0.0:6000", "UDP bind for incoming RTP")
		listenQUIC  = flag.String("listen-quic", "0.0.0.0:4433", "QUIC bind for outgoing WebTransport")
		maxClients  = flag.Int("max-clients", 8, "Max concurrent WebTransport sessions")
		queueDepth  = flag.Int("queue-depth", 8192, "Per-session packet queue depth (drop-oldest on overrun)")
		certPath    = flag.String("cert", "", "PEM cert chain (use instead of --dev for a CA-issued cert)")
		keyPath     = flag.String("key", "", "PEM private key matching --cert")
		dev         = flag.Bool("dev", false, "Generate an ephemeral self-signed cert and print SHA-256 hash")
		initialMTU  = flag.Int("initial-mtu", 1200, "Initial QUIC packet size in bytes (1200-1452); lower for VPN/tunnel paths")
	)
	flag.Parse()

	if !*dev && (*certPath == "" || *keyPath == "") {
		log.Fatalf("either --dev or both --cert and --key are required")
	}
	if *initialMTU < 1200 || *initialMTU > 1452 {
		log.Fatalf("--initial-mtu must be between 1200 and 1452")
	}
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	udpAddr, err := net.ResolveUDPAddr("udp", *listenUDP)
	if err != nil {
		log.Fatalf("resolve --listen-udp: %v", err)
	}

	var bundle *certBundle
	switch {
	case *dev:
		bundle, err = generateDevCert()
		if err != nil {
			log.Fatalf("generate dev cert: %v", err)
		}
		hash := bundle.sha256ColonHex()
		fmt.Fprintf(os.Stderr, "[wt_bridge] dev cert SHA-256:\n")
		fmt.Fprintf(os.Stderr, "[wt_bridge]   %s\n", hash)
		fmt.Fprintf(os.Stderr, "[wt_bridge] viewer URL hint: ?certHash=%s\n", hash)
	default:
		bundle, err = loadPEM(*certPath, *keyPath)
		if err != nil {
			log.Fatalf("load PEM: %v", err)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Fan-out: one UDP listener -> N session goroutines.
	fan := newFanout(*queueDepth)
	go func() {
		if err := fan.runUDPIn(ctx, udpAddr); err != nil {
			log.Printf("udp_in exited: %v", err)
		}
	}()

	// QUIC + WebTransport server.  webtransport-go composes on top of
	// http3.Server: every WebTransport session is an "extended CONNECT"
	// HTTP/3 request that we upgrade via WebTransportServer.Upgrade.
	tlsConf := &tls.Config{
		Certificates: []tls.Certificate{bundle.tlsCert},
		MinVersion:   tls.VersionTLS13,
		NextProtos:   []string{http3.NextProtoH3},
	}
	quicConf := &quic.Config{
		MaxIdleTimeout:    30 * time.Second,
		KeepAlivePeriod:   10 * time.Second,
		EnableDatagrams:   true,
		InitialPacketSize: uint16(*initialMTU),
	}

	wtServer := &webtransport.Server{
		H3: http3.Server{
			Addr:       *listenQUIC,
			TLSConfig:  tlsConf,
			QUICConfig: quicConf,
		},
		// Browsers need permissive CORS-like behaviour because the WT URL
		// origin and the page origin can differ in real deployments.  For
		// dev, accept anything; tighten in --cert mode later.
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if fan.subscriberCount() >= *maxClients {
			http.Error(w, "max clients reached", http.StatusServiceUnavailable)
			return
		}
		sess, err := wtServer.Upgrade(w, r)
		if err != nil {
			log.Printf("upgrade: %v", err)
			return
		}
		runSession(ctx, sess, fan)
	})
	wtServer.H3.Handler = mux

	log.Printf("webtransport listening %s", *listenQUIC)
	errCh := make(chan error, 1)
	go func() { errCh <- wtServer.ListenAndServe() }()

	select {
	case <-ctx.Done():
		log.Printf("shutdown signal")
	case err := <-errCh:
		log.Printf("server exited: %v", err)
	}
	_ = wtServer.Close()
}
