// Certificate handling.  Dev mode generates an ephemeral ECDSA P-256 cert
// at startup and prints its SHA-256 hash so the viewer can pin it via
// `serverCertificateHashes`.  Chromium enforces three constraints on
// pinnable certs:
//   * ECDSA with P-256
//   * validity strictly less than 2 weeks
//   * key usage includes digitalSignature, EKU includes serverAuth
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"strings"
	"time"
)

type certBundle struct {
	tlsCert  tls.Certificate
	derBytes []byte
}

func (b *certBundle) sha256ColonHex() string {
	h := sha256.Sum256(b.derBytes)
	parts := make([]string, len(h))
	for i, v := range h {
		parts[i] = fmt.Sprintf("%02x", v)
	}
	return strings.Join(parts, ":")
}

func generateDevCert() (*certBundle, error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate p256 key: %w", err)
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("serial: %w", err)
	}
	now := time.Now().UTC()
	tmpl := x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "wt_bridge dev"},
		NotBefore:    now,
		// 13 days — Chromium's serverCertificateHashes requires strictly
		// less than 2 weeks.  14 days is rejected.
		NotAfter:              now.Add(13 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
		DNSNames:              []string{"localhost"},
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &priv.PublicKey, priv)
	if err != nil {
		return nil, fmt.Errorf("sign cert: %w", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, fmt.Errorf("marshal key: %w", err)
	}
	tlsCert := tls.Certificate{
		Certificate: [][]byte{der},
		PrivateKey:  priv,
		Leaf:        nil, // populated by tls.Certificate during use; not needed here
	}
	_ = keyDER // kept around for parity with the PEM path; tls.Certificate doesn't need it
	return &certBundle{tlsCert: tlsCert, derBytes: der}, nil
}

// loadPEM reads cert + key PEM files for the production path (not yet wired
// to the CLI; kept for future use).
//
//nolint:unused
func loadPEM(certPath, keyPath string) (*certBundle, error) {
	tlsCert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("load PEM: %w", err)
	}
	if len(tlsCert.Certificate) == 0 {
		return nil, fmt.Errorf("no cert blocks in %s", certPath)
	}
	return &certBundle{tlsCert: tlsCert, derBytes: tlsCert.Certificate[0]}, nil
}

// pemEncodeCert is a small helper kept for `--print-cert` style dumps; not
// currently invoked but useful for ops debugging.
//
//nolint:unused
func pemEncodeCert(der []byte) []byte {
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}
