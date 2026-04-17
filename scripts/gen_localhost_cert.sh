#!/bin/sh
# Generate a self-signed TLS certificate for localhost QUIC testing.
# Usage: ./scripts/gen_localhost_cert.sh [output_dir]
#
# Produces server.key and server.cert in the output directory (default: cwd).
# The certificate is valid for 365 days and covers localhost + 127.0.0.1.

set -e
OUT="${1:-.}"
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout "$OUT/server.key" -out "$OUT/server.cert" \
    -days 365 -nodes \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo "Created $OUT/server.key and $OUT/server.cert"
