#!/bin/bash
# Generate a short-lived self-signed ECDSA-P256 certificate for the static
# server (web/perf/serve.mjs) when running in HTTPS mode.  This cert is
# independent of the bridge's WebTransport dev cert — it secures only the
# page-load HTTP request, not the WebTransport session.
#
# The user will see Chrome's "Your connection is not private" warning the
# first time and must click "Advanced → Proceed".  Browsers remember the
# acceptance per-origin, so subsequent loads are silent until the cert
# expires (13 days).
#
# Usage:
#   gen_static_cert.sh <out-dir> [extra-IP-or-DNS …]
#   gen_static_cert.sh /tmp/wt_static_cert 192.168.0.14 my-host.local
set -e

OUT=${1:?usage: gen_static_cert.sh <out-dir> [extra SAN entries …]}
shift
mkdir -p "$OUT"

# Build the SAN list: always include localhost / loopbacks; append any
# extras (typically the bridge host's LAN IPs).
SAN="DNS:localhost,IP:127.0.0.1,IP:::1"
for entry in "$@"; do
  if [[ "$entry" =~ ^[0-9.:a-fA-F]+$ ]] && [[ "$entry" =~ [.:] ]]; then
    SAN="$SAN,IP:$entry"
  else
    SAN="$SAN,DNS:$entry"
  fi
done

# Skip regeneration if a recent cert already exists.  Browsers cache
# the user's "proceed unsafely" decision per cert fingerprint, so
# preserving the cert across runs avoids re-prompting the user.
CERT="$OUT/cert.pem"
KEY="$OUT/key.pem"
if [ -s "$CERT" ] && [ -s "$KEY" ]; then
  # Refresh if the cert is more than 10 days old (cert is valid for 13).
  if [ "$(find "$CERT" -mtime -10 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[gen_static_cert] reusing $CERT (still fresh)"
    exit 0
  fi
fi

openssl req -x509 \
    -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
    -nodes \
    -days 13 \
    -keyout "$KEY" \
    -out    "$CERT" \
    -subj   "/CN=wt-viewer-dev-static" \
    -addext "subjectAltName=$SAN" \
    -addext "keyUsage=digitalSignature" \
    -addext "extendedKeyUsage=serverAuth" \
    2>&1 | grep -v -E '^([+-]+|Generating)' || true

chmod 600 "$KEY"
echo "[gen_static_cert] wrote $CERT (SAN: $SAN)"
