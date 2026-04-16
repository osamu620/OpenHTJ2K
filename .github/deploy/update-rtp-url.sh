#!/usr/bin/env bash
# Replaces the default URL value on the rtp_demo.html <input id="rtp-url">.
# Run from the repo root after uploading a new clip to R2 / S3 / wherever.
#
# Usage:
#   ./.github/deploy/update-rtp-url.sh <full-https-url>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <full-https-url>" >&2
  exit 1
fi
NEW_URL="$1"

TARGET="subprojects/rtp_demo.html"
if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: $TARGET not found (run from repo root)" >&2
  exit 1
fi

# Match the value="..." line following the <input id="rtp-url"> declaration.
# We use perl because macOS sed's in-place flag is non-portable.
perl -i -pe 's{^(               value=")[^"]+(">)$}{${1}'"$NEW_URL"'${2}}' "$TARGET"

echo "Updated $TARGET default URL →"
grep -A0 'id="rtp-url"' -A2 "$TARGET" | grep value=
