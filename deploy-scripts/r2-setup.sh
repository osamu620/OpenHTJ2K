#!/usr/bin/env bash
# One-time setup for hosting the HTJ2K RTP sample clips on Cloudflare R2.
#
# Prerequisites:
#   - npm install -g wrangler            # or use `npx wrangler ...`
#   - wrangler login                     # OAuth flow in browser
#   - A Cloudflare account with R2 enabled (Workers & Pages → R2 → Enable).
#
# What this does:
#   1. Creates an R2 bucket `htj2k-samples` in your account
#   2. Applies the CORS policy from r2-cors.json (required for the demo's
#      cross-origin fetch + streaming range reads)
#   3. Uploads the sample .rtp file (replace SAMPLE_LOCAL_PATH below)
#   4. Prints the `r2.dev` public URL you can paste into rtp_demo.html's
#      default URL field (or bind to a custom domain later)
#
# To bind a custom domain (e.g. samples.htj2k-demo.pages.dev):
#   Cloudflare Dashboard → R2 → htj2k-samples → Settings → Domain Access →
#   Connect Custom Domain → enter samples.htj2k-demo.pages.dev
# (Cloudflare manages the DNS automatically if the parent zone is on CF.)

set -euo pipefail

BUCKET="htj2k-samples"
CORS_FILE="$(dirname "$0")/r2-cors.json"

# Local path to the .rtp file — EDIT THIS to wherever you have it.
SAMPLE_LOCAL_PATH="${SAMPLE_LOCAL_PATH:-$HOME/Documents/1080p2997_10bit_150frames.rtp}"
# Object key inside the bucket (keeps URL tidy).
SAMPLE_KEY="1080p2997_10bit_150frames.rtp"

echo "==> 1/4  Create R2 bucket: $BUCKET"
if wrangler r2 bucket list 2>/dev/null | grep -q "^${BUCKET}$"; then
  echo "    already exists — skipping"
else
  wrangler r2 bucket create "$BUCKET"
fi

echo ""
echo "==> 2/4  Apply CORS policy from $CORS_FILE"
wrangler r2 bucket cors put "$BUCKET" --rules "$CORS_FILE"

echo ""
echo "==> 3/4  Upload sample  ($SAMPLE_LOCAL_PATH → $BUCKET/$SAMPLE_KEY)"
if [[ ! -f "$SAMPLE_LOCAL_PATH" ]]; then
  echo "    ERROR: file not found: $SAMPLE_LOCAL_PATH"
  echo "    Set SAMPLE_LOCAL_PATH env var or edit this script."
  exit 1
fi
wrangler r2 object put "$BUCKET/$SAMPLE_KEY" --file "$SAMPLE_LOCAL_PATH"

echo ""
echo "==> 4/4  Enable public r2.dev access (opt-in; use custom domain in prod)"
cat <<EOF

Next manual steps:

  1. Cloudflare Dashboard → R2 → $BUCKET → Settings
     → "Public access" → "Allow Access" via r2.dev subdomain.
     You'll get a URL like:
       https://pub-<hash>.r2.dev/$SAMPLE_KEY

  2. (Recommended) Connect a custom domain:
     → "Connect Custom Domain" → "samples.htj2k-demo.pages.dev"
     (or any subdomain of a zone you manage on Cloudflare).

  3. Update subprojects/rtp_demo.html's default URL to the R2 URL
     from step 1 or 2.  The file deploy-scripts/update-rtp-url.sh
     in this directory does that in one command:
       ./deploy-scripts/update-rtp-url.sh  <new-url>

Done.
EOF
