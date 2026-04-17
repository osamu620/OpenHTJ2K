# JPIP Server Deployment

Deploy the OpenHTJ2K JPIP server on a VPS with Cloudflare Tunnel.

## Quick start (any Linux VPS)

```bash
# 1. Clone and build the Docker image
git clone https://github.com/osamu620/OpenHTJ2K.git
cd OpenHTJ2K
docker build -t jpip-server -f deploy/Dockerfile .

# 2. Copy your test codestream into a volume
mkdir -p data
cp /path/to/input.j2c data/

# 3. Run with quick tunnel (anonymous, temporary URL)
docker run --rm -v $PWD/data:/data \
  -e JPIP_TUNNEL=1 \
  jpip-server /data/input.j2c

# The tunnel URL is printed to stdout.
# Open: https://htj2k-demo.pages.dev/jpip_demo.html?server=<tunnel-url>
```

## Production setup (named tunnel + custom domain)

```bash
# 1. Create a named tunnel on your Cloudflare account
#    (run once, on any machine with cloudflared logged in)
cloudflared tunnel create jpip
cloudflared tunnel route dns jpip jpip.yourdomain.com

# 2. Copy the tunnel token from the Cloudflare dashboard:
#    Zero Trust → Networks → Tunnels → jpip → Configure → token

# 3. Run with the named tunnel
docker run -d --restart=unless-stopped \
  -v $PWD/data:/data \
  jpip-server /data/input.j2c --tunnel-token <TOKEN>

# Open: https://htj2k-demo.pages.dev/jpip_demo.html?server=https://jpip.yourdomain.com
```

## Without Docker

```bash
# Build natively
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run server
./build/bin/open_htj2k_jpip_server input.j2c &

# Quick tunnel
cloudflared tunnel --url http://localhost:8080

# Or named tunnel
cloudflared tunnel run --token <TOKEN>
```

## Recommended VPS providers (Tokyo region for low latency to Cloudflare)

| Provider | Plan | Cost | Notes |
|---|---|---|---|
| Oracle Cloud | ARM A1 Flex | Free | 4 cores, 24 GB RAM |
| AWS Lightsail | 2 GB | $5/mo | Easy setup |
| DigitalOcean | Basic | $6/mo | Tokyo datacenter |
| Vultr | Cloud Compute | $6/mo | Tokyo datacenter |
