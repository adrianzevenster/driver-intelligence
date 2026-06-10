#!/usr/bin/env bash
# Bootstrap a fresh Hetzner Ubuntu 22.04/24.04 instance.
# Run as root (or with sudo) from the project directory.
#
# Recommended instance: CX32 (8 vCPU, 16 GB RAM) — required for Ollama 8B.
# Usage:
#   git clone <repo> /opt/f1di && cd /opt/f1di
#   cp .env.prod.example .env && nano .env   # set passwords
#   sudo bash scripts/hetzner_setup.sh
#
# Override Ollama model:  OLLAMA_MODEL=qwen2.5:7b sudo bash scripts/hetzner_setup.sh
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq curl git ufw

# ── 2. Docker ─────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

# ── 3. Firewall ───────────────────────────────────────────────────────────────
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp   comment "SSH"
ufw allow 8080/tcp comment "F1DI API"
ufw allow 3000/tcp comment "Grafana"
ufw --force enable
echo "UFW active. Ports open: 22 (SSH), 8080 (API), 3000 (Grafana)."
echo "To restrict Grafana to your IP only: ufw delete allow 3000/tcp && ufw allow from <YOUR_IP> to any port 3000"

# ── 4. Validate .env ──────────────────────────────────────────────────────────
cd "$APP_DIR"
if [[ ! -f .env ]]; then
  cp .env.prod.example .env
  echo ""
  echo "ERROR: .env not found — copied .env.prod.example as .env."
  echo "Edit $APP_DIR/.env (set POSTGRES_PASSWORD, GRAFANA_PASSWORD, F1DI_STORAGE_URL),"
  echo "then re-run this script."
  exit 1
fi

if grep -q "changeme" .env; then
  echo ""
  echo "WARNING: .env still contains 'changeme' placeholders — update passwords before exposing publicly."
fi

# ── 5. Build and start services ───────────────────────────────────────────────
docker compose -f docker-compose.prod.yml pull --quiet
docker compose -f docker-compose.prod.yml up -d --build
echo "Services starting..."

# ── 6. Pull Ollama model ──────────────────────────────────────────────────────
echo "Waiting for Ollama service..."
for i in $(seq 1 30); do
  if docker compose -f docker-compose.prod.yml exec -T ollama ollama list &>/dev/null 2>&1; then
    break
  fi
  sleep 3
done

echo "Pulling Ollama model: $OLLAMA_MODEL (this can take several minutes)..."
docker compose -f docker-compose.prod.yml exec -T ollama ollama pull "$OLLAMA_MODEL"

# ── 7. Summary ────────────────────────────────────────────────────────────────
PUBLIC_IP="$(curl -sf https://api.ipify.org || hostname -I | awk '{print $1}')"
echo ""
echo "Deployment complete."
echo "  API:     http://${PUBLIC_IP}:8080"
echo "  Docs:    http://${PUBLIC_IP}:8080/docs"
echo "  Grafana: http://${PUBLIC_IP}:3000  (admin / \$GRAFANA_PASSWORD)"
echo ""
echo "Useful commands:"
echo "  docker compose -f docker-compose.prod.yml logs -f api"
echo "  docker compose -f docker-compose.prod.yml ps"
