#!/usr/bin/env bash
# One-time setup of TIIM on a fresh Ubuntu 22.04/24.04 server. Run as root:
#   curl -fsSL https://raw.githubusercontent.com/tradingwithedgar-del/tradingbot-dev1/claude/clever-lamport-2atw36/deploy/setup_server.sh | bash
# Safe to run again: it only adds what's missing.
set -euo pipefail

REPO="https://github.com/tradingwithedgar-del/tradingbot-dev1.git"
BRANCH="${TIIM_BRANCH:-claude/clever-lamport-2atw36}"
APP=/opt/tiim

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl ufw >/dev/null

echo "==> Creating the 'tiim' user"
id tiim >/dev/null 2>&1 || useradd --create-home --home-dir /home/tiim --shell /bin/bash tiim

echo "==> Downloading TIIM"
if [ ! -d "$APP/.git" ]; then
  git clone --quiet -b "$BRANCH" "$REPO" "$APP"
fi
git config --global --add safe.directory "$APP"
chown -R tiim:tiim "$APP"

echo "==> Installing Python libraries (a few minutes)"
sudo -u tiim python3 -m venv "$APP/.venv"
sudo -u tiim "$APP/.venv/bin/pip" install -q --upgrade pip
sudo -u tiim "$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"

echo "==> Installing Claude Code (lets TIIM read headlines with your Claude subscription)"
if [ ! -x /home/tiim/.local/bin/claude ]; then
  sudo -u tiim bash -c 'curl -fsSL https://claude.ai/install.sh | bash' || echo "   (Claude Code install failed - TIIM still runs with the keyword news reader)"
fi

if [ ! -f "$APP/.env" ]; then
  echo
  echo "==> Your settings (stored only on this server in $APP/.env)"
  read -r -p "TradeLocker email: " TL_EMAIL </dev/tty
  read -r -s -p "TradeLocker demo password: " TL_PASSWORD </dev/tty; echo
  read -r -p "TradeLocker server [PLEXY]: " TL_SERVER </dev/tty; TL_SERVER=${TL_SERVER:-PLEXY}
  read -r -s -p "Choose a password for the dashboard: " DASH_PW </dev/tty; echo
  cp "$APP/.env.example" "$APP/.env"
  sed -i "s|^TL_EMAIL=.*|TL_EMAIL=$TL_EMAIL|; s|^TL_SERVER=.*|TL_SERVER=$TL_SERVER|" "$APP/.env"
  python3 - "$APP/.env" "$TL_PASSWORD" "$DASH_PW" <<'PY'
import sys, re
path, pw, dash = sys.argv[1:4]
s = open(path).read()
s = re.sub(r"^TL_PASSWORD=.*$", lambda m: "TL_PASSWORD=" + pw, s, flags=re.M)
s = re.sub(r"^DASHBOARD_PASSWORD=.*$", lambda m: "DASHBOARD_PASSWORD=" + dash, s, flags=re.M)
open(path, "w").write(s)
PY
  chown tiim:tiim "$APP/.env"; chmod 600 "$APP/.env"
fi

echo "==> Installing the 'tiim' command"
cat > /usr/local/bin/tiim <<'SH'
#!/usr/bin/env bash
# Run any TIIM command as the tiim user, e.g. `tiim status`, `tiim strategies`, `tiim news`
cd /opt/tiim && exec sudo -u tiim env PATH="/home/tiim/.local/bin:$PATH" /opt/tiim/.venv/bin/python -m tiim "$@"
SH
chmod +x /usr/local/bin/tiim

echo "==> Setting TIIM to run 24/7 (restarts itself if it ever crashes or the server reboots)"
install -m 644 "$APP/deploy/tiim.service" /etc/systemd/system/tiim.service
install -m 644 "$APP/deploy/tiim-dashboard.service" /etc/systemd/system/tiim-dashboard.service
install -m 644 "$APP/deploy/tiim-update.service" /etc/systemd/system/tiim-update.service
install -m 644 "$APP/deploy/tiim-update.timer" /etc/systemd/system/tiim-update.timer
chmod +x "$APP/deploy/update.sh"
systemctl daemon-reload
systemctl enable --now tiim.service tiim-dashboard.service tiim-update.timer

echo "==> Firewall: only SSH (8000 for the password-protected dashboard)"
ufw allow OpenSSH >/dev/null
ufw allow 8000/tcp >/dev/null
ufw --force enable >/dev/null

IP=$(curl -fsS https://api.ipify.org || hostname -I | awk '{print $1}')
echo
echo "TIIM is running."
echo "  Dashboard:  http://$IP:8000   (log in with any name + your dashboard password)"
echo "  Logs:       journalctl -u tiim -f"
echo "  Commands:   tiim status | tiim strategies | tiim news | tiim stop | tiim resume"
echo "  Updates:    automatic every 15 minutes (only applied if all checks pass)"
