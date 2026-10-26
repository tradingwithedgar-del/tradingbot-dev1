#!/usr/bin/env bash
# Pulls new code from GitHub (your new strategies, fixes), runs TIIM's checks, and restarts TIIM
# only if they all pass. A broken update is rolled back automatically. Runs every 15 minutes.
set -uo pipefail
APP=/opt/tiim
cd "$APP"
as_tiim() { sudo -u tiim "$@"; }
ensure_swap() {
  # A 2 GB swap file keeps small (1 GB) servers from running out of memory.
  if ! swapon --show 2>/dev/null | grep -q '/swapfile'; then
    if [ ! -f /swapfile ]; then
      fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
      chmod 600 /swapfile
      mkswap /swapfile >/dev/null
    fi
    swapon /swapfile && echo "swap enabled (2 GB)"
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    sysctl -q vm.swappiness=10
    grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
  fi
}
ensure_swap
sync_units() {
  # Install service files added by later versions (e.g. the nightly Strategy Lab). Runs every time,
  # so units arriving with an update are installed on the next check.
  local changed=0
  for unit in tiim-lab.service tiim-lab.timer; do
    if [ -f "$APP/deploy/$unit" ] && ! cmp -s "$APP/deploy/$unit" "/etc/systemd/system/$unit"; then
      install -m 644 "$APP/deploy/$unit" "/etc/systemd/system/$unit"
      changed=1
    fi
  done
  if [ "$changed" = 1 ]; then
    systemctl daemon-reload
    systemctl enable --now tiim-lab.timer
    echo "installed the Strategy Lab nightly timer"
  fi
}
sync_units

sync_agent_name() {
  # The agent's name (AGENT_NAME in .env) becomes the command you type and the service description.
  local name lower changed=0
  name=$(grep -E '^AGENT_NAME=' "$APP/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -cd 'A-Za-z0-9_-')
  [ -z "$name" ] && return 0
  lower=$(echo "$name" | tr 'A-Z' 'a-z')
  if [ "$lower" != "tiim" ] && [ ! -e "/usr/local/bin/$lower" ]; then
    ln -s /usr/local/bin/tiim "/usr/local/bin/$lower" && echo "command '$lower' ready"
  fi
  for unit in tiim tiim-dashboard tiim-update tiim-lab; do
    mkdir -p "/etc/systemd/system/$unit.service.d"
    want="[Unit]\nDescription=$name ($unit)\n"
    if [ "$(printf "$want")" != "$(cat "/etc/systemd/system/$unit.service.d/name.conf" 2>/dev/null)" ]; then
      printf "$want" > "/etc/systemd/system/$unit.service.d/name.conf"; changed=1
    fi
  done
  [ "$changed" = 1 ] && systemctl daemon-reload
  return 0
}
sync_agent_name

as_tiim git fetch --quiet origin || exit 0
BRANCH=$(as_tiim git rev-parse --abbrev-ref HEAD)
OLD=$(as_tiim git rev-parse HEAD)
NEW=$(as_tiim git rev-parse "origin/$BRANCH")
[ "$OLD" = "$NEW" ] && exit 0

echo "TIIM update: $OLD -> $NEW"
as_tiim git merge --ff-only --quiet "origin/$BRANCH" || { echo "update could not be applied cleanly"; exit 1; }
as_tiim "$APP/.venv/bin/pip" install -q -r requirements.txt
if as_tiim "$APP/.venv/bin/python" -m pytest -q -x >/tmp/tiim-update-tests.log 2>&1; then
  systemctl restart tiim.service tiim-dashboard.service
  echo "TIIM updated and restarted: $(as_tiim git log -1 --format=%s)"
else
  echo "Checks FAILED on the update - rolling back, TIIM keeps running the previous version"
  tail -20 /tmp/tiim-update-tests.log
  as_tiim git reset --hard --quiet "$OLD"
fi
