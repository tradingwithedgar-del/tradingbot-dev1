#!/usr/bin/env bash
# Pulls new code from GitHub (your new strategies, fixes), runs TIIM's checks, and restarts TIIM
# only if they all pass. A broken update is rolled back automatically. Runs every 15 minutes.
set -uo pipefail
APP=/opt/tiim
cd "$APP"
as_tiim() { sudo -u tiim "$@"; }

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
