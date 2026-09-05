#!/bin/bash
# ------------------------------------------------------------------------
# Scheduled refresh of the States site -- Mac Mini only.
#
# Runs the engine inside the `states` container for the previous month's
# FRED-MD vintage, exactly the command the app's Refresh button runs. The
# engine publishes only if the vintage checks out and the acceptance gate
# passes; otherwise it exits non-zero and the site keeps serving the last
# good run. A non-zero exit here pushes an alert to the ntfy topic named in
# ~/apps/states/.refresh.env (NTFY_TOPIC=...), with the tail of the log.
# Success is logged, not pushed.
#
# The app's own Refresh button takes a lock inside the container; this
# script does not share it. A button press during the ~10-minute scheduled
# run would start a second engine run -- harmless, since publication is
# atomic, but wasteful.
#
# Manual run:  ~/apps/states/scripts/refresh_states.sh
#              VINTAGE=2026-08 ~/apps/states/scripts/refresh_states.sh
# Scheduled:   ~/Library/LaunchAgents/com.lazyeconomist.states.refresh.plist
#              (from deploy/; 07:00 local on the 10th of each month)
# ------------------------------------------------------------------------
set -u

APP_DIR="$HOME/apps/states"
TASK="refresh"
LOG_DIR="$APP_DIR/logs"
LOG="$LOG_DIR/$TASK.log"
LOCK="$LOG_DIR/$TASK.lock"
ENV_FILE="$APP_DIR/.refresh.env"
CONTAINER="states"

# Absolute paths: launchd and non-interactive SSH do not source .zprofile, so a
# bare `docker` fails with "command not found". /usr/local/bin/docker is
# OrbStack's system-wide symlink.
DOCKER=/usr/local/bin/docker
CURL=/usr/bin/curl

mkdir -p "$LOG_DIR"

# The previous month: the same rule as publish.default_vintage in the app.
VINTAGE="${VINTAGE:-$(date -v-1m +%Y-%m)}"

stamp() { date -u +%FT%TZ; }

notify() {
    # $1 = step name, $2 = exit code. Without a topic the failure is only logged.
    local topic=""
    if [ -f "$ENV_FILE" ]; then
        topic=$(sed -n 's/^NTFY_TOPIC=//p' "$ENV_FILE" | tr -d "\"' \r")
    fi
    if [ -z "$topic" ]; then
        echo "[$(stamp)] no NTFY_TOPIC in $ENV_FILE; alert not sent" >> "$LOG"
        return 0
    fi
    local body
    body=$(printf 'States refresh failed at %s (vintage %s): step %s exited %s.\nThe site is still serving the last good run. Log: %s\n\n--- last 40 log lines ---\n%s\n' \
        "$(stamp)" "$VINTAGE" "$1" "$2" "$LOG" "$(tail -n 40 "$LOG")")
    if ! "$CURL" -s -o /dev/null -m 20 \
        -H "Title: States refresh failed ($VINTAGE)" -H "Priority: high" -H "Tags: warning" \
        --data-binary "$body" "https://ntfy.sh/$topic"; then
        echo "[$(stamp)] ntfy push failed" >> "$LOG"
    fi
}

# mkdir is atomic and exists on stock macOS (flock does not). launchd will not
# start a second copy of a running job; this covers a manual run overlapping it.
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(stamp)] $TASK already running, exiting" >> "$LOG"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

{
    echo ""
    echo "=== [$(stamp)] Starting $TASK for vintage $VINTAGE ==="
} >> "$LOG"

if ! "$DOCKER" ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "[$(stamp)] container $CONTAINER is not running" >> "$LOG"
    notify container_check 1
    exit 1
fi

# Capture the status into a variable rather than testing with `if ! ...`: inside
# an `if ! cmd; then` body, $? is the status of the negation, always 0.
"$DOCKER" exec -w /app/regime_v2 "$CONTAINER" python run.py --vintage "$VINTAGE" \
    --out-dir /app/var/output --figs-dir /app/var/figs \
    --returns-cache /app/var/returns_yfinance.parquet --refresh-returns >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[$(stamp)] run.py exited $rc; the last good run stays published" >> "$LOG"
    notify run_py "$rc"
    exit "$rc"
fi

echo "=== [$(stamp)] $TASK completed for vintage $VINTAGE ===" >> "$LOG"
exit 0
