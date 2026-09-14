#!/bin/bash
# ------------------------------------------------------------------------
# Scheduled refresh of the States site -- Mac Mini only.
#
# Runs daily and lets the engine decide whether there is anything to do.
# FRED-MD does not post a vintage during its own month and its release day
# in the following month drifts (verified 2026-09-13: 2026-09 was still the
# site's HTML 404 while 2026-08 was real; archived snapshots of the FRED-MD
# page on 2025-10-17, 2026-01-16 and 2026-06-06 each showed the previous
# month as the newest listed). A fixed day of the month therefore has to
# guess a vintage, and misses a late one for a whole month. So this asks:
# `run.py --vintage latest --if-newer` probes backwards from the current
# month for the newest vintage that actually downloads and exits 3, in about
# two seconds, unless it is newer than the one already published.
#
# The engine publishes only if the vintage checks out and the acceptance gate
# passes; otherwise it exits non-zero and the site keeps serving the last
# good run.
#
# Exit codes from run.py, and what each does here:
#   0  published a new vintage  -> ntfy push naming the month and state
#   3  nothing new to publish   -> one log line, no push (the usual day)
#   4  no vintage downloaded    -> the source is unreachable; pushes only
#                                  after UNREACHABLE_ALERT_AFTER consecutive
#                                  days, then weekly, so an outage does not
#                                  alert every morning
#   *  the run failed           -> ntfy push with the tail of the log
# The topic lives in ~/apps/states/.refresh.env (NTFY_TOPIC=...), untracked.
#
# The app's own Refresh button takes a lock inside the container; this
# script does not share it. A button press during the ~10-minute run would
# start a second engine run -- harmless, since publication is atomic, but
# wasteful.
#
# Manual run:  ~/apps/states/scripts/refresh_states.sh
#              VINTAGE=2026-08 ~/apps/states/scripts/refresh_states.sh
#              (naming a vintage pins it and skips the --if-newer gate, so a
#              republish of the current vintage is still possible by hand)
# Scheduled:   ~/Library/LaunchAgents/com.lazyeconomist.states.refresh.plist
#              (from deploy/; 07:00 local every day)
# ------------------------------------------------------------------------
set -u

APP_DIR="${APP_DIR:-$HOME/apps/states}"
TASK="refresh"
LOG_DIR="$APP_DIR/logs"
LOG="$LOG_DIR/$TASK.log"
LOCK="$LOG_DIR/$TASK.lock"
ENV_FILE="$APP_DIR/.refresh.env"
CONTAINER="states"

# Absolute paths: launchd and non-interactive SSH do not source .zprofile, so a
# bare `docker` fails with "command not found". /usr/local/bin/docker is
# OrbStack's system-wide symlink. Overridable only so tests/test_refresh_script.py
# can drive the exit-code routing below against stubs; nothing sets these in
# production, where the defaults are the whole point.
DOCKER="${DOCKER:-/usr/local/bin/docker}"
CURL="${CURL:-/usr/bin/curl}"

mkdir -p "$LOG_DIR"

# Which vintage to ask for. `latest` lets run.py resolve it from what the Fed
# has actually posted; an explicit YYYY-MM pins it and bypasses the gate.
VINTAGE="${VINTAGE:-latest}"

# Consecutive days on which no vintage could be downloaded at all.
UNREACHABLE_STREAK="$LOG_DIR/$TASK.unreachable"
UNREACHABLE_ALERT_AFTER=3
UNREACHABLE_REPEAT_EVERY=7

stamp() { date -u +%FT%TZ; }

read_topic() {
    # Without a topic nothing can be pushed; the caller falls back to logging.
    if [ -f "$ENV_FILE" ]; then
        sed -n 's/^NTFY_TOPIC=//p' "$ENV_FILE" | tr -d "\"' \r"
    fi
}

push() {
    # $1 = title, $2 = priority, $3 = tags, $4 = body
    local topic
    topic=$(read_topic)
    if [ -z "$topic" ]; then
        echo "[$(stamp)] no NTFY_TOPIC in $ENV_FILE; push not sent" >> "$LOG"
        return 0
    fi
    if ! "$CURL" -s -o /dev/null -m 20 \
        -H "Title: $1" -H "Priority: $2" -H "Tags: $3" \
        --data-binary "$4" "https://ntfy.sh/$topic"; then
        echo "[$(stamp)] ntfy push failed" >> "$LOG"
    fi
}

notify() {
    # $1 = step name, $2 = exit code.
    push "States refresh failed ($VINTAGE)" "high" "warning" \
        "$(printf 'States refresh failed at %s (vintage %s): step %s exited %s.\nThe site is still serving the last good run. Log: %s\n\n--- last 40 log lines ---\n%s\n' \
            "$(stamp)" "$VINTAGE" "$1" "$2" "$LOG" "$(tail -n 40 "$LOG")")"
}

notify_published() {
    # Read the month and state back out of the run that was just published, so the
    # push says what landed rather than only that something did. python is in the
    # image; parsing summary.json with sed would break on any reformatting.
    local line
    line=$("$DOCKER" exec "$CONTAINER" python -c \
        'import json;s=json.load(open("/app/var/output/summary.json"));print(s["run"]["vintage"], s["current"]["month"], s["current"]["regime"])' \
        2>> "$LOG")
    if [ -z "$line" ]; then
        line="(summary.json unreadable)"
    fi
    echo "[$(stamp)] published: $line" >> "$LOG"
    push "States published a new month" "default" "chart_with_upwards_trend" \
        "$(printf 'States refreshed at %s.\nvintage / month / state: %s\nhttps://states.lazyeconomist.com\n' \
            "$(stamp)" "$line")"
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
if [ "$VINTAGE" = "latest" ]; then
    "$DOCKER" exec -w /app/regime_v2 "$CONTAINER" python run.py --vintage latest --if-newer \
        --out-dir /app/var/output --figs-dir /app/var/figs \
        --returns-cache /app/var/returns_yfinance.parquet --refresh-returns >> "$LOG" 2>&1
    rc=$?
else
    "$DOCKER" exec -w /app/regime_v2 "$CONTAINER" python run.py --vintage "$VINTAGE" \
        --out-dir /app/var/output --figs-dir /app/var/figs \
        --returns-cache /app/var/returns_yfinance.parquet --refresh-returns >> "$LOG" 2>&1
    rc=$?
fi

if [ "$rc" -eq 4 ]; then
    # No vintage downloaded at all, not even the published one: the Fed's site is
    # unreachable. Count the streak and push sparingly -- a daily schedule would
    # otherwise alert every morning for the length of an outage.
    streak=0
    if [ -f "$UNREACHABLE_STREAK" ]; then
        streak=$(cat "$UNREACHABLE_STREAK" 2>/dev/null || echo 0)
    fi
    streak=$((streak + 1))
    echo "$streak" > "$UNREACHABLE_STREAK"
    echo "[$(stamp)] no vintage could be downloaded ($streak consecutive days)" >> "$LOG"
    if [ "$streak" -eq "$UNREACHABLE_ALERT_AFTER" ] || \
       { [ "$streak" -gt "$UNREACHABLE_ALERT_AFTER" ] && \
         [ $(((streak - UNREACHABLE_ALERT_AFTER) % UNREACHABLE_REPEAT_EVERY)) -eq 0 ]; }; then
        notify source_unreachable "$rc"
    fi
    exit "$rc"
fi

# The source answered, whatever it said next; the outage streak is over.
rm -f "$UNREACHABLE_STREAK"

if [ "$rc" -eq 3 ]; then
    echo "=== [$(stamp)] $TASK: no new vintage, nothing to do ===" >> "$LOG"
    exit 0
fi

if [ "$rc" -ne 0 ]; then
    echo "[$(stamp)] run.py exited $rc; the last good run stays published" >> "$LOG"
    notify run_py "$rc"
    exit "$rc"
fi

notify_published
echo "=== [$(stamp)] $TASK completed for vintage $VINTAGE ===" >> "$LOG"
exit 0
