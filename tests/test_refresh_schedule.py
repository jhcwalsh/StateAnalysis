"""The scheduled refresh on the Mini: deploy/com.lazyeconomist.states.refresh.plist runs
scripts/refresh_states.sh, which must run the same engine command as the app's Refresh
button, take a lock, and alert on a non-zero exit."""
import plistlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "regime_v2"))

from regime_v2 import publish  # noqa: E402

PLIST = ROOT / "deploy" / "com.lazyeconomist.states.refresh.plist"
SCRIPT = ROOT / "scripts" / "refresh_states.sh"


def test_plist_schedules_the_script_on_the_tenth():
    with PLIST.open("rb") as f:
        p = plistlib.load(f)
    assert p["Label"] == "com.lazyeconomist.states.refresh"
    assert p["ProgramArguments"] == ["/Users/jameswalsh/apps/states/scripts/refresh_states.sh"]
    assert p["StartCalendarInterval"] == {"Day": 10, "Hour": 7, "Minute": 0}
    assert p["RunAtLoad"] is False
    assert p["StandardOutPath"].startswith("/Users/jameswalsh/apps/states/logs/")


def test_script_runs_the_apps_refresh_command():
    text = SCRIPT.read_text(encoding="utf-8")
    # The flags the dashboard's Refresh button passes (publish.refresh_command), in the container.
    expected = publish.refresh_command("python", "run.py", "VINTAGE", "/app/var/output", "/app/var/figs",
                                       "/app/var/returns_yfinance.parquet")
    for flag in expected[2:]:
        if flag == "VINTAGE":
            continue
        assert flag in text, flag
    assert 'docker" exec -w /app/regime_v2' in text.replace("$DOCKER", "docker")
    assert "date -v-1m +%Y-%m" in text  # the previous month, like publish.default_vintage


def test_script_locks_uses_absolute_binaries_and_alerts_on_failure():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash")
    assert "set -u" in text
    assert 'mkdir "$LOCK"' in text and "rmdir \"$LOCK\"" in text
    assert "DOCKER=/usr/local/bin/docker" in text and "CURL=/usr/bin/curl" in text
    assert "NTFY_TOPIC" in text and "https://ntfy.sh/$topic" in text
    # Every failure exit is preceded by a notify call.
    for step in ("container_check", "run_py"):
        assert re.search(rf"notify {step} .*\n\s*exit", text), step
    assert 'notify run_py "$rc"' in text
