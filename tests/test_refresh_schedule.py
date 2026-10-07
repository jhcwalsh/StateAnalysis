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


def test_plist_schedules_the_script_daily():
    """FRED-MD's release day within the month drifts, so the job runs every day and decides for
    itself whether there is anything to do; a fixed day of the month would miss a late vintage
    for a whole month."""
    with PLIST.open("rb") as f:
        p = plistlib.load(f)
    assert p["Label"] == "com.lazyeconomist.states.refresh"
    assert p["ProgramArguments"] == ["/Users/jameswalsh/apps/states/scripts/refresh_states.sh"]
    assert p["StartCalendarInterval"] == {"Hour": 7, "Minute": 0}
    assert p["RunAtLoad"] is False
    assert p["StandardOutPath"].startswith("/Users/jameswalsh/apps/states/logs/")


def test_script_runs_the_apps_refresh_command():
    text = SCRIPT.read_text(encoding="utf-8")
    # The flags the dashboard's Refresh button passes (publish.refresh_command), in the container.
    expected = publish.refresh_command("python", "run.py", "VINTAGE", "/app/var/output", "/app/var/figs",
                                       "/app/var/returns_yfinance.parquet", "/app/var/vintages")
    assert "--vintage-archive" in expected
    for flag in expected[2:]:
        if flag == "VINTAGE":
            continue
        assert flag in text, flag
    assert 'docker" exec -w /app/regime_v2' in text.replace("$DOCKER", "docker")


def test_the_container_gives_the_refresh_button_the_scripts_archive():
    """The Refresh button reads REGIME_VINTAGE_ARCHIVE; the Dockerfile must set it to the archive
    the scheduled script passes, so both refreshes publish the real-time-vintage comparison."""
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "REGIME_VINTAGE_ARCHIVE=/app/var/vintages" in docker
    assert "--vintage-archive /app/var/vintages" in SCRIPT.read_text(encoding="utf-8")


def test_script_asks_the_engine_which_vintage_is_newest():
    """No calendar arithmetic: the script must not name a vintage the Fed may not have posted."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--if-newer" in text
    assert "date -v-1m" not in text            # the old guess, replaced by --vintage latest
    assert "${VINTAGE:-latest}" in text        # a manual run can still pin one


def test_script_is_quiet_when_there_is_no_new_vintage():
    """Exit 3 from run.py is the ordinary daily outcome: logged, never pushed, never a failure."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r'"\$rc" -eq 3', text)
    assert re.search(r'-eq 3.*\n(.*\n)*?\s*exit 0', text)


def test_script_pushes_when_a_new_month_publishes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "notify_published" in text
    assert "summary.json" in text              # the new month and state are read back from the run
    body = text.split("notify_published() {", 1)[1].split("\n}", 1)[0]
    assert '"default"' in body and '"high"' not in body   # not the failure alert's priority


def test_script_debounces_an_unreachable_source():
    """Exit 4 means no vintage downloaded at all. On a daily schedule that would otherwise push
    every morning for as long as the Fed's site is unreachable."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert re.search(r'"\$rc" -eq 4', text)
    assert "UNREACHABLE_STREAK" in text and "UNREACHABLE_ALERT_AFTER=3" in text


def test_script_locks_uses_absolute_binaries_and_alerts_on_failure():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash")
    assert "set -u" in text
    assert 'mkdir "$LOCK"' in text and "rmdir \"$LOCK\"" in text
    assert 'DOCKER="${DOCKER:-/usr/local/bin/docker}"' in text
    assert 'CURL="${CURL:-/usr/bin/curl}"' in text
    assert "NTFY_TOPIC" in text and "https://ntfy.sh/$topic" in text
    # Every failure exit is preceded by a notify call.
    for step in ("container_check", "run_py"):
        assert re.search(rf"notify {step} .*\n\s*exit", text), step
    assert 'notify run_py "$rc"' in text


def test_entrypoint_redraws_doc_figures_when_a_run_is_published():
    text = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    head, _, tail = text.partition("else")
    assert "run.py data/fredmd_2026-07.csv" in head            # first start: the pinned run, as before
    assert "scripts/redraw_doc_figures.py" in tail              # later starts: redraw from the volume
    assert "scripts/redraw_doc_figures.py" not in head
    assert "|| echo" in tail                                   # a failure never blocks streamlit
    assert text.rstrip().endswith('--server.port="${STREAMLIT_SERVER_PORT:-8505}"')
