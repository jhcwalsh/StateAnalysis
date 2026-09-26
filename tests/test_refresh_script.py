"""scripts/refresh_states.sh actually run, against stub docker and curl.

The daily schedule makes the script's exit-code routing the thing that decides whether the site
ever updates and how often the phone buzzes, so it is exercised rather than grepped:
run.py's 0 / 3 / 4 / other become publish-and-push, silence, a debounced outage alert, and an
immediate failure alert. tests/test_refresh_schedule.py owns the contract with the plist and
with publish.refresh_command; this file owns the behaviour.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_states.sh"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on PATH")

SUMMARY_LINE = "fredmd_2026-09.csv 2026-08 Goldilocks"

FAKE_DOCKER = """#!/bin/bash
# `docker ps` lists the container; `docker exec -w ...` is the engine run, whose exit code the
# test dictates; `docker exec states python -c ...` is the read-back of summary.json.
case "$1" in
  ps) echo states ;;
  exec)
    if [ "$2" = "-w" ]; then exit "$ENGINE_RC"; fi
    echo "%s"
    ;;
esac
""" % SUMMARY_LINE

# One line per push: the failure body carries a 40-line log tail, so newlines are flattened
# to keep "how many pushes went out" countable.
FAKE_CURL = """#!/bin/bash
printf '%s' "$*" | tr '\\n' ' ' >> "$PUSH_LOG"
printf '\\n' >> "$PUSH_LOG"
"""


@pytest.fixture
def run_refresh(tmp_path):
    app = tmp_path / "apps" / "states"
    (app / "logs").mkdir(parents=True)
    (app / ".refresh.env").write_text("NTFY_TOPIC=states-test\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
        p = bin_dir / name
        p.write_text(body, encoding="utf-8", newline="\n")
        p.chmod(0o755)
    pushes = tmp_path / "pushes.txt"

    def run(engine_rc):
        env = dict(os.environ, APP_DIR=str(app), DOCKER=str(bin_dir / "docker"),
                   CURL=str(bin_dir / "curl"), ENGINE_RC=str(engine_rc), PUSH_LOG=str(pushes))
        env.pop("VINTAGE", None)
        proc = subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True)
        log = (app / "logs" / "refresh.log")
        return proc.returncode, (log.read_text(encoding="utf-8") if log.exists() else "")

    run.pushes = lambda: pushes.read_text(encoding="utf-8").splitlines() if pushes.exists() else []
    run.streak = app / "logs" / "refresh.unreachable"
    return run


def test_no_new_vintage_is_silent_and_succeeds(run_refresh):
    """The ordinary day: run.py exits 3, the job must not fail and must not push."""
    rc, log = run_refresh(3)
    assert rc == 0
    assert "no new vintage, nothing to do" in log
    assert run_refresh.pushes() == []


def test_a_published_month_pushes_what_landed(run_refresh):
    rc, log = run_refresh(0)
    assert rc == 0
    pushes = run_refresh.pushes()
    assert len(pushes) == 1
    assert SUMMARY_LINE in pushes[0]
    assert "Priority: high" not in pushes[0]
    assert "completed for vintage latest" in log


def test_an_engine_failure_alerts_immediately(run_refresh):
    rc, log = run_refresh(1)
    assert rc == 1
    pushes = run_refresh.pushes()
    assert len(pushes) == 1 and "Priority: high" in pushes[0]
    assert "the last good run stays published" in log


def test_an_unreachable_source_alerts_only_after_three_days(run_refresh):
    """Exit 4 on a daily schedule would otherwise push every morning of an outage."""
    for day in (1, 2):
        rc, _ = run_refresh(4)
        assert rc == 4
        assert run_refresh.pushes() == [], f"pushed on day {day}"
    rc, log = run_refresh(4)
    assert rc == 4
    assert len(run_refresh.pushes()) == 1
    assert "3 consecutive days" in log


def test_the_outage_streak_resets_once_the_source_answers(run_refresh):
    run_refresh(4); run_refresh(4)
    assert run_refresh.streak.read_text(encoding="utf-8").strip() == "2"
    run_refresh(3)
    assert not run_refresh.streak.exists()
    run_refresh(4); run_refresh(4)
    assert run_refresh.pushes() == []          # the count started over, so still no alert


def test_refresh_passes_the_vintage_archive():
    """Both engine invocations name the archive on the volume so the rt stage runs on the Mini."""
    script = (ROOT / "scripts" / "refresh_states.sh").read_text(encoding="utf-8")
    assert script.count("--vintage-archive /app/var/vintages") == 2
