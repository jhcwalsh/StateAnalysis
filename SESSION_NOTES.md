# Session notes

Newest first. Rules for this file are in CLAUDE.md "Session notes".

## 4 Oct 2026 - CLAUDE.md and session notes set up

**Done:** Added CLAUDE.md with James's shared house rules, SESSION_NOTES.md, and the
`.claude/` hook that refuses a commit unless these notes go in with it.

**Why:** The same working rules in every repo, and a record of each session that the next
one can pick up from.

**Broken / unsure:** The dashboard tests pass only on Streamlit 1.59-1.60 (see CLAUDE.md "Run and
test"); `requirements.txt` allows anything from 1.50 to below 2. Engine suite 263 passed, 1 skipped.
`.gitignore` now ignores `.claude/*` except `settings.json` and `hooks/`, so the hook is tracked.

**Next:** Keep the project facts in CLAUDE.md current as the code changes.
