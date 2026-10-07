"""Redraw the documentation figures from the published run, without an engine run.

The container entrypoint calls this on every start that finds a published run on the volume,
so a deploy that only changed docfigs.py (or theme.py) shows its figures at once. The twelve
engine figures need fitted objects and are refreshed by the next engine run instead.

Exit 0 when the figures were redrawn, 1 when there is no published run or the redraw failed
(the message goes to stderr; the entrypoint prints it and starts the app regardless).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "regime_v2"))

from regime_v2.publish import PublishedMissing, redraw_doc_figures  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.environ.get("REGIME_OUTPUT_DIR", str(ROOT / "regime_v2" / "output")))
    ap.add_argument("--figs-dir", default=os.environ.get("REGIME_FIGS_DIR", str(ROOT / "regime_v2" / "figs")))
    a = ap.parse_args(argv)
    try:
        names = redraw_doc_figures(a.out_dir, a.figs_dir)
    except PublishedMissing as e:
        print(f"no published run to redraw from: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"doc figures not redrawn: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    drawn = sorted(k for k, v in names.items() if v)
    skipped = sorted(k for k, v in names.items() if not v)
    print(f"doc figures redrawn in {a.figs_dir}: {', '.join(drawn)}" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
