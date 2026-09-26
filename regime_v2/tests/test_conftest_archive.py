import pandas as pd

from regime_v2 import data as D


def test_make_archive_truncates_each_vintage_to_the_previous_month(make_archive, tmp_path):
    root = make_archive(tmp_path / "arch", ["2026-05", "2026-06", "2026-07"])
    for v, last in (("2026-05", "2026-04-01"), ("2026-06", "2026-05-01"), ("2026-07", "2026-06-01")):
        levels, _ = D.load_fredmd(str(root / f"fredmd_{v}.csv"), reference=None)
        assert levels.index[-1] == pd.Timestamp(last), v
        assert levels.index[0] == pd.Timestamp("1959-01-01")


def test_make_archive_applies_a_plant_to_one_vintage_only(make_archive, tmp_path):
    def double_indpro(raw):
        raw = raw.copy()
        raw.loc[1:, "INDPRO"] = (pd.to_numeric(raw.loc[1:, "INDPRO"]) * 2).to_numpy()
        return raw
    root = make_archive(tmp_path / "arch", ["2026-06", "2026-07"], plant={"2026-07": double_indpro})
    a, _ = D.load_fredmd(str(root / "fredmd_2026-06.csv"), reference=None)
    b, _ = D.load_fredmd(str(root / "fredmd_2026-07.csv"), reference=None)
    assert b.loc["2020-01-01", "INDPRO"] == 2 * a.loc["2020-01-01", "INDPRO"]
