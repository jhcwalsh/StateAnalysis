import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from regime_v2 import figures as F, regimes as R, theme
from regime_v2.pipeline import run_pipeline
from regime_v2.walkforward import fit_hmm4_walkforward


@pytest.fixture(scope="module")
def ctx(vintage_path):
    res = run_pipeline(vintage_path)
    free = R.fit_free_hmm4(res.G, res.P, res.est_mask)
    gmm = R.fit_gmm4(res.G, res.P, res.est_mask)
    wf = fit_hmm4_walkforward(vintage_path, start="2008-06-01", end="2009-12-01")
    return res, free, gmm, wf


def _ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 10_000


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _corner(path):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((0, 0))


def test_every_engine_figure_saves_on_the_cream_ground(ctx, tmp_path):
    res, free, gmm, wf = ctx
    p = lambda n: str(tmp_path / n)
    F.fig1_factors_gaps(res, p("fig1.png"))
    F.fig2_regime_timeline(res, wf, free, gmm, p("fig2.png"))
    F.fig3_state_space(res, wf, p("fig3.png"))
    F.fig4_hmm_probabilities(res, wf, p("fig4.png"))
    F.fig5_revisions(res, p("fig5.png"))
    F.fig6_classifier_comparison(res, free, gmm, p("fig6.png"))
    F.fig7_walkforward(res, wf, p("fig7.png"))
    for n in range(1, 8):
        assert _corner(p(f"fig{n}.png")) == _hex_rgb(theme.BG), f"fig{n}"


def test_figures_module_has_no_colour_literals():
    # Every colour comes from regime_v2.theme or regimes.COLORS; the restyle must not erode.
    src = (Path(F.__file__)).read_text(encoding="utf-8")
    assert not re.search(r'"#[0-9a-fA-F]{6}"', src)
    assert not re.search(r'color="(grey|gray|black|white)"', src)


def test_themed_figure_leaves_rcparams_alone(ctx, tmp_path):
    import matplotlib.pyplot as plt
    res, _, _, wf = ctx
    before = plt.rcParams["figure.facecolor"]
    F.fig3_state_space(res, wf, str(tmp_path / "x.png"))
    assert plt.rcParams["figure.facecolor"] == before


def test_all_figures_render(ctx, tmp_path):
    res, free, gmm, wf = ctx
    p = lambda n: str(tmp_path / n)
    F.fig1_factors_gaps(res, p("fig1.png")); assert _ok(p("fig1.png"))
    F.fig2_regime_timeline(res, wf, free, gmm, p("fig2.png")); assert _ok(p("fig2.png"))
    F.fig3_state_space(res, wf, p("fig3.png")); assert _ok(p("fig3.png"))
    F.fig4_hmm_probabilities(res, wf, p("fig4.png")); assert _ok(p("fig4.png"))
    rev = F.fig5_revisions(res, p("fig5.png")); assert _ok(p("fig5.png"))
    assert set(rev) == {"corr_first_final", "noise_to_signal_rmse", "sign_agreement", "n"}
    F.fig6_classifier_comparison(res, free, gmm, p("fig6.png")); assert _ok(p("fig6.png"))
    lags = F.fig7_walkforward(res, wf, p("fig7.png")); assert _ok(p("fig7.png"))
    assert list(lags.columns) == ["peak", "first_low_growth_rt", "lag_months", "censored"]
    row = lags.set_index("peak").loc["2007-12"]
    assert row["censored"]            # the fixture window opens 2008-06, after the peak


def test_figures_accept_missing_walkforward(ctx, tmp_path):
    res, free, gmm, _ = ctx
    F.fig2_regime_timeline(res, None, free, gmm, str(tmp_path / "a.png"))
    F.fig3_state_space(res, None, str(tmp_path / "b.png"))
    F.fig4_hmm_probabilities(res, None, str(tmp_path / "c.png"))
    assert _ok(str(tmp_path / "c.png"))


def test_primary_caption_and_challenger_palette(ctx):
    res, free, gmm, wf = ctx
    assert "real-time" in F._primary(res, wf)[2] and "NOT" not in F._primary(res, wf)[2]
    assert "NOT real-time" in F._primary(res, None)[2]
    assert F._primary(res, None)[0].equals(res.hmm.labels_filtered)   # never the smoothed labels
    from matplotlib.colors import to_hex
    regime_hex = {c.lower() for c in R.COLORS.values()}
    assert not {to_hex(c).lower() for c in F._palette(gmm.cluster_names).values()} & regime_hex


def test_nber_lags_uncensored_and_missing():
    idx = pd.date_range("2007-06-01", "2009-12-01", freq="MS")
    lab = pd.Series("Goldilocks", index=idx)
    lab.loc["2008-03-01":"2009-06-01"] = "Contraction"
    lags = F.nber_lags(lab).set_index("peak")
    assert lags.loc["2007-12", "lag_months"] == 3 and not lags.loc["2007-12", "censored"]
    assert lags.loc["2001-03", "first_low_growth_rt"] is None and np.isnan(lags.loc["2001-03", "lag_months"])
    assert not lags.loc["2001-03", "censored"]


def test_fig12_rt_vintage_draws_two_strips(tmp_path):
    import numpy as np
    from regime_v2 import figures as F, rtvintage as RT
    from regime_v2.regimes import REGIMES
    idx = pd.date_range("2010-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(0)
    final = rng.choice(REGIMES, 60)
    rt_l = final.copy(); rt_l[::7] = "Contraction"
    df = pd.DataFrame({"hmm_walkforward": final, "growth_gap": rng.normal(size=60), "inflation_gap": rng.normal(size=60)},
                      index=pd.DatetimeIndex(idx, name="date"))
    probs = pd.DataFrame(0.0, index=df.index, columns=REGIMES)
    rtv = RT.RtVintageResult(labels=pd.Series(rt_l, index=df.index, name="hmm_rt_vintage"), probs=probs,
                             growth_gap=df["growth_gap"], inflation_gap=df["inflation_gap"],
                             provenance=pd.Series(["reconstructed"] * 30 + ["published"] * 30, index=df.index), start="2010-01")
    out = tmp_path / "fig12.png"
    F.fig12_rt_vintage(df, rtv, str(out))
    assert out.exists() and out.stat().st_size > 10_000
    assert _corner(str(out)) == _hex_rgb(theme.BG)


def test_asset_figures_save_on_the_cream_ground(tmp_path):
    idx = pd.date_range("2010-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.005, 0.03, (36, 3)), index=idx,
                        columns=["PIT_MaxSharpe", "Static_6040", "InSample_MaxSharpe_expost"])
    F.fig10_backtest_wealth(rets, str(tmp_path / "f10.png"))
    w = pd.DataFrame(rng.normal(0, 0.3, (36, 4)), index=idx, columns=list("ABCD"))
    F.fig11_pit_weights(w, str(tmp_path / "f11.png"))
    path_df = pd.DataFrame({"mu": rng.normal(0.05, 0.01, 36), "sigma": rng.normal(0.1, 0.01, 36)}, index=idx)
    F.fig9_mixture_6040(path_df, str(tmp_path / "f9.png"))
    table = pd.DataFrame({"ann_ret": [0.05, 0.02, -0.01, 0.03], "se_ann_ret": [0.01] * 4, "n": [20, 15, 10, 5]},
                         index=pd.MultiIndex.from_product([["SPY"], R.REGIMES], names=["asset", "regime"]))
    F.fig8_regime_returns(table, str(tmp_path / "f8.png"))
    for n in (8, 9, 10, 11):
        assert _corner(str(tmp_path / f"f{n}.png")) == _hex_rgb(theme.BG), n


def test_fig10_legend_is_backed_on_the_cream_ground(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    captured = {}
    real_close = plt.close
    def spy(fig=None):
        if fig is not None and "fig" not in captured:
            captured["fig"] = fig
            return          # keep it open for the assertion
        return real_close(fig)
    monkeypatch.setattr(plt, "close", spy)
    idx = pd.date_range("2010-01-01", periods=36, freq="MS")
    rets = pd.DataFrame(np.random.default_rng(2).normal(0.005, 0.03, (36, 3)), index=idx,
                        columns=["PIT_MaxSharpe", "Static_6040", "InSample_MaxSharpe_expost"])
    F.fig10_backtest_wealth(rets, str(tmp_path / "f10.png"))
    fig = captured["fig"]
    try:
        frame = fig.axes[0].get_legend().get_frame()
        assert frame.get_facecolor()[3] > 0, "legend box is transparent over the wealth curves"
        from matplotlib.colors import to_hex
        assert to_hex(frame.get_facecolor()[:3]) == theme.BG
    finally:
        real_close(fig)


def test_fig12_is_a_published_figure_name():
    from regime_v2 import publish
    assert "fig12_rt_vintage" in publish.FIGURES
