"""Find the pitcher (and pitch) worth a post, for each rotating idea.

The numbers are found here, in pandas, from the same Savant frames the
Pitching Summary card uses (savant_pitching). The model only explains a
finding; it never searches for one. Every rule below is one of Lance
Brozdowski's (see pitching_lens):

  throw_it_more   his best pitch by whiff and xwOBA is one he barely throws
                  ("throw your worst pitch less", usage as the cheapest lever)
  era_vs_skill    ERA a full run away from xERA ("a full-run gap demands an
                  explanation")
  best_offspeed   changeups/splitters by drop gap off the four-seam (drop
                  separation is the #1 offspeed trait)
  bat_missers     whiff rate furthest above its pitch-type average (whiffs
                  are the north star, always read against the pitch type)

Each finder returns candidates best-first, so a generator can skip anyone
already featured and still have someone to post.
"""

from __future__ import annotations

import logging

import pandas as pd

from .pitching_lens import WHIFF_AVG

log = logging.getLogger(__name__)

PITCH_NAMES = {
    "FF": "four-seam", "SI": "sinker", "FC": "cutter", "SL": "slider", "ST": "sweeper",
    "CU": "curveball", "KC": "knuckle curve", "SV": "slurve", "CH": "changeup",
    "FS": "splitter", "FO": "forkball",
}
# Enough pitches of one type that its whiff rate and xwOBA mean something.
MIN_TYPE_PITCHES = 150
_FASTBALLS = ("FF", "SI", "FC")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(float("nan"), index=df.index)


def _qualified(season_df: pd.DataFrame, pitches_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch rows for pitchers on the qualified season board."""
    ids = set(pd.to_numeric(season_df["pitcher_id"], errors="coerce").dropna().astype(int))
    p = pitches_df.copy()
    p["pid"] = pd.to_numeric(p["pitcher_id"], errors="coerce")
    p = p[p["pid"].isin(ids)]
    for c in ("percentage_thrown", "pitches_thrown", "whiff_rate", "est_woba", "velocity", "ivb", "hb"):
        p[c] = _num(p, c)
    if p["whiff_rate"].max() <= 1.5:
        p["whiff_rate"] = p["whiff_rate"] * 100
    return p


def throw_it_more(season_df: pd.DataFrame, pitches_df: pd.DataFrame) -> list[dict]:
    """A pitch used 5-22% of the time that beats his most-used pitch on both
    whiff rate (by 8+ points) and xwOBA (by .060+)."""
    p = _qualified(season_df, pitches_df)
    p = p[p["pitches_thrown"] >= MIN_TYPE_PITCHES]
    out = []
    for pid, grp in p.groupby("pid"):
        top = grp.loc[grp["percentage_thrown"].idxmax()]
        for _, r in grp.iterrows():
            if r["pitch_type"] == top["pitch_type"] or not (0.05 <= r["percentage_thrown"] <= 0.22):
                continue
            dw = r["whiff_rate"] - top["whiff_rate"]
            dx = top["est_woba"] - r["est_woba"]
            if dw >= 8 and dx >= 0.060:
                out.append({
                    "pitcher_id": int(pid), "pitcher_name": r["pitcher_name"], "pitch_type": r["pitch_type"],
                    "score": dw * (1 - r["percentage_thrown"]) + dx * 100,
                    "facts": {
                        "pitch": PITCH_NAMES.get(r["pitch_type"], r["pitch_type"]),
                        "usage": f"{r['percentage_thrown'] * 100:.1f}%",
                        "whiff": f"{r['whiff_rate']:.1f}%",
                        "xwOBA": f"{r['est_woba']:.3f}",
                        "most-used pitch": PITCH_NAMES.get(top["pitch_type"], top["pitch_type"]),
                        "most-used usage": f"{top['percentage_thrown'] * 100:.1f}%",
                        "most-used whiff": f"{top['whiff_rate']:.1f}%",
                        "most-used xwOBA": f"{top['est_woba']:.3f}",
                    },
                })
    return sorted(out, key=lambda d: -d["score"])


def era_vs_skill(xstats: pd.DataFrame, season_df: pd.DataFrame, min_pa: int = 300) -> list[dict]:
    """ERA at least a full run from xERA, biggest gap first."""
    x = xstats.copy()
    x["pa"], x["era"], x["xera"] = _num(x, "pa"), _num(x, "era"), _num(x, "xera")
    x = x[(x["pa"] >= min_pa)].dropna(subset=["era", "xera"])
    x["gap"] = x["era"] - x["xera"]
    x = x[x["gap"].abs() >= 1.0]
    x = x.loc[x["gap"].abs().sort_values(ascending=False).index]
    sb = season_df.assign(pid=pd.to_numeric(season_df["pitcher_id"], errors="coerce")).set_index("pid")
    out = []
    for _, r in x.iterrows():
        pid = int(r["player_id"])
        if pid not in sb.index:
            continue
        s = sb.loc[pid]
        facts = {"ERA": f"{r['era']:.2f}", "xERA": f"{r['xera']:.2f}",
                 "direction": "ERA worse than xERA (due for better luck)" if r["gap"] > 0
                 else "ERA better than xERA (regression watch)"}
        for col, label in (("strike_out_percentage", "K%"), ("walk_percentage", "BB%"),
                           ("barrel_rate", "Barrel%"), ("hard_hit_rate", "Hard-hit%")):
            v = pd.to_numeric(s.get(col), errors="coerce")
            if pd.notna(v):
                facts[label] = f"{v * 100:.1f}%"
        try:
            facts["K-BB%"] = f"{(float(s['strike_out_percentage']) - float(s['walk_percentage'])) * 100:.1f}%"
        except (TypeError, ValueError, KeyError):
            pass
        facts["batters faced"] = str(int(r["pa"]))
        out.append({"pitcher_id": pid, "pitcher_name": s["pitcher_name"], "pitch_type": "",
                    "score": abs(r["gap"]), "facts": facts})
    return out


def best_offspeed(season_df: pd.DataFrame, pitches_df: pd.DataFrame) -> list[dict]:
    """Changeups/splitters by drop gap off the four-seam (primary fastball if
    no four-seam), velocity gap kept for context."""
    p = _qualified(season_df, pitches_df)
    out = []
    for pid, grp in p.groupby("pid"):
        fbs = grp[grp["pitch_type"].isin(_FASTBALLS) & grp["ivb"].notna() & grp["velocity"].notna()]
        if fbs.empty:
            continue
        ff = fbs[fbs["pitch_type"] == "FF"]
        fb = ff.iloc[0] if not ff.empty else fbs.loc[fbs["percentage_thrown"].idxmax()]
        for _, r in grp[grp["pitch_type"].isin(("CH", "FS")) & (grp["pitches_thrown"] >= MIN_TYPE_PITCHES)].iterrows():
            if pd.isna(r["ivb"]) or pd.isna(r["velocity"]):
                continue
            drop, gap = fb["ivb"] - r["ivb"], fb["velocity"] - r["velocity"]
            out.append({
                "pitcher_id": int(pid), "pitcher_name": r["pitcher_name"], "pitch_type": r["pitch_type"],
                "score": drop,
                "facts": {"pitch": PITCH_NAMES[r["pitch_type"]], "drop gap": f"{drop:.1f} in",
                          "velocity gap": f"{gap:.1f} mph", "velocity": f"{r['velocity']:.1f} mph",
                          "fastball": f"{PITCH_NAMES.get(fb['pitch_type'])} {fb['velocity']:.1f} mph, IVB {fb['ivb']:.1f} in",
                          "IVB": f"{r['ivb']:.1f} in", "whiff": f"{r['whiff_rate']:.1f}%",
                          "usage": f"{r['percentage_thrown'] * 100:.1f}%",
                          "xwOBA": f"{r['est_woba']:.3f}" if pd.notna(r["est_woba"]) else "n/a"},
            })
    return sorted(out, key=lambda d: -d["score"])


def bat_missers(season_df: pd.DataFrame, pitches_df: pd.DataFrame, min_pitches: int = 200) -> list[dict]:
    """Whiff rate furthest above its pitch-type average (FF/SL/CH/FS only:
    the four types he gives an average for)."""
    p = _qualified(season_df, pitches_df)
    p = p[p["pitch_type"].isin(list(WHIFF_AVG)) & (p["pitches_thrown"] >= min_pitches)]
    out = []
    for _, r in p.iterrows():
        avg = WHIFF_AVG[r["pitch_type"]]
        out.append({
            "pitcher_id": int(r["pid"]), "pitcher_name": r["pitcher_name"], "pitch_type": r["pitch_type"],
            "score": r["whiff_rate"] - avg,
            "facts": {"pitch": PITCH_NAMES[r["pitch_type"]], "whiff": f"{r['whiff_rate']:.1f}%",
                      "whiff vs pitch-type norm": f"+{r['whiff_rate'] - avg:.1f} points",
                      "velocity": f"{r['velocity']:.1f} mph" if pd.notna(r["velocity"]) else "n/a",
                      "IVB": f"{r['ivb']:.1f} in" if pd.notna(r["ivb"]) else "n/a",
                      "HB": f"{r['hb']:.1f} in" if pd.notna(r["hb"]) else "n/a",
                      "usage": f"{r['percentage_thrown'] * 100:.1f}%",
                      "xwOBA": f"{r['est_woba']:.3f}" if pd.notna(r["est_woba"]) else "n/a"},
        })
    return sorted(out, key=lambda d: -d["score"])
