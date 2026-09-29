"""Pitching data from Baseball Savant, shaped like the Pitch Profiler API.

WHY THIS EXISTS
---------------
`pitch_profiler.get_season_pitchers()` went stale. On 2026-09-29 -- the last day
of the regular season -- its 2026 board topped out at SEVEN games started and
47.3 innings, with a league-wide total of 8,095 IP against a real ~43,000. Every
figure in it is frozen somewhere around the first week of May. Sandy Alcantara
came back as 47.1 IP / 3.04 ERA; he actually threw 214.0 at 3.87.

It was not our request: the endpoint returns `hasMore: false`, 521 rows, one per
pitcher. The feed itself stopped updating. So every card captioned "2026 Pitching
Summary" since May has shown about six starts.

This module returns the same two DataFrames `charts.plot_pitching_summary()`
already expects -- same column names, same shapes -- from Savant instead, so the
renderer does not change.

WHAT IS LOST, AND WHY THAT IS THE RIGHT TRADE
---------------------------------------------
`stuff_plus`, `pitching_plus` and `location_plus` are Pitch Profiler's own
models. No public API has them and we are not going to approximate them: a
made-up Stuff+ is worse than none. They are simply absent from the frame, and
the card already handles a missing column by omitting that row.

Keeping the stale ones was the alternative and it is not a real option -- a
Stuff+ computed on seven May starts does not describe a season, and printing it
next to real September numbers is exactly the kind of thing this is fixing.

SOURCES
-------
  season rates   leaderboard/custom (one call, ~327 qualified pitchers)
  per-pitch      leaderboard/pitch-arsenal-stats  (usage, whiff, RV/100, woba)
  velocity       leaderboard/pitch-arsenals
  movement       leaderboard/pitch-movement       (IVB / HB / extension)

All four are public CSV endpoints; none needs a key, unlike the Patreon-gated
endpoint they replace.
"""

from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

from .config import MLB_SEASON

log = logging.getLogger(__name__)

_BASE = "https://baseballsavant.mlb.com/leaderboard"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TJStatsBot/1.0)"}
_TIMEOUT = 120

# Savant serves these as one CSV per call and they change once a day at most, so
# one process should fetch each at most once.
_CACHE: dict[str, pd.DataFrame] = {}

# Minimum pitches for a pitcher to appear at all. Low enough to keep relievers,
# high enough that a September call-up with two innings does not get a card.
# Applied in pandas, NOT as the URL's `min=` -- that parameter is a batters-faced
# threshold with its own scale, and `min=250` silently returns an EMPTY csv with
# HTTP 200. `min=50` is the widest useful net (327 pitchers); the real filter
# happens below where it can be reasoned about.
MIN_PITCHES = 250
_URL_MIN = 50

_SEASON_SELECTIONS = (
    "player_age,p_game,p_formatted_ip,pitch_count,p_era,p_win,p_loss,"
    "p_strikeout,p_walk,xba,xslg,xwoba,exit_velocity_avg,hard_hit_percent,"
    # oz_swing_percent, NOT out_zone_swing_percent. ** Savant echoes back any
    # column name you ask for, valid or not, filled with NaN ** -- so a wrong
    # field looks present and simply has no data. Chase% was silently empty on
    # the first build for exactly this reason. Check .notna().any(), not the
    # column's existence, when adding a selection.
    "whiff_percent,swing_percent,in_zone_percent,oz_swing_percent,"
    "pitch_hand,barrel_batted_rate"
)


def _get(url: str, key: str) -> pd.DataFrame:
    if key in _CACHE:
        return _CACHE[key]
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            _CACHE[key] = df
            log.info("Savant %s: %d rows", key, len(df))
            return df
        except Exception as exc:                       # noqa: BLE001
            log.warning("Savant %s failed (attempt %d): %s", key, attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    _CACHE[key] = pd.DataFrame()
    return _CACHE[key]


def _flip_name(s: str) -> str:
    """Savant writes 'Gausman, Kevin'; everything downstream wants 'Kevin Gausman'."""
    s = str(s).strip()
    if "," not in s:
        return s
    last, first = [p.strip() for p in s.split(",", 1)]
    return f"{first} {last}"


def _rate(series: pd.Series) -> pd.Series:
    """Savant rate -> a 0-1 FRACTION, because that is what Pitch Profiler gave.

    ** This is the whole reason the swap is a drop-in. ** Pitch Profiler's rate
    columns are fractions (whiff_rate maxes at 1.000, chase at 0.452) and every
    consumer multiplies by 100 to display them -- pitching_summary's reply line,
    charts.py's Chase% cell, and so on. Savant serves the same rates as 0-100.
    Handing those straight through produced "K%: 1711.4%" on the first dry run.

    Decided per column, not per value: a column whose maximum is over 1.5 is a
    percentage and gets divided. Per-value would mangle a legitimate 0.9%.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any() and s.max() > 1.5:
        return s / 100.0
    return s


# The movement board takes ONE pitch type per call. Leaving pitch_type blank does
# not mean "all" -- it silently returns FOUR-SEAMERS ONLY (735 rows, every one
# FF), so a naive call leaves every breaking ball on the card with no velocity or
# break at all. KC is absent from 2026 and returns an empty csv, which is fine.
_PITCH_TYPES = ("FF", "SI", "FC", "SL", "CU", "CH", "ST", "FS", "KC", "SV")


def _movement(season: int) -> "pd.DataFrame":
    """Movement for every pitch type, one request each, concatenated."""
    key = f"movement_all_{season}"
    if key in _CACHE:
        return _CACHE[key]
    frames = []
    for pt in _PITCH_TYPES:
        d = _get(f"{_BASE}/pitch-movement?year={season}&team=&min=10"
                 f"&pitch_type={pt}&hand=&csv=true", f"movement_{season}_{pt}")
        if not d.empty:
            frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("Savant movement %s: %d rows across %d pitch types",
             season, len(out), len(frames))
    _CACHE[key] = out
    return out


def _spin(season: int) -> "pd.DataFrame":
    """Average spin per pitcher per pitch type.

    The arsenals board is WIDE -- one column per pitch type (ff_avg_spin,
    sl_avg_spin, ...) -- so it is melted back to the long (pitcher, pitch_type)
    shape everything else here uses.
    """
    raw = _get(f"{_BASE}/pitch-arsenals?year={season}&min=10&type=avg_spin"
               f"&hand=&csv=true", f"arsenal_spin_{season}")
    if raw.empty or "pitcher" not in raw.columns:
        return pd.DataFrame(columns=["pitcher_id", "pitch_type", "spin_rate"])
    cols = [c for c in raw.columns if c.endswith("_avg_spin")]
    out = raw.melt(id_vars=["pitcher"], value_vars=cols,
                   var_name="pt", value_name="spin_rate")
    out["pitch_type"] = out["pt"].str.replace("_avg_spin", "", regex=False).str.upper()
    out["pitcher_id"] = pd.to_numeric(out["pitcher"], errors="coerce").astype("Int64")
    out["spin_rate"] = pd.to_numeric(out["spin_rate"], errors="coerce")
    return out.dropna(subset=["spin_rate"])[["pitcher_id", "pitch_type", "spin_rate"]]


def _batters_faced(season: int) -> "pd.Series":
    """pitcher_id -> batters faced, summed over his pitch types."""
    a = _get(
        f"{_BASE}/pitch-arsenal-stats?type=pitcher&pitchType=&year={season}"
        f"&team=&min=10&csv=true", f"arsenal_stats_{season}")
    if a.empty or "pa" not in a.columns:
        return pd.Series(dtype="float64")
    ids = pd.to_numeric(a["player_id"], errors="coerce").astype("Int64")
    return pd.to_numeric(a["pa"], errors="coerce").groupby(ids).sum()


def get_season_pitchers(season: int = MLB_SEASON) -> pd.DataFrame:
    """Season pitcher board, in the Pitch Profiler column vocabulary."""
    url = (f"{_BASE}/custom?year={season}&type=pitcher&filter=&min={_URL_MIN}"
           f"&selections={_SEASON_SELECTIONS}&chart=false&x=p_game&y=p_game"
           f"&r=no&chartType=beeswarm&csv=true")
    raw = _get(url, f"season_pitchers_{season}")
    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["pitcher_name"] = raw["last_name, first_name"].map(_flip_name)
    df["pitcher_id"] = pd.to_numeric(raw["player_id"], errors="coerce").astype("Int64")
    df["game_year"] = season
    df["p_throws"] = raw.get("pitch_hand")

    # Counting stats.
    df["innings_pitched"] = raw.get("p_formatted_ip")
    # "177.2" is 177 and two thirds, not 177.2 -- a decimal read of it understates
    # every workload on the card by up to a third of an inning.
    ip = pd.to_numeric(raw.get("p_formatted_ip"), errors="coerce")
    whole = ip.fillna(0).astype(int)
    outs = ((ip - whole) * 10).round().astype(int).clip(0, 2)
    df["ip_decimal"] = whole + outs / 3.0
    df["pitches_thrown"] = pd.to_numeric(raw.get("pitch_count"), errors="coerce")
    df["games_played"] = pd.to_numeric(raw.get("p_game"), errors="coerce")
    df["wins"] = pd.to_numeric(raw.get("p_win"), errors="coerce")
    df["losses"] = pd.to_numeric(raw.get("p_loss"), errors="coerce")
    df["strike_outs"] = pd.to_numeric(raw.get("p_strikeout"), errors="coerce")
    df["walks"] = pd.to_numeric(raw.get("p_walk"), errors="coerce")
    df["era"] = pd.to_numeric(raw.get("p_era"), errors="coerce")

    # K% and BB% need BATTERS FACED, and the custom board leaves p_k_percent /
    # p_bb_percent blank. Summing `pa` across a pitcher's pitch types on the
    # arsenal board gives it exactly -- checked against MLB StatsAPI:
    # Alcantara 894 vs 894, Sanchez 857 vs 857, K% within 0.3 points. (Pitchers
    # whose rarest pitch fell under the board's own min=10 lose a batter or two,
    # which moves a rate by hundredths.)
    #
    # Estimating it from innings instead -- IP*3 plus walks -- was the first
    # attempt and it is wrong: batters faced also includes every hit, so it
    # understates the denominator and inflates every K% on the card.
    bf = _batters_faced(season)
    faced = df["pitcher_id"].map(bf)
    df["batters_faced"] = faced
    df["strike_out_percentage"] = df["strike_outs"] / faced
    df["walk_percentage"] = df["walks"] / faced

    df["xba"] = pd.to_numeric(raw.get("xba"), errors="coerce")
    df["xslg"] = pd.to_numeric(raw.get("xslg"), errors="coerce")
    df["woba"] = pd.to_numeric(raw.get("xwoba"), errors="coerce")
    df["avg_exit_velo"] = pd.to_numeric(raw.get("exit_velocity_avg"), errors="coerce")
    df["hard_hit_rate"] = _rate(raw.get("hard_hit_percent"))
    df["barrel_rate"] = _rate(raw.get("barrel_batted_rate"))
    df["whiff_rate"] = _rate(raw.get("whiff_percent"))
    df["swing_rate"] = _rate(raw.get("swing_percent"))
    df["zone_rate"] = _rate(raw.get("in_zone_percent"))
    df["chase_percentage"] = _rate(raw.get("oz_swing_percent"))

    # CSW% (called strikes + whiffs) is not on this board. Left absent rather
    # than approximated -- the card omits a column it cannot find.
    df = df[df["pitches_thrown"].fillna(0) >= MIN_PITCHES]
    return df.dropna(subset=["pitcher_name"]).reset_index(drop=True)


def get_season_pitches(season: int = MLB_SEASON) -> pd.DataFrame:
    """Per-pitcher, per-pitch-type board, in the Pitch Profiler vocabulary."""
    arsenal = _get(
        f"{_BASE}/pitch-arsenal-stats?type=pitcher&pitchType=&year={season}"
        f"&team=&min=10&csv=true", f"arsenal_stats_{season}")
    movement = _movement(season)
    if arsenal.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["pitcher_name"] = arsenal["last_name, first_name"].map(_flip_name)
    df["pitcher_id"] = pd.to_numeric(arsenal["player_id"], errors="coerce").astype("Int64")
    df["pitch_type"] = arsenal["pitch_type"]
    df["pitch_name"] = arsenal.get("pitch_name")
    df["percentage_thrown"] = _rate(arsenal.get("pitch_usage"))
    df["pitches_thrown"] = pd.to_numeric(arsenal.get("pitches"), errors="coerce")
    df["whiff_rate"] = _rate(arsenal.get("whiff_percent"))
    df["run_value_per_100_pitches"] = pd.to_numeric(
        arsenal.get("run_value_per_100"), errors="coerce")
    df["woba"] = pd.to_numeric(arsenal.get("woba"), errors="coerce")
    df["xba"] = pd.to_numeric(arsenal.get("ba"), errors="coerce")
    # Per-pitch stats Savant has and Pitch Profiler did not. They take the place
    # of Stuff+, per-pitch extension and per-pitch chase on the card -- none of
    # which Savant publishes, so those columns would have been blank forever.
    df["k_percent"] = _rate(arsenal.get("k_percent"))
    df["put_away"] = _rate(arsenal.get("put_away"))
    df["hard_hit_rate"] = _rate(arsenal.get("hard_hit_percent"))
    df["est_woba"] = pd.to_numeric(arsenal.get("est_woba"), errors="coerce")

    if not movement.empty:
        m = pd.DataFrame({
            "pitcher_id": pd.to_numeric(movement["pitcher_id"], errors="coerce").astype("Int64"),
            "pitch_type": movement["pitch_type"],
            "velocity": pd.to_numeric(movement.get("avg_speed"), errors="coerce"),
            # Savant reports break in inches with gravity taken out on _z.
            "ivb": pd.to_numeric(movement.get("pitcher_break_z_induced",
                                              movement.get("pitcher_break_z")),
                                 errors="coerce"),
            "hb": pd.to_numeric(movement.get("pitcher_break_x"), errors="coerce"),
        })
        df = df.merge(m, on=["pitcher_id", "pitch_type"], how="left")

    spin = _spin(season)
    if not spin.empty:
        df = df.merge(spin, on=["pitcher_id", "pitch_type"], how="left")

    for col in ("velocity", "ivb", "hb", "spin_rate"):
        if col not in df.columns:
            df[col] = pd.NA
    return df.dropna(subset=["pitcher_name"]).reset_index(drop=True)


def clear_cache() -> None:
    _CACHE.clear()
