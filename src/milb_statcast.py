"""MiLB Statcast data fetcher and aggregator.

Uses Baseball Savant's CSV export with sportId parameter to get
MiLB pitch-level data in the exact same format as MLB Statcast.
Aggregates into season-level pitcher/pitch-type DataFrames
matching the Pitch Profiler format used by charts.py.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, MLB_SEASON

log = logging.getLogger(__name__)

# Sport IDs per level
SPORT_IDS = {
    "MLB": 1, "AAA": 11, "AA": 12, "A+": 13, "A": 14, "CL": 16, "DSL": 16,
}

LEVEL_NAMES = {
    "AAA": "Triple-A", "AA": "Double-A", "A+": "High-A",
    "A": "Single-A", "CL": "Complex", "DSL": "DSL",
}

# Noise pitch types to filter
_NOISE = {"PO", "IN", "EP", "AB", "AS", "UN", "XX", "NP", "SC", ""}

# Cache dir
_CACHE_DIR = DATA_DIR / "milb_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory caches
_raw_cache: dict[str, pd.DataFrame] = {}
_season_pitchers_cache: dict[str, pd.DataFrame] = {}
_season_pitches_cache: dict[str, pd.DataFrame] = {}

# Baseball Savant search endpoint (same as pybaseball uses for MLB)
_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


# ── Fetch raw pitch data from Savant ──────────────────────────────────

def _fetch_savant_chunk(start_date: str, end_date: str,
                         sport_id: int) -> pd.DataFrame:
    """Fetch one date-range chunk from Baseball Savant CSV endpoint."""
    params = {
        "all": "true",
        "type": "details",
        "hfGT": "R|",
        "hfSea": f"{start_date[:4]}|",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "player_type": "pitcher",
        "sportId": sport_id,
        "csv": "true",
    }
    log.info("Fetching Savant MiLB data: %s to %s (sportId=%d)",
             start_date, end_date, sport_id)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; tjstats-bot/1.0)",
    }

    for attempt in range(3):
        try:
            resp = requests.get(_SAVANT_URL, params=params, headers=headers,
                                timeout=120)
            resp.raise_for_status()

            if len(resp.text.strip()) < 100:
                log.warning("Empty/small response for %s to %s", start_date, end_date)
                return pd.DataFrame()

            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            log.info("Got %d rows for %s to %s", len(df), start_date, end_date)
            return df

        except Exception as e:
            log.warning("Savant fetch attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    return pd.DataFrame()


def fetch_milb_season(level: str = "AAA",
                      season: int = MLB_SEASON) -> pd.DataFrame:
    """Fetch a full season of MiLB Statcast data from Baseball Savant.

    Returns a DataFrame with the same columns as MLB Statcast.
    Caches to disk as parquet.
    """
    key = f"{level}_{season}"
    if key in _raw_cache:
        return _raw_cache[key]

    cache_file = _CACHE_DIR / f"milb_{level}_{season}.parquet"
    if cache_file.exists():
        log.info("Loading cached MiLB data: %s", cache_file)
        df = pd.read_parquet(cache_file)
        _raw_cache[key] = df
        return df

    sport_id = SPORT_IDS.get(level, 11)

    # Fetch in monthly chunks to avoid Savant timeouts
    start = date(season, 4, 1)
    end = date(season, 9, 30)

    # If mid-season, use yesterday as end
    today = date.today()
    if today.year == season and today.month <= 9:
        end = today - timedelta(days=1)

    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=13), end)
        df = _fetch_savant_chunk(
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            sport_id,
        )
        if not df.empty:
            chunks.append(df)
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(2)  # Rate limit

    if not chunks:
        log.warning("No MiLB data returned for %s %s", level, season)
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)

    # Save cache
    try:
        result.to_parquet(cache_file, index=False)
        log.info("Cached %d pitches to %s", len(result), cache_file)
    except Exception:
        log.warning("Failed to cache MiLB data", exc_info=True)

    _raw_cache[key] = result
    return result


# ── Fetch data for a single pitcher ──────────────────────────────────

def fetch_milb_pitcher(pitcher_id: int, level: str = "AAA",
                        season: int = MLB_SEASON) -> pd.DataFrame | None:
    """Fetch Statcast data for a single MiLB pitcher.

    Works like pybaseball.statcast_pitcher but for MiLB.
    """
    raw = fetch_milb_season(level=level, season=season)
    if raw.empty:
        return None

    pid_col = "pitcher" if "pitcher" in raw.columns else "pitcher_id"
    pitcher_df = raw[raw[pid_col] == pitcher_id]

    if pitcher_df.empty:
        return None
    return pitcher_df.copy()


# ── Player info lookup ────────────────────────────────────────────────

_player_cache: dict[int, dict] = {}


def _fetch_player_info(player_id: int) -> dict:
    """Fetch player name, team, etc. from MLB Stats API."""
    if player_id in _player_cache:
        return _player_cache[player_id]

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=currentTeam"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        person = resp.json()["people"][0]
        info = {
            "name": person.get("fullName", "Unknown"),
            "team": person.get("currentTeam", {}).get("abbreviation", ""),
            "team_name": person.get("currentTeam", {}).get("name", ""),
            "p_throws": person.get("pitchHand", {}).get("code", "R"),
            "age": person.get("currentAge", 0),
        }
        _player_cache[player_id] = info
        return info
    except Exception:
        log.warning("Failed to fetch player info for %s", player_id)
        return {"name": "Unknown", "team": "", "team_name": "",
                "p_throws": "R", "age": 0}


# ── Aggregation: pitch-type level stats ───────────────────────────────

def aggregate_pitch_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw Statcast data into per-pitcher, per-pitch-type stats.

    Returns a DataFrame with columns matching Pitch Profiler format.
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # Identify column names (Savant format)
    pid_col = "pitcher" if "pitcher" in df.columns else "pitcher_id"
    pt_col = "pitch_type"
    name_col = "player_name" if "player_name" in df.columns else None

    # Filter noise
    df = df[df[pt_col].notna() & ~df[pt_col].isin(_NOISE)]
    if df.empty:
        return pd.DataFrame()

    # Compute per-pitch flags
    swing_descs = {
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    }
    whiff_descs = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

    if "description" in df.columns:
        desc_lower = df["description"].str.lower().str.replace(" ", "_")
        df["_is_swing"] = desc_lower.isin(swing_descs)
        df["_is_whiff"] = desc_lower.isin(whiff_descs)
        df["_is_called_strike"] = desc_lower == "called_strike"
    else:
        df["_is_swing"] = False
        df["_is_whiff"] = False
        df["_is_called_strike"] = False

    # Zone check
    if all(c in df.columns for c in ("plate_x", "plate_z", "sz_top", "sz_bot")):
        df["_in_zone"] = (
            (df["plate_x"].abs() <= 0.83) &
            (df["plate_z"].between(df["sz_bot"], df["sz_top"]))
        )
    else:
        df["_in_zone"] = True

    df["_is_chase"] = df["_is_swing"] & ~df["_in_zone"]

    # Break columns: Savant uses pfx_x/pfx_z in feet, convert to inches
    if "pfx_x" in df.columns:
        med = df["pfx_x"].dropna().abs().median()
        if med < 2:  # in feet
            df["hb"] = df["pfx_x"] * 12
            df["ivb"] = df["pfx_z"] * 12
        else:  # already inches
            df["hb"] = df["pfx_x"]
            df["ivb"] = df["pfx_z"]
    else:
        df["hb"] = np.nan
        df["ivb"] = np.nan

    # Total pitches per pitcher
    total_per_pitcher = df.groupby(pid_col).size().rename("_total")

    grouped = df.groupby([pid_col, pt_col])

    agg = grouped.agg(
        count=(pt_col, "size"),
        velocity=("release_speed", "mean"),
        spin_rate=("release_spin_rate" if "release_spin_rate" in df.columns
                   else "spin_rate" if "spin_rate" in df.columns
                   else pid_col, lambda x: x.mean() if x.dtype != object else np.nan),
        hb=("hb", "mean"),
        ivb=("ivb", "mean"),
        release_extension=("release_extension", "mean"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        chases=("_is_chase", "sum"),
        out_of_zone=("_in_zone", lambda x: (~x).sum()),
    ).reset_index()

    # Player name
    if name_col:
        name_map = df.groupby(pid_col)[name_col].first()
        agg["pitcher_name"] = agg[pid_col].map(name_map)
    else:
        agg["pitcher_name"] = agg[pid_col].apply(
            lambda x: _fetch_player_info(int(x))["name"]
        )

    # Handedness
    if "p_throws" in df.columns:
        hand_map = df.groupby(pid_col)["p_throws"].first()
        agg["p_throws"] = agg[pid_col].map(hand_map)

    # Team
    team_cols = [c for c in ("home_team", "away_team") if c in df.columns]
    if team_cols:
        # Use the team the pitcher appeared on most
        if "pitcher_team" in df.columns:
            team_map = df.groupby(pid_col)["pitcher_team"].first()
        else:
            team_map = df.groupby(pid_col)[team_cols[0]].first()
        agg["team"] = agg[pid_col].map(team_map)

    # Join total pitches
    agg = agg.merge(total_per_pitcher.reset_index(), on=pid_col)

    # Compute rates
    agg["percentage_thrown"] = agg["count"] / agg["_total"]
    agg["whiff_rate"] = np.where(agg["swings"] > 0,
                                  agg["whiffs"] / agg["swings"], 0)
    agg["chase_percentage"] = np.where(agg["out_of_zone"] > 0,
                                        agg["chases"] / agg["out_of_zone"], 0)

    # Rename for compatibility
    agg = agg.rename(columns={pid_col: "player_id"})

    # Clean up
    agg = agg.drop(columns=["_total", "swings", "whiffs", "chases",
                             "out_of_zone"], errors="ignore")

    return agg


# ── Aggregation: season-level pitcher stats ───────────────────────────

def aggregate_pitcher_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw Statcast data into per-pitcher season stats.

    Returns a DataFrame matching Pitch Profiler's season pitcher format.
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    pid_col = "pitcher" if "pitcher" in df.columns else "pitcher_id"
    name_col = "player_name" if "player_name" in df.columns else None

    # Filter noise
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(_NOISE)]
    if df.empty:
        return pd.DataFrame()

    # Compute flags
    swing_descs = {
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    }
    whiff_descs = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

    if "description" in df.columns:
        desc_lower = df["description"].str.lower().str.replace(" ", "_")
        df["_is_swing"] = desc_lower.isin(swing_descs)
        df["_is_whiff"] = desc_lower.isin(whiff_descs)
        df["_is_called_strike"] = desc_lower == "called_strike"
    else:
        df["_is_swing"] = False
        df["_is_whiff"] = False
        df["_is_called_strike"] = False

    # Zone check
    if all(c in df.columns for c in ("plate_x", "plate_z", "sz_top", "sz_bot")):
        df["_in_zone"] = (
            (df["plate_x"].abs() <= 0.83) &
            (df["plate_z"].between(df["sz_bot"], df["sz_top"]))
        )
    else:
        df["_in_zone"] = True

    df["_is_chase"] = df["_is_swing"] & ~df["_in_zone"]

    # Strikeout/walk detection from events
    if "events" in df.columns:
        df["_is_strikeout"] = df["events"].str.lower().str.contains(
            "strikeout", na=False
        )
        df["_is_walk"] = df["events"].str.lower().isin(
            ["walk", "intent_walk"]
        )
    else:
        df["_is_strikeout"] = False
        df["_is_walk"] = False

    # Batters faced = unique at-bat events
    ab_col = "at_bat_number" if "at_bat_number" in df.columns else "event_index"
    game_col = "game_pk" if "game_pk" in df.columns else "game_id"

    grouped = df.groupby(pid_col)

    agg = grouped.agg(
        total_pitches=("pitch_type", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        called_strikes=("_is_called_strike", "sum"),
        chases=("_is_chase", "sum"),
        out_of_zone=("_in_zone", lambda x: (~x).sum()),
        strikeouts=("_is_strikeout", "sum"),
        walks=("_is_walk", "sum"),
    ).reset_index()

    # Batters faced (unique game + at_bat combos)
    if game_col in df.columns and ab_col in df.columns:
        bf = df.groupby(pid_col).apply(
            lambda x: x[[game_col, ab_col]].drop_duplicates().shape[0]
        ).rename("batters_faced")
        agg = agg.merge(bf.reset_index(), on=pid_col)
    else:
        agg["batters_faced"] = agg["total_pitches"] / 4  # rough estimate

    # Player name
    if name_col:
        name_map = df.groupby(pid_col)[name_col].first()
        agg["pitcher_name"] = agg[pid_col].map(name_map)
    else:
        agg["pitcher_name"] = agg[pid_col].apply(
            lambda x: _fetch_player_info(int(x))["name"]
        )

    # Handedness & team
    if "p_throws" in df.columns:
        agg["p_throws"] = agg[pid_col].map(df.groupby(pid_col)["p_throws"].first())

    team_cols = [c for c in ("home_team", "away_team", "pitcher_team") if c in df.columns]
    if team_cols:
        agg["team"] = agg[pid_col].map(df.groupby(pid_col)[team_cols[-1]].first())

    # Compute rates
    agg["strike_out_percentage"] = np.where(
        agg["batters_faced"] > 0, agg["strikeouts"] / agg["batters_faced"], 0
    )
    agg["walk_percentage"] = np.where(
        agg["batters_faced"] > 0, agg["walks"] / agg["batters_faced"], 0
    )
    agg["whiff_rate"] = np.where(
        agg["swings"] > 0, agg["whiffs"] / agg["swings"], 0
    )
    agg["chase_percentage"] = np.where(
        agg["out_of_zone"] > 0, agg["chases"] / agg["out_of_zone"], 0
    )
    agg["called_strikes_plus_whiffs_percentage"] = np.where(
        agg["total_pitches"] > 0,
        (agg["whiffs"] + agg["called_strikes"]) / agg["total_pitches"], 0
    )

    # Rename & filter
    agg = agg.rename(columns={pid_col: "player_id"})
    agg = agg[agg["total_pitches"] >= 100]  # minimum threshold

    # Clean up internal columns
    agg = agg.drop(columns=["swings", "whiffs", "called_strikes", "chases",
                             "out_of_zone", "strikeouts", "walks"],
                    errors="ignore")

    return agg


# ── Public API ────────────────────────────────────────────────────────

def get_milb_season_pitchers(level: str = "AAA",
                              season: int = MLB_SEASON) -> pd.DataFrame:
    """Get season-level MiLB pitcher stats (like pitch_profiler.get_season_pitchers)."""
    key = f"{level}_{season}"
    if key in _season_pitchers_cache:
        return _season_pitchers_cache[key]

    raw = fetch_milb_season(level=level, season=season)
    if raw.empty:
        return pd.DataFrame()

    result = aggregate_pitcher_stats(raw)
    _season_pitchers_cache[key] = result
    return result


def get_milb_season_pitches(level: str = "AAA",
                             season: int = MLB_SEASON) -> pd.DataFrame:
    """Get pitch-type level MiLB stats (like pitch_profiler.get_season_pitches)."""
    key = f"{level}_{season}"
    if key in _season_pitches_cache:
        return _season_pitches_cache[key]

    raw = fetch_milb_season(level=level, season=season)
    if raw.empty:
        return pd.DataFrame()

    result = aggregate_pitch_stats(raw)
    _season_pitches_cache[key] = result
    return result


def pick_milb_player(level: str = "AAA",
                     season: int = MLB_SEASON,
                     min_pitches: int = 200) -> dict | None:
    """Pick a random MiLB pitcher with enough data for a card."""
    import random

    pitchers = get_milb_season_pitchers(level=level, season=season)
    if pitchers.empty:
        return None

    qualified = pitchers[pitchers["total_pitches"] >= min_pitches]
    if qualified.empty:
        qualified = pitchers

    row = qualified.sample(1).iloc[0]
    pid = int(row["player_id"])
    info = _fetch_player_info(pid)

    return {
        "name": row.get("pitcher_name", info["name"]),
        "id": pid,
        "team": row.get("team", info.get("team", "")),
        "level": level,
    }
