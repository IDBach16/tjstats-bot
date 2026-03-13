"""MiLB traditional stats fetcher for non-AAA levels.

Fetches season stats, game logs, and monthly splits from the MLB Stats API
for levels that lack Statcast pitch-tracking data (AA, A+, A, Complex).

Data is cached per-level on disk as parquet so only new rosters need fetching.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, MLB_SEASON
from .milb_statcast import SPORT_IDS, LEVEL_NAMES

log = logging.getLogger(__name__)

# Cache dirs
_CACHE_DIR = DATA_DIR / "milb_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory caches
_roster_cache: dict[str, pd.DataFrame] = {}
_gamelog_cache: dict[int, list[dict]] = {}
_monthly_cache: dict[int, list[dict]] = {}

_STATS_API = "https://statsapi.mlb.com/api/v1"

# Traditional stat columns we extract from the API
_SEASON_STAT_COLS = [
    "era", "whip", "inningsPitched", "wins", "losses", "gamesPlayed",
    "gamesStarted", "saves", "strikeOuts", "baseOnBalls", "hits",
    "homeRuns", "hitByPitch", "earnedRuns", "runs", "avg", "obp", "ops",
    "slg", "strikeoutsPer9Inn", "walksPer9Inn", "hitsPer9Inn",
    "homeRunsPer9", "strikeoutWalkRatio", "groundOutsToAirouts",
    "strikePercentage", "pitchesPerInning", "battersFaced",
    "numberOfPitches", "wildPitches",
]


# ── Team/roster discovery ────────────────────────────────────────────

def _fetch_teams(sport_id: int, season: int = MLB_SEASON) -> list[dict]:
    """Get all teams at a level for a season."""
    try:
        resp = requests.get(
            f"{_STATS_API}/teams",
            params={"sportId": sport_id, "season": season},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("teams", [])
    except Exception:
        log.warning("Failed to fetch teams for sportId=%d", sport_id, exc_info=True)
        return []


def _fetch_pitcher_stats(player_id: int, sport_id: int,
                         season: int = MLB_SEASON) -> dict | None:
    """Fetch season pitching stats for a single player."""
    try:
        resp = requests.get(
            f"{_STATS_API}/people/{player_id}/stats",
            params={
                "stats": "season",
                "season": season,
                "group": "pitching",
                "sportId": sport_id,
            },
            timeout=15,
        )
        resp.raise_for_status()
        for sg in resp.json().get("stats", []):
            for split in sg.get("splits", []):
                return split.get("stat", {})
    except Exception:
        pass
    return None


def _fetch_roster_pitcher_ids(team_id: int, season: int = MLB_SEASON) -> list[dict]:
    """Fetch roster and return list of {id, name} for pitchers."""
    pitchers = []
    try:
        resp = requests.get(
            f"{_STATS_API}/teams/{team_id}/roster",
            params={"rosterType": "fullRoster", "season": season},
            timeout=15,
        )
        resp.raise_for_status()
        for p in resp.json().get("roster", []):
            pos = p.get("position", {})
            if pos.get("abbreviation") == "P" or pos.get("type") == "Pitcher":
                person = p.get("person", {})
                pitchers.append({
                    "id": person.get("id"),
                    "name": person.get("fullName", "Unknown"),
                })
    except Exception:
        log.debug("Roster fetch failed for team %d", team_id, exc_info=True)
    return pitchers


# ── Bulk roster fetch ────────────────────────────────────────────────

def fetch_level_pitchers(level: str = "AA",
                         season: int = MLB_SEASON) -> pd.DataFrame:
    """Fetch season stats for all pitchers at a MiLB level.

    Returns a DataFrame with one row per pitcher, with traditional stats
    plus player_id, pitcher_name, team columns.
    """
    key = f"trad_{level}_{season}"
    if key in _roster_cache:
        return _roster_cache[key]

    sport_id = SPORT_IDS.get(level, 12)
    parquet_file = _CACHE_DIR / f"trad_{level}_{season}.parquet"

    # Try loading from cache
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            log.info("Loaded %d cached traditional pitchers for %s %d",
                     len(df), level, season)
            _roster_cache[key] = df
            return df
        except Exception:
            log.warning("Failed to load trad parquet cache", exc_info=True)

    # Discover all teams at this level
    teams = _fetch_teams(sport_id, season)
    if not teams:
        return pd.DataFrame()

    log.info("Fetching rosters for %d %s teams", len(teams), level)

    # Get all pitcher IDs from all rosters
    all_pitchers: list[dict] = []  # {id, name, team, team_id}
    for t in teams:
        tid = t["id"]
        abbr = t.get("abbreviation", "")
        roster = _fetch_roster_pitcher_ids(tid, season)
        for p in roster:
            p["team"] = abbr
            p["team_id"] = tid
        all_pitchers.extend(roster)
        time.sleep(0.1)  # Be gentle with API

    log.info("Found %d pitchers across %d %s teams",
             len(all_pitchers), len(teams), level)

    # Fetch season stats for each pitcher in parallel
    rows: list[dict] = []
    fetched = 0

    def _fetch_one(p: dict) -> dict | None:
        stats = _fetch_pitcher_stats(p["id"], sport_id, season)
        if stats and stats.get("inningsPitched"):
            row = {"player_id": p["id"], "pitcher_name": p["name"],
                   "team": p["team"]}
            for col in _SEASON_STAT_COLS:
                val = stats.get(col)
                if val is not None:
                    try:
                        row[col] = float(val)
                    except (TypeError, ValueError):
                        row[col] = np.nan
                else:
                    row[col] = np.nan
            return row
        return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, p): p for p in all_pitchers}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rows.append(result)
                    fetched += 1
            except Exception:
                pass
            if (fetched) % 50 == 0 and fetched > 0:
                log.info("Progress: %d pitchers fetched for %s", fetched, level)

    if not rows:
        log.warning("No pitchers with stats found for %s %d", level, season)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Compute derived stats
    if "strikeOuts" in df.columns and "battersFaced" in df.columns:
        df["k_pct"] = np.where(
            df["battersFaced"] > 0,
            df["strikeOuts"] / df["battersFaced"], 0
        )
    if "baseOnBalls" in df.columns and "battersFaced" in df.columns:
        df["bb_pct"] = np.where(
            df["battersFaced"] > 0,
            df["baseOnBalls"] / df["battersFaced"], 0
        )
    if "k_pct" in df.columns and "bb_pct" in df.columns:
        df["k_minus_bb"] = df["k_pct"] - df["bb_pct"]

    # Estimate FIP: ((13*HR + 3*BB - 2*K) / IP) + 3.10
    if all(c in df.columns for c in ("homeRuns", "baseOnBalls",
                                      "strikeOuts", "inningsPitched")):
        ip = df["inningsPitched"].replace(0, np.nan)
        df["fip"] = (
            (13 * df["homeRuns"] + 3 * df["baseOnBalls"]
             - 2 * df["strikeOuts"]) / ip + 3.10
        )

    # Save cache
    try:
        df.to_parquet(parquet_file, index=False)
        log.info("Saved %d pitchers to %s", len(df), parquet_file)
    except Exception:
        log.warning("Failed to save trad parquet", exc_info=True)

    _roster_cache[key] = df
    return df


# ── Single pitcher data ──────────────────────────────────────────────

def fetch_game_log(player_id: int, level: str = "AA",
                   season: int = MLB_SEASON) -> list[dict]:
    """Fetch game-by-game pitching log for a player."""
    if player_id in _gamelog_cache:
        return _gamelog_cache[player_id]

    sport_id = SPORT_IDS.get(level, 12)
    entries = []
    try:
        resp = requests.get(
            f"{_STATS_API}/people/{player_id}/stats",
            params={
                "stats": "gameLog",
                "season": season,
                "group": "pitching",
                "sportId": sport_id,
            },
            timeout=15,
        )
        resp.raise_for_status()
        for sg in resp.json().get("stats", []):
            for split in sg.get("splits", []):
                stat = split.get("stat", {})
                entry = {
                    "date": split.get("date", ""),
                    "opponent": split.get("opponent", {}).get("name", ""),
                    "opp_abbr": split.get("opponent", {}).get("abbreviation", ""),
                }
                for col in ["inningsPitched", "era", "strikeOuts",
                            "baseOnBalls", "hits", "earnedRuns", "runs",
                            "homeRuns", "numberOfPitches", "strikes",
                            "whip", "wins", "losses"]:
                    val = stat.get(col)
                    try:
                        entry[col] = float(val) if val is not None else np.nan
                    except (TypeError, ValueError):
                        entry[col] = np.nan
                entries.append(entry)
    except Exception:
        log.warning("Game log fetch failed for %d", player_id, exc_info=True)

    _gamelog_cache[player_id] = entries
    return entries


def fetch_monthly_splits(player_id: int, level: str = "AA",
                         season: int = MLB_SEASON) -> list[dict]:
    """Fetch monthly pitching splits for a player."""
    if player_id in _monthly_cache:
        return _monthly_cache[player_id]

    sport_id = SPORT_IDS.get(level, 12)
    entries = []
    month_names = {
        4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
        8: "Aug", 9: "Sep", 10: "Oct",
    }
    try:
        resp = requests.get(
            f"{_STATS_API}/people/{player_id}/stats",
            params={
                "stats": "byMonth",
                "season": season,
                "group": "pitching",
                "sportId": sport_id,
            },
            timeout=15,
        )
        resp.raise_for_status()
        for sg in resp.json().get("stats", []):
            for split in sg.get("splits", []):
                stat = split.get("stat", {})
                month_num = split.get("month")
                try:
                    month_num = int(month_num) if month_num else None
                except (TypeError, ValueError):
                    month_num = None
                entry = {
                    "month": month_names.get(month_num, str(month_num or "?")),
                    "month_num": month_num or 0,
                }
                for col in ["inningsPitched", "era", "strikeOuts",
                            "baseOnBalls", "hits", "earnedRuns", "homeRuns",
                            "whip", "avg", "strikeoutsPer9Inn",
                            "walksPer9Inn", "groundOutsToAirouts"]:
                    val = stat.get(col)
                    try:
                        entry[col] = float(val) if val is not None else np.nan
                    except (TypeError, ValueError):
                        entry[col] = np.nan
                entries.append(entry)
    except Exception:
        log.warning("Monthly splits fetch failed for %d", player_id, exc_info=True)

    entries.sort(key=lambda e: e.get("month_num", 0))
    _monthly_cache[player_id] = entries
    return entries


# ── League averages ──────────────────────────────────────────────────

def get_league_averages(level: str = "AA",
                        season: int = MLB_SEASON,
                        min_ip: float = 20.0) -> dict:
    """Compute league average stats for qualified pitchers at a level."""
    df = fetch_level_pitchers(level=level, season=season)
    if df.empty:
        return {}

    qualified = df[df["inningsPitched"] >= min_ip]
    if qualified.empty:
        qualified = df[df["inningsPitched"] >= 10]
    if qualified.empty:
        return {}

    avgs = {}
    for col in ["era", "whip", "strikeoutsPer9Inn", "walksPer9Inn",
                "homeRunsPer9", "groundOutsToAirouts", "avg", "obp",
                "hitsPer9Inn", "strikeoutWalkRatio", "k_pct", "bb_pct",
                "k_minus_bb", "fip", "strikePercentage"]:
        if col in qualified.columns:
            vals = qualified[col].dropna()
            if not vals.empty:
                avgs[col] = float(vals.mean())
    return avgs


# ── Player picking ───────────────────────────────────────────────────

_mlb_ids_cache: set[int] | None = None


def _get_mlb_player_ids() -> set[int]:
    """Get MLB player IDs to exclude from MiLB picks."""
    try:
        from . import pitch_profiler
        mlb_df = pitch_profiler.get_season_pitchers()
        if mlb_df.empty:
            return set()
        pid_col = "pitcher_id" if "pitcher_id" in mlb_df.columns else "player_id"
        if pid_col in mlb_df.columns:
            return set(mlb_df[pid_col].dropna().astype(int).tolist())
    except Exception:
        log.warning("Could not load MLB IDs for filtering", exc_info=True)
    return set()


def pick_traditional_player(level: str = "AA",
                            season: int = MLB_SEASON,
                            min_ip: float = 30.0) -> dict | None:
    """Pick a random qualified pitcher from a non-AAA MiLB level."""
    global _mlb_ids_cache

    df = fetch_level_pitchers(level=level, season=season)
    if df.empty:
        return None

    if _mlb_ids_cache is None:
        _mlb_ids_cache = _get_mlb_player_ids()
        log.info("Loaded %d MLB IDs for traditional MiLB filtering",
                 len(_mlb_ids_cache))

    qualified = df[df["inningsPitched"] >= min_ip]
    if _mlb_ids_cache:
        qualified = qualified[~qualified["player_id"].isin(_mlb_ids_cache)]

    if qualified.empty:
        qualified = df[df["inningsPitched"] >= 15]
        if _mlb_ids_cache:
            qualified = qualified[~qualified["player_id"].isin(_mlb_ids_cache)]

    if qualified.empty:
        log.warning("No qualified traditional pitchers at %s", level)
        return None

    row = qualified.sample(1).iloc[0]
    pid = int(row["player_id"])

    # Fetch player info for hand/age
    info = {"p_throws": "R", "age": 0}
    try:
        resp = requests.get(
            f"{_STATS_API}/people/{pid}",
            params={"hydrate": "currentTeam"},
            timeout=10,
        )
        resp.raise_for_status()
        person = resp.json()["people"][0]
        info["p_throws"] = person.get("pitchHand", {}).get("code", "R")
        info["age"] = person.get("currentAge", 0)
    except Exception:
        pass

    return {
        "name": row.get("pitcher_name", "Unknown"),
        "id": pid,
        "team": row.get("team", ""),
        "level": level,
        "p_throws": info["p_throws"],
        "age": info["age"],
    }
