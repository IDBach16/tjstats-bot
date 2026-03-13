"""MiLB Statcast data fetcher and aggregator.

Fetches real AAA pitch-level Statcast data from the MLB Stats API
game feeds. Each AAA game's live feed contains full tracking data
(velocity, spin rate, movement, extension, zone coordinates, etc.).

Data is cached per-game on disk (games are immutable once Final),
so only new games are fetched on each run.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
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

# Cache dirs
_CACHE_DIR = DATA_DIR / "milb_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_GAME_CACHE_DIR = _CACHE_DIR / "games"
_GAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory caches
_raw_cache: dict[str, pd.DataFrame] = {}
_season_pitchers_cache: dict[str, pd.DataFrame] = {}
_season_pitches_cache: dict[str, pd.DataFrame] = {}

# MLB Stats API
_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

# Map live-feed call descriptions to Savant-style description strings
_CALL_MAP = {
    "Called Strike": "called_strike",
    "Ball": "ball",
    "Ball In Dirt": "ball",
    "Foul": "foul",
    "Foul Tip": "foul_tip",
    "Foul Bunt": "foul_bunt",
    "Swinging Strike": "swinging_strike",
    "Swinging Strike (Blocked)": "swinging_strike_blocked",
    "Missed Bunt": "missed_bunt",
    "In play, out(s)": "hit_into_play",
    "In play, no out": "hit_into_play_no_out",
    "In play, run(s)": "hit_into_play_score",
    "Hit By Pitch": "hit_by_pitch",
    "Pitchout": "pitchout",
    "Bunt Foul Tip": "foul_tip",
    "Swinging Pitchout": "swinging_strike",
    "Automatic Ball": "ball",
    "Automatic Strike": "called_strike",
}


# ── Schedule fetching ────────────────────────────────────────────────

def _fetch_schedule(sport_id: int, start_date: str,
                    end_date: str) -> list[dict]:
    """Fetch game schedule from MLB Stats API for a date range.

    Returns list of dicts with game_pk, date, home_team, away_team.
    Only includes Final (completed) regular-season games.
    """
    games = []
    # API supports startDate/endDate but limit to ~30 days per call
    sd = date.fromisoformat(start_date)
    ed = date.fromisoformat(end_date)

    while sd <= ed:
        chunk_end = min(sd + timedelta(days=29), ed)
        try:
            resp = requests.get(
                f"{_STATS_API_BASE}/schedule",
                params={
                    "sportId": sport_id,
                    "startDate": sd.isoformat(),
                    "endDate": chunk_end.isoformat(),
                    "gameType": "R",
                },
                timeout=30,
            )
            resp.raise_for_status()
            for dt in resp.json().get("dates", []):
                for g in dt.get("games", []):
                    status = g.get("status", {}).get("detailedState", "")
                    if status == "Final":
                        games.append({
                            "game_pk": g["gamePk"],
                            "date": dt["date"],
                            "home_team": g["teams"]["home"]["team"]["name"],
                            "away_team": g["teams"]["away"]["team"]["name"],
                        })
        except Exception:
            log.warning("Schedule fetch failed for %s to %s",
                        sd.isoformat(), chunk_end.isoformat(), exc_info=True)

        sd = chunk_end + timedelta(days=1)

    log.info("Found %d completed games (sportId=%d) from %s to %s",
             len(games), sport_id, start_date, end_date)
    return games


# ── Game feed fetching with disk cache ───────────────────────────────

def _fetch_game_feed(game_pk: int) -> dict | None:
    """Fetch a game's live feed, using disk cache if available."""
    cache_file = _GAME_CACHE_DIR / f"{game_pk}.json"

    # Check disk cache
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Fetch from API
    for attempt in range(3):
        try:
            resp = requests.get(
                _GAME_FEED_URL.format(game_pk=game_pk),
                timeout=30,
            )
            resp.raise_for_status()
            feed = resp.json()

            # Cache to disk (only the parts we need to save space)
            try:
                cache_file.write_text(
                    json.dumps(feed, separators=(",", ":")),
                    encoding="utf-8",
                )
            except Exception:
                pass

            return feed
        except Exception as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            else:
                log.warning("Failed to fetch game feed %d: %s", game_pk, e)

    return None


# ── Pitch extraction from game feed ─────────────────────────────────

def _extract_pitches_from_feed(feed: dict) -> list[dict]:
    """Extract pitch-by-pitch rows from a live game feed JSON."""
    pitches = []

    game_data = feed.get("gameData", {})
    game_pk = game_data.get("game", {}).get("pk", 0)
    game_date = game_data.get("datetime", {}).get("officialDate", "")

    teams = game_data.get("teams", {})
    home_abbr = teams.get("home", {}).get("abbreviation", "")
    away_abbr = teams.get("away", {}).get("abbreviation", "")

    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

    for play in plays:
        matchup = play.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        pitcher_id = pitcher.get("id")
        pitcher_name = pitcher.get("fullName", "Unknown")
        p_throws = matchup.get("pitchHand", {}).get("code", "R")
        batter_id = matchup.get("batter", {}).get("id")

        about = play.get("about", {})
        ab_index = about.get("atBatIndex", 0)
        half = about.get("halfInning", "top")

        # Pitcher's team: if top of inning, home team pitches; bottom = away
        pitcher_team = home_abbr if half == "top" else away_abbr

        result = play.get("result", {})
        result_event = result.get("event", "")

        play_events = play.get("playEvents", [])
        pitch_events = [e for e in play_events if e.get("isPitch")]

        for i, event in enumerate(pitch_events):
            pd_ = event.get("pitchData", {})
            if not pd_.get("startSpeed"):
                continue  # skip pitches without tracking data

            brk = pd_.get("breaks", {})
            coords = pd_.get("coordinates", {})
            details = event.get("details", {})
            hit = event.get("hitData", {})
            count = event.get("count", {})

            call_desc = details.get("call", {}).get("description", "")
            description = _CALL_MAP.get(call_desc, call_desc.lower().replace(" ", "_"))

            pitch_type = details.get("type", {}).get("code", "")
            pitch_name = details.get("type", {}).get("description", "")

            # Determine if this is the last pitch of the PA
            is_last = (i == len(pitch_events) - 1)
            events_val = result_event if is_last else ""

            pitches.append({
                "game_pk": game_pk,
                "game_date": game_date,
                "pitcher": pitcher_id,
                "player_name": pitcher_name,
                "p_throws": p_throws,
                "batter": batter_id,
                "at_bat_number": ab_index,
                "pitch_type": pitch_type,
                "pitch_name": pitch_name,
                "description": description,
                "events": events_val if events_val else np.nan,
                "release_speed": pd_.get("startSpeed"),
                "release_spin_rate": brk.get("spinRate"),
                "pfx_x": brk.get("breakHorizontal"),
                "pfx_z": brk.get("breakVerticalInduced"),
                "release_extension": pd_.get("extension"),
                "plate_x": coords.get("pX"),
                "plate_z": coords.get("pZ"),
                "sz_top": pd_.get("strikeZoneTop"),
                "sz_bot": pd_.get("strikeZoneBottom"),
                "zone": pd_.get("zone"),
                "launch_speed": hit.get("launchSpeed"),
                "launch_angle": hit.get("launchAngle"),
                "hit_distance_sc": hit.get("totalDistance"),
                "home_team": home_abbr,
                "away_team": away_abbr,
                "pitcher_team": pitcher_team,
                "balls": count.get("balls", 0),
                "strikes": count.get("strikes", 0),
            })

    return pitches


# ── Fetch full season of MiLB data ──────────────────────────────────

def fetch_milb_season(level: str = "AAA",
                      season: int = MLB_SEASON) -> pd.DataFrame:
    """Fetch a full season of MiLB Statcast data from Stats API game feeds.

    Returns a pitch-level DataFrame. Uses incremental caching — only
    fetches game feeds for games not already in the cache.
    """
    key = f"{level}_{season}"
    if key in _raw_cache:
        return _raw_cache[key]

    sport_id = SPORT_IDS.get(level, 11)
    parquet_file = _CACHE_DIR / f"milb_{level}_{season}.parquet"

    # Load existing cached pitches
    existing_df = pd.DataFrame()
    existing_game_pks: set[int] = set()
    if parquet_file.exists():
        try:
            existing_df = pd.read_parquet(parquet_file)
            if "game_pk" in existing_df.columns:
                existing_game_pks = set(existing_df["game_pk"].unique())
            log.info("Loaded %d cached pitches (%d games) from %s",
                     len(existing_df), len(existing_game_pks), parquet_file)
        except Exception:
            log.warning("Failed to load cached parquet", exc_info=True)

    # Determine date range
    start = date(season, 4, 1)
    end = date(season, 9, 30)
    today = date.today()
    if today.year == season and today < end:
        end = today - timedelta(days=1)

    if start > end:
        # Season hasn't started yet
        if not existing_df.empty:
            _raw_cache[key] = existing_df
            return existing_df
        return pd.DataFrame()

    # Get schedule
    schedule = _fetch_schedule(sport_id, start.isoformat(), end.isoformat())
    all_game_pks = {g["game_pk"] for g in schedule}
    new_game_pks = all_game_pks - existing_game_pks

    if not new_game_pks:
        log.info("No new games to fetch for %s %d", level, season)
        if not existing_df.empty:
            _raw_cache[key] = existing_df
            return existing_df
        return pd.DataFrame()

    log.info("Fetching %d new game feeds for %s %d (%d already cached)",
             len(new_game_pks), level, season, len(existing_game_pks))

    # Fetch game feeds in parallel
    new_pitches: list[dict] = []
    fetched = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_game_feed, gpk): gpk
            for gpk in new_game_pks
        }
        for future in as_completed(futures):
            gpk = futures[future]
            try:
                feed = future.result()
                if feed:
                    rows = _extract_pitches_from_feed(feed)
                    new_pitches.extend(rows)
                    fetched += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
                log.warning("Error processing game %d", gpk, exc_info=True)

            if (fetched + failed) % 50 == 0:
                log.info("Progress: %d/%d games fetched (%d failed)",
                         fetched, fetched + failed, failed)

    log.info("Fetched %d games (%d pitches), %d failed",
             fetched, len(new_pitches), failed)

    # Combine with existing data
    if new_pitches:
        new_df = pd.DataFrame(new_pitches)
        if not existing_df.empty:
            result = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            result = new_df

        # Save updated parquet
        try:
            result.to_parquet(parquet_file, index=False)
            log.info("Saved %d total pitches to %s", len(result), parquet_file)
        except Exception:
            log.warning("Failed to save parquet cache", exc_info=True)
    else:
        result = existing_df

    if result.empty:
        return pd.DataFrame()

    _raw_cache[key] = result
    return result


# ── Fetch data for a single pitcher ──────────────────────────────────

def fetch_milb_pitcher(pitcher_id: int, level: str = "AAA",
                        season: int = MLB_SEASON) -> pd.DataFrame | None:
    """Fetch Statcast data for a single MiLB pitcher."""
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
    """Aggregate raw pitch data into per-pitcher, per-pitch-type stats."""
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    pid_col = "pitcher" if "pitcher" in df.columns else "pitcher_id"
    name_col = "player_name" if "player_name" in df.columns else None

    # Filter noise
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(_NOISE)]
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
    elif "zone" in df.columns:
        df["_in_zone"] = df["zone"].between(1, 9)
    else:
        df["_in_zone"] = True

    df["_is_chase"] = df["_is_swing"] & ~df["_in_zone"]

    # Movement: pfx_x/pfx_z from game feed are already in inches
    if "pfx_x" in df.columns and "hb" not in df.columns:
        df["hb"] = df["pfx_x"]
        df["ivb"] = df["pfx_z"]
    elif "hb" not in df.columns:
        df["hb"] = np.nan
        df["ivb"] = np.nan

    # Total pitches per pitcher
    total_per_pitcher = df.groupby(pid_col).size().rename("_total")

    # Per-pitch batted ball flags
    if "launch_speed" in df.columns:
        df["_pt_has_bb"] = df["launch_speed"].notna()
        df["_pt_hard_hit"] = df["launch_speed"] >= 95
    else:
        df["_pt_has_bb"] = False
        df["_pt_hard_hit"] = False

    pt_col = "pitch_type"
    grouped = df.groupby([pid_col, pt_col])

    spin_col = ("release_spin_rate" if "release_spin_rate" in df.columns
                else "spin_rate" if "spin_rate" in df.columns
                else None)

    agg_dict = {
        "count": (pt_col, "size"),
        "velocity": ("release_speed", "mean"),
        "hb": ("hb", "mean"),
        "ivb": ("ivb", "mean"),
        "release_extension": ("release_extension", "mean"),
        "swings": ("_is_swing", "sum"),
        "whiffs": ("_is_whiff", "sum"),
        "called_strikes": ("_is_called_strike", "sum"),
        "in_zone": ("_in_zone", "sum"),
        "chases": ("_is_chase", "sum"),
        "out_of_zone": ("_in_zone", lambda x: (~x).sum()),
        "batted_balls": ("_pt_has_bb", "sum"),
        "hard_hits": ("_pt_hard_hit", "sum"),
    }
    if spin_col:
        agg_dict["spin_rate"] = (spin_col, "mean")

    agg = grouped.agg(**agg_dict).reset_index()

    # Per-pitch-type exit velo
    if "launch_speed" in df.columns:
        ev_pt = df[df["launch_speed"].notna()].groupby([pid_col, pt_col]).agg(
            avg_exit_velo=("launch_speed", "mean"),
        ).reset_index()
        agg = agg.merge(ev_pt, on=[pid_col, pt_col], how="left")
    else:
        agg["avg_exit_velo"] = np.nan

    # xBA (not available from game feed, but keep column for compatibility)
    agg["xba"] = np.nan

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
    if "pitcher_team" in df.columns:
        team_map = df.groupby(pid_col)["pitcher_team"].first()
        agg["team"] = agg[pid_col].map(team_map)
    elif "home_team" in df.columns:
        team_map = df.groupby(pid_col)["home_team"].first()
        agg["team"] = agg[pid_col].map(team_map)

    # Join total pitches
    agg = agg.merge(total_per_pitcher.reset_index(), on=pid_col)

    # Compute rates
    agg["percentage_thrown"] = agg["count"] / agg["_total"]
    agg["whiff_rate"] = np.where(agg["swings"] > 0,
                                  agg["whiffs"] / agg["swings"], 0)
    agg["chase_percentage"] = np.where(agg["out_of_zone"] > 0,
                                        agg["chases"] / agg["out_of_zone"], 0)
    agg["csw"] = np.where(
        agg["count"] > 0,
        (agg["called_strikes"] + agg["whiffs"]) / agg["count"], 0
    )
    agg["zone_rate"] = np.where(
        agg["count"] > 0, agg["in_zone"] / agg["count"], 0
    )
    agg["swing_rate"] = np.where(
        agg["count"] > 0, agg["swings"] / agg["count"], 0
    )
    agg["hard_hit_rate"] = np.where(
        agg["batted_balls"] > 0, agg["hard_hits"] / agg["batted_balls"], 0
    )

    # Rename for compatibility
    agg = agg.rename(columns={pid_col: "player_id"})

    # Clean up
    agg = agg.drop(columns=["_total", "swings", "whiffs", "called_strikes",
                             "in_zone", "chases", "out_of_zone",
                             "batted_balls", "hard_hits"], errors="ignore")

    return agg


# ── Aggregation: season-level pitcher stats ───────────────────────────

def aggregate_pitcher_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw pitch data into per-pitcher season stats."""
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
    elif "zone" in df.columns:
        df["_in_zone"] = df["zone"].between(1, 9)
    else:
        df["_in_zone"] = True

    df["_is_chase"] = df["_is_swing"] & ~df["_in_zone"]

    # First-pitch strike detection
    if all(c in df.columns for c in ("balls", "strikes")):
        df["_is_first_pitch"] = (df["balls"] == 0) & (df["strikes"] == 0)
        df["_is_first_pitch_strike"] = df["_is_first_pitch"] & (
            df["_is_called_strike"] | df["_is_swing"]
        )
    else:
        df["_is_first_pitch"] = False
        df["_is_first_pitch_strike"] = False

    # Batted ball stats
    if "launch_speed" in df.columns:
        df["_has_batted_ball"] = df["launch_speed"].notna()
        df["_is_hard_hit"] = df["launch_speed"] >= 95
        df["_is_barrel"] = (
            df["launch_speed"].ge(98) &
            df["launch_angle"].between(26, 30)
        ) if "launch_angle" in df.columns else False
    else:
        df["_has_batted_ball"] = False
        df["_is_hard_hit"] = False
        df["_is_barrel"] = False

    # Ground ball detection
    if "launch_angle" in df.columns:
        df["_is_ground_ball"] = (
            df["launch_angle"].notna() & (df["launch_angle"] < 10)
        )
    else:
        df["_is_ground_ball"] = False

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

    ab_col = "at_bat_number" if "at_bat_number" in df.columns else "event_index"
    game_col = "game_pk" if "game_pk" in df.columns else "game_id"

    grouped = df.groupby(pid_col)

    agg = grouped.agg(
        total_pitches=("pitch_type", "size"),
        swings=("_is_swing", "sum"),
        whiffs=("_is_whiff", "sum"),
        called_strikes=("_is_called_strike", "sum"),
        chases=("_is_chase", "sum"),
        in_zone=("_in_zone", "sum"),
        out_of_zone=("_in_zone", lambda x: (~x).sum()),
        strikeouts=("_is_strikeout", "sum"),
        walks=("_is_walk", "sum"),
        first_pitches=("_is_first_pitch", "sum"),
        first_pitch_strikes=("_is_first_pitch_strike", "sum"),
        batted_balls=("_has_batted_ball", "sum"),
        hard_hits=("_is_hard_hit", "sum"),
        barrels=("_is_barrel", "sum"),
        ground_balls=("_is_ground_ball", "sum"),
    ).reset_index()

    # Exit velo
    if "launch_speed" in df.columns:
        ev_agg = df[df["launch_speed"].notna()].groupby(pid_col).agg(
            avg_exit_velo=("launch_speed", "mean"),
        ).reset_index()
        agg = agg.merge(ev_agg, on=pid_col, how="left")
    else:
        agg["avg_exit_velo"] = np.nan

    # Launch angle
    if "launch_angle" in df.columns:
        la_agg = df[df["launch_angle"].notna()].groupby(pid_col).agg(
            avg_launch_angle=("launch_angle", "mean"),
        ).reset_index()
        agg = agg.merge(la_agg, on=pid_col, how="left")
    else:
        agg["avg_launch_angle"] = np.nan

    # xBA/xwOBA not available from game feed
    agg["xba_against"] = np.nan
    agg["xwoba_against"] = np.nan

    # Batters faced (unique game + at_bat combos)
    if game_col in df.columns and ab_col in df.columns:
        bf = df.groupby(pid_col).apply(
            lambda x: x[[game_col, ab_col]].drop_duplicates().shape[0],
            include_groups=False,
        ).rename("batters_faced")
        agg = agg.merge(bf.reset_index(), on=pid_col)
    else:
        agg["batters_faced"] = agg["total_pitches"] / 4

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

    if "pitcher_team" in df.columns:
        agg["team"] = agg[pid_col].map(df.groupby(pid_col)["pitcher_team"].first())
    elif "home_team" in df.columns:
        agg["team"] = agg[pid_col].map(df.groupby(pid_col)["home_team"].first())

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
    agg["zone_percentage"] = np.where(
        agg["total_pitches"] > 0, agg["in_zone"] / agg["total_pitches"], 0
    )
    agg["first_pitch_strike_percentage"] = np.where(
        agg["first_pitches"] > 0,
        agg["first_pitch_strikes"] / agg["first_pitches"], 0
    )
    agg["hard_hit_percentage"] = np.where(
        agg["batted_balls"] > 0, agg["hard_hits"] / agg["batted_balls"], 0
    )
    agg["barrel_percentage"] = np.where(
        agg["batted_balls"] > 0, agg["barrels"] / agg["batted_balls"], 0
    )
    agg["ground_ball_percentage"] = np.where(
        agg["batted_balls"] > 0, agg["ground_balls"] / agg["batted_balls"], 0
    )
    agg["k_minus_bb"] = agg["strike_out_percentage"] - agg["walk_percentage"]

    # Rename & filter
    agg = agg.rename(columns={pid_col: "player_id"})
    agg = agg[agg["total_pitches"] >= 100]  # minimum threshold

    # Clean up internal columns
    agg = agg.drop(columns=["swings", "whiffs", "called_strikes", "chases",
                             "in_zone", "out_of_zone", "strikeouts", "walks",
                             "first_pitches", "first_pitch_strikes",
                             "batted_balls", "hard_hits", "barrels",
                             "ground_balls"],
                    errors="ignore")

    return agg


# ── Public API ────────────────────────────────────────────────────────

def get_milb_season_pitchers(level: str = "AAA",
                              season: int = MLB_SEASON) -> pd.DataFrame:
    """Get season-level MiLB pitcher stats."""
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
    """Get pitch-type level MiLB stats."""
    key = f"{level}_{season}"
    if key in _season_pitches_cache:
        return _season_pitches_cache[key]

    raw = fetch_milb_season(level=level, season=season)
    if raw.empty:
        return pd.DataFrame()

    result = aggregate_pitch_stats(raw)
    _season_pitches_cache[key] = result
    return result


# ── Player picking ────────────────────────────────────────────────────

_mlb_ids_cache: set[int] | None = None


def _get_mlb_player_ids(season: int = MLB_SEASON) -> set[int]:
    """Get MLB player IDs from Pitch Profiler to exclude from MiLB picks."""
    try:
        from . import pitch_profiler
        mlb_df = pitch_profiler.get_season_pitchers()
        if mlb_df.empty:
            return set()
        pid_col = "pitcher_id" if "pitcher_id" in mlb_df.columns else "player_id"
        if pid_col in mlb_df.columns:
            return set(mlb_df[pid_col].dropna().astype(int).tolist())
    except Exception:
        log.warning("Could not load MLB player IDs for filtering", exc_info=True)
    return set()


def pick_milb_player(level: str = "AAA",
                     season: int = MLB_SEASON,
                     min_pitches: int = 400,
                     min_batters_faced: int = 100) -> dict | None:
    """Pick a random MiLB pitcher with enough data for a card.

    Filters out MLB players by cross-referencing against the MLB
    Pitch Profiler dataset.
    """
    global _mlb_ids_cache

    pitchers = get_milb_season_pitchers(level=level, season=season)
    if pitchers.empty:
        return None

    # Load MLB player IDs to exclude
    if _mlb_ids_cache is None:
        _mlb_ids_cache = _get_mlb_player_ids(season)
        log.info("Loaded %d MLB player IDs for MiLB filtering", len(_mlb_ids_cache))

    # Filter: enough pitches/BF AND not in MLB dataset
    qualified = pitchers[
        (pitchers["total_pitches"] >= min_pitches) &
        (pitchers["batters_faced"] >= min_batters_faced)
    ]

    if _mlb_ids_cache:
        qualified = qualified[~qualified["player_id"].isin(_mlb_ids_cache)]

    if qualified.empty:
        log.warning("No qualified MiLB-only pitchers at %s after filtering", level)
        # Relax pitch threshold but keep MLB filter
        qualified = pitchers[pitchers["total_pitches"] >= 100]
        if _mlb_ids_cache:
            qualified = qualified[~qualified["player_id"].isin(_mlb_ids_cache)]

    if qualified.empty:
        log.warning("No MiLB pitchers found at %s after all filters", level)
        return None

    row = qualified.sample(1).iloc[0]
    pid = int(row["player_id"])
    info = _fetch_player_info(pid)

    return {
        "name": row.get("pitcher_name", info["name"]),
        "id": pid,
        "team": row.get("team", info.get("team", "")),
        "level": level,
    }
