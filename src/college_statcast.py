"""Amateur (college / draft-showcase) Statcast fetcher and prospect picker.

Covers everything under the MLB Stats API's amateur sport (``sportId=22``):
NCAA D1 ("College Baseball") in spring, and the Hawk-Eye-tracked summer
draft showcases — Cape Cod, MLB Draft League, Appalachian — in June–August.

Statcast-tracked amateur games expose full Hawk-Eye tracking through the
same Baseball Savant gameday feed used for MLB/MiLB — but under a
*different* JSON shape. The ``gf?game_pk=`` endpoint returns pitches
nested inside ``home_pitchers`` / ``away_pitchers`` dicts (keyed by
pitcher id), each pitch carrying velocity, spin, movement (pfxX/pfxZ in
feet), plate location, zone, count and the play result.

We flatten that into the *same* raw pitch schema ``milb_statcast`` uses,
then reuse its aggregators (``aggregate_pitch_stats`` /
``aggregate_pitcher_stats``) so the existing prospect pitcher card renders
unchanged.

College Savant has **no video** (``sporty-videos`` returns "No Video" for
amateur plays), so ``find_player_video`` scrapes YouTube search results to
surface a highlight clip for the post.

Only a subset of college games are tracked (postseason / neutral-site
parks with Hawk-Eye). ``fetch_college_window`` scans a date range, keeps
only games whose feed actually yields tracked pitches, and disk-caches
each game (feeds are immutable once Final).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import CLIPS_DIR, DATA_DIR, MLB_SEASON
from .milb_statcast import (
    _CALL_MAP,
    aggregate_pitch_stats,
    aggregate_pitcher_stats,
)

log = logging.getLogger(__name__)

SPORT_ID = 22  # "amateur/collegiate" sport on the MLB Stats API — a
# catch-all that resolves to different leagues by season: NCAA D1
# ("College Baseball") in spring, and the Hawk-Eye-tracked summer draft
# showcases (Cape Cod, MLB Draft League, Appalachian) in June–August.

_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
_GF_URL = "https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

# league name → short badge shown on the card, in draft-showcase priority
# order (most-scouted first). Used both for labelling and target selection.
LEAGUE_LABELS: dict[str, str] = {
    "Cape Cod Baseball League": "CAPE COD",
    "MLB Draft League": "DRAFT LG",
    "College Baseball": "NCAA",
    "Appalachian League": "APPY LG",
    "Northwoods League": "NORTHWDS",
    "Prospect League": "PROSPECT",
}
# preference order when several leagues are tracked in the same window
LEAGUE_PRIORITY = list(LEAGUE_LABELS.keys())

# Cache dirs
_CACHE_DIR = DATA_DIR / "college_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_GAME_CACHE_DIR = _CACHE_DIR / "games"
_GAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory caches
_window_cache: dict[str, pd.DataFrame] = {}
_pitchers_cache: dict[str, pd.DataFrame] = {}
_pitches_cache: dict[str, pd.DataFrame] = {}

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0"})


# ── Helpers ──────────────────────────────────────────────────────────

def _f(v) -> float:
    """Coerce a value (possibly a string like '77.0' or None) to float/NaN."""
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── Schedule ─────────────────────────────────────────────────────────

def _fetch_schedule(start_date: str, end_date: str,
                    leagues: set[str] | None = None) -> list[dict]:
    """Return Final amateur games in a date range, tagged with league.

    ``leagues`` optionally restricts to a set of league names (e.g.
    ``{"Cape Cod Baseball League"}``).
    """
    games: list[dict] = []
    try:
        resp = _session.get(
            f"{_STATS_API_BASE}/schedule",
            params={
                "sportId": SPORT_ID,
                "startDate": start_date,
                "endDate": end_date,
                "hydrate": "team,league",
            },
            timeout=30,
        )
        resp.raise_for_status()
        for dt in resp.json().get("dates", []):
            for g in dt.get("games", []):
                if g.get("status", {}).get("detailedState", "") != "Final":
                    continue
                home = g["teams"]["home"]["team"]
                league = home.get("league", {}).get("name", "") or ""
                if leagues and league not in leagues:
                    continue
                games.append({
                    "game_pk": g["gamePk"],
                    "date": dt["date"],
                    "home_team": home.get("name", ""),
                    "away_team": g["teams"]["away"]["team"].get("name", ""),
                    "league": league,
                })
    except Exception:
        log.warning("College schedule fetch failed %s..%s",
                    start_date, end_date, exc_info=True)
    log.info("Found %d Final amateur games %s..%s",
             len(games), start_date, end_date)
    return games


# ── Feed flattening ──────────────────────────────────────────────────

def _pitches_from_gf(gf: dict) -> list[dict]:
    """Flatten a Savant college gameday feed into raw pitch rows.

    Maps the gf pitch shape onto the ``milb_statcast`` raw schema so the
    shared aggregators work unchanged. ``pfxX``/``pfxZ`` are in feet in the
    feed → multiplied by 12 to inches (signed) to match MiLB's hb/ivb.
    """
    game_pk = gf.get("game_pk") or gf.get("scoreboard", {}).get("game_pk")
    game_date = (gf.get("gameDate") or "")[:10]

    home = gf.get("home_team_data", {}) or {}
    away = gf.get("away_team_data", {}) or {}
    home_abbr = home.get("abbreviation") or home.get("triCode") or ""
    away_abbr = away.get("abbreviation") or away.get("triCode") or ""
    home_name = home.get("name", "")
    away_name = away.get("name", "")

    rows: list[dict] = []
    # home_pitchers pitch against the away team, and vice-versa
    for side, team_name in (("home_pitchers", home_name),
                            ("away_pitchers", away_name)):
        for pid, pitches in (gf.get(side) or {}).items():
            for p in pitches or []:
                speed = _f(p.get("start_speed"))
                if not speed or np.isnan(speed):
                    continue  # untracked pitch — skip
                raw_desc = p.get("description", "")
                desc = _CALL_MAP.get(
                    raw_desc, str(raw_desc).lower().replace(" ", "_"))
                pfx_x = _f(p.get("pfxX"))
                pfx_z = _f(p.get("pfxZ"))
                rows.append({
                    "game_pk": _int(p.get("game_pk")) or _int(game_pk),
                    "game_date": game_date,
                    "pitcher": _int(p.get("pitcher")) or _int(pid),
                    "player_name": p.get("pitcher_name", "Unknown"),
                    "p_throws": p.get("p_throws", "R"),
                    "batter": _int(p.get("batter")),
                    "batter_name": p.get("batter_name", ""),
                    "at_bat_number": p.get("ab_number"),
                    "pitch_number": p.get("pitch_number"),
                    "pitch_type": p.get("pitch_type", "") or "",
                    "pitch_name": p.get("pitch_name", ""),
                    "description": desc,
                    "events": p.get("events") or "",
                    "release_speed": speed,
                    "release_spin_rate": _f(p.get("spin_rate")),
                    "pfx_x": pfx_x * 12 if not np.isnan(pfx_x) else np.nan,
                    "pfx_z": pfx_z * 12 if not np.isnan(pfx_z) else np.nan,
                    "release_extension": _f(p.get("extension")),
                    "plate_x": _f(p.get("plate_x", p.get("px"))),
                    "plate_z": _f(p.get("plate_z", p.get("pz"))),
                    "release_pos_x": _f(p.get("x0")),   # release side (ft)
                    "release_pos_z": _f(p.get("z0")),   # release height (ft)
                    "sz_top": _f(p.get("sz_top")),
                    "sz_bot": _f(p.get("sz_bot")),
                    "zone": p.get("zone"),
                    "launch_speed": _f(p.get("hit_speed")),
                    "launch_angle": _f(p.get("hit_angle")),
                    "hit_distance_sc": _f(p.get("hit_distance")),
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "pitcher_team": p.get("team_fielding", "") or "",
                    "pitcher_team_name": team_name,
                    "balls": p.get("pre_balls", 0),
                    "strikes": p.get("pre_strikes", 0),
                    "play_id": p.get("play_id"),
                })

    if not rows:
        return rows

    # Keep the PA result (``events``) only on the terminal pitch of each
    # at-bat, matching MiLB semantics — the feed stamps it on every pitch
    # of the PA, which would otherwise over-count Ks/BBs.
    df = pd.DataFrame(rows)
    if {"at_bat_number", "pitch_number"}.issubset(df.columns):
        last_idx = df.groupby("at_bat_number")["pitch_number"].transform("max")
        df.loc[df["pitch_number"] != last_idx, "events"] = ""
    return df.to_dict("records")


# ── Per-game fetch (disk-cached) ─────────────────────────────────────

def fetch_college_game(game_pk: int) -> list[dict]:
    """Fetch + flatten one college game's tracked pitches (disk-cached).

    Returns a (possibly empty) list of raw pitch dicts. Empty means the
    game had no Hawk-Eye tracking — still cached so we don't refetch.
    """
    cache_file = _GAME_CACHE_DIR / f"{game_pk}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    rows: list[dict] = []
    for attempt in range(2):
        try:
            resp = _session.get(_GF_URL.format(game_pk=game_pk), timeout=15)
            resp.raise_for_status()
            rows = _pitches_from_gf(resp.json())
            break
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
            else:
                log.warning("Failed to fetch college gf %s: %s", game_pk, e)

    try:
        cache_file.write_text(json.dumps(rows, separators=(",", ":")),
                              encoding="utf-8")
    except Exception:
        pass
    return rows


# ── Window fetch ─────────────────────────────────────────────────────

def fetch_college_window(start_date: str, end_date: str,
                         leagues: set[str] | None = None) -> pd.DataFrame:
    """Fetch all tracked amateur pitches across a date range.

    Each row is tagged with its ``league``. ``leagues`` optionally filters
    the schedule to specific leagues before fetching.
    """
    key = f"{start_date}_{end_date}_{'|'.join(sorted(leagues)) if leagues else 'all'}"
    if key in _window_cache:
        return _window_cache[key]

    schedule = _fetch_schedule(start_date, end_date, leagues=leagues)
    if not schedule:
        return pd.DataFrame()

    pk_league = {g["game_pk"]: g.get("league", "") for g in schedule}

    all_rows: list[dict] = []
    tracked = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_college_game, g["game_pk"]): g
            for g in schedule
        }
        for future in as_completed(futures):
            try:
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
                    tracked += 1
            except Exception:
                log.warning("College game fetch errored", exc_info=True)

    log.info("Amateur window %s..%s: %d/%d games tracked, %d pitches",
             start_date, end_date, tracked, len(schedule), len(all_rows))

    if not all_rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)
        df["league"] = df["game_pk"].map(pk_league).fillna("")
    _window_cache[key] = df
    return df


# ── Public aggregation API ───────────────────────────────────────────

def _league_key(leagues: set[str] | None) -> str:
    return "|".join(sorted(leagues)) if leagues else "all"


def get_college_pitchers(start_date: str, end_date: str,
                         min_pitches: int = 40,
                         leagues: set[str] | None = None) -> pd.DataFrame:
    """Season-ish per-pitcher stats over an amateur date window.

    ``leagues`` restricts the pool (and hence percentile baseline) to
    specific leagues — pass a single league for fair same-league
    percentiles on the card.
    """
    key = f"{start_date}_{end_date}_{min_pitches}_{_league_key(leagues)}"
    if key in _pitchers_cache:
        return _pitchers_cache[key]
    # Fetch only the target league's games when a single league is requested
    # (avoids scraping every amateur league across the window).
    raw = fetch_college_window(start_date, end_date, leagues=leagues)
    if raw.empty:
        return pd.DataFrame()
    if leagues and "league" in raw.columns:
        raw = raw[raw["league"].isin(leagues)]
        if raw.empty:
            return pd.DataFrame()
    result = aggregate_pitcher_stats(raw, min_pitches=min_pitches)
    # carry full team name + league through for the card / video query
    if not result.empty:
        if "pitcher_team_name" in raw.columns:
            tn = raw.groupby("pitcher")["pitcher_team_name"].first()
            result["team_name"] = result["player_id"].map(tn)
        if "league" in raw.columns:
            lg = raw.groupby("pitcher")["league"].first()
            result["league"] = result["player_id"].map(lg)
    _pitchers_cache[key] = result
    return result


def get_college_pitches(start_date: str, end_date: str,
                        leagues: set[str] | None = None) -> pd.DataFrame:
    """Per-pitch-type stats over an amateur date window."""
    key = f"{start_date}_{end_date}_{_league_key(leagues)}"
    if key in _pitches_cache:
        return _pitches_cache[key]
    raw = fetch_college_window(start_date, end_date, leagues=leagues)
    if raw.empty:
        return pd.DataFrame()
    if leagues and "league" in raw.columns:
        raw = raw[raw["league"].isin(leagues)]
        if raw.empty:
            return pd.DataFrame()
    result = aggregate_pitch_stats(raw)
    _pitches_cache[key] = result
    return result


# ── Prospect picking ─────────────────────────────────────────────────

NCAA_LEAGUE = "College Baseball"  # MLB Stats API league name for NCAA D1

# Curated 2026 MLB Draft college arms, pulled from ESPN's pre-draft Top 150
# board (Kiley McDaniel), updated 2026-07-07 ahead of the July 11–12 draft.
# ``rank`` is the OVERALL board rank; names are matched case-insensitively
# against tracked Statcast pitcher names so the highest-ranked arm with
# tracked college data gets featured (with a rank chip). ``fv`` is optional
# (ESPN's board isn't graded on the 20–80 FV scale) — add grades later if a
# graded source is handy; the card omits the FV bit of the chip when absent.
DRAFT_RANKINGS_SOURCE = "ESPN"
DRAFT_PROSPECTS: list[dict] = [
    {"rank": 4,   "name": "Jackson Flora",     "school": "UC Santa Barbara",  "hand": "R"},
    {"rank": 9,   "name": "Cameron Flukey",    "school": "Coastal Carolina",  "hand": "R"},
    {"rank": 13,  "name": "Liam Peterson",     "school": "Florida",           "hand": "R"},
    {"rank": 15,  "name": "Cole Carlon",       "school": "Arizona State",     "hand": "L"},
    {"rank": 16,  "name": "Cade Townsend",     "school": "Ole Miss",          "hand": "R"},
    {"rank": 18,  "name": "Tegan Kuhns",       "school": "Tennessee",         "hand": "R"},
    {"rank": 20,  "name": "Mason Edwards",     "school": "USC",               "hand": "L"},
    {"rank": 23,  "name": "Logan Reddemann",   "school": "UCLA",              "hand": "R"},
    {"rank": 24,  "name": "Hunter Dietz",      "school": "Arkansas",          "hand": "L"},
    {"rank": 42,  "name": "Ben Blair",         "school": "Liberty",           "hand": "R"},
    {"rank": 46,  "name": "Ethan Kleinschmit", "school": "Oregon State",      "hand": "L"},
    {"rank": 49,  "name": "Jack Radel",        "school": "Notre Dame",        "hand": "R"},
    {"rank": 50,  "name": "Wes Mendes",        "school": "Florida State",     "hand": "L"},
    {"rank": 60,  "name": "Ethan Norby",       "school": "East Carolina",     "hand": "L"},
    {"rank": 65,  "name": "Ruger Riojas",      "school": "Texas",             "hand": "R"},
    {"rank": 67,  "name": "Joey Volchko",      "school": "Georgia",           "hand": "R"},
    {"rank": 76,  "name": "Brett Renfrow",     "school": "Virginia Tech",     "hand": "R"},
    {"rank": 78,  "name": "Jacob Dudan",       "school": "NC State",          "hand": "R"},
    {"rank": 80,  "name": "Ryan Peterson",     "school": "Sam Houston State", "hand": "R"},
    {"rank": 85,  "name": "Ryan Marohn",       "school": "NC State",          "hand": "L"},
    {"rank": 86,  "name": "Deven Sheerin",     "school": "LSU",               "hand": "R"},
    {"rank": 107, "name": "Ryan Lynch",        "school": "North Carolina",    "hand": "R"},
    {"rank": 120, "name": "Jason Decaro",      "school": "North Carolina",    "hand": "R"},
    {"rank": 126, "name": "Trey Beard",        "school": "Florida State",     "hand": "L"},
    {"rank": 129, "name": "Aidan Knaak",       "school": "Clemson",           "hand": "R"},
    {"rank": 139, "name": "Luke McNeillie",    "school": "Florida",           "hand": "R"},
    {"rank": 140, "name": "Tommy LaPour",      "school": "TCU",               "hand": "R"},
    {"rank": 144, "name": "Maxx Yehl",         "school": "West Virginia",     "hand": "L"},
    {"rank": 145, "name": "Carson Wiggins",    "school": "Arkansas",          "hand": "R"},
    {"rank": 147, "name": "Duncan Marsten",    "school": "Wake Forest",       "hand": "R"},
    {"rank": 151, "name": "Justin LeGuernic",  "school": "Clemson",           "hand": "L"},
    {"rank": 152, "name": "Taylor Rabe",       "school": "Ole Miss",          "hand": "R"},
]


def _norm_name(n: str) -> str:
    """Normalise a name for matching ('Last, First' → 'first last')."""
    n = str(n or "").strip()
    if "," in n:
        parts = n.split(", ")
        if len(parts) == 2:
            n = f"{parts[1]} {parts[0]}"
    return " ".join(n.lower().split())


_PROSPECT_BY_NAME = {_norm_name(p["name"]): p for p in DRAFT_PROSPECTS}


def match_prospect(name: str) -> dict | None:
    """Return the curated ranking entry for a pitcher name, or None."""
    return _PROSPECT_BY_NAME.get(_norm_name(name))


def college_season_window(season: int = MLB_SEASON) -> tuple[str, str]:
    """Full NCAA D1 season window (Feb → Jun) — the tracked College
    Baseball games in the MLB Stats API are marquee/postseason events
    spread across the whole spring, so draft-prospect content pulls from
    the entire season rather than a trailing window."""
    return (f"{season}-02-01", f"{season}-06-30")


def default_window(days: int = 14) -> tuple[str, str]:
    """A recent trailing window ending today.

    Draft-showcase content is best when fresh, and the tracked leagues
    change with the calendar (NCAA in spring, Cape Cod / Draft League /
    Appalachian in summer). A trailing window always lands on whatever is
    currently being played and tracked.
    """
    today = date.today()
    start = date.fromordinal(today.toordinal() - days)
    return (start.isoformat(), today.isoformat())


def pick_target_league(start_date: str, end_date: str,
                       min_pitches: int = 40,
                       min_pitchers: int = 8) -> str | None:
    """Choose which league to feature from a window.

    Prefers the most-scouted showcase (Cape Cod > Draft League > NCAA >
    Appalachian …) that has a usable percentile pool (``>= min_pitchers``
    qualified arms). Falls back to the league with the most qualified arms.
    """
    pitchers = get_college_pitchers(start_date, end_date, min_pitches)
    if pitchers.empty or "league" not in pitchers.columns:
        return None
    counts = pitchers["league"].value_counts().to_dict()
    for lg in LEAGUE_PRIORITY:
        if counts.get(lg, 0) >= min_pitchers:
            return lg
    # nothing hit the priority list with a full pool — take the biggest
    return max(counts, key=counts.get) if counts else None


def pick_college_prospect(start_date: str | None = None,
                          end_date: str | None = None,
                          min_pitches: int = 40,
                          min_batters_faced: int = 15,
                          top_n: int = 20,
                          leagues: set[str] | None = None,
                          exclude_ids: set[int] | None = None,
                          rng=None) -> dict | None:
    """Pick a standout amateur pitching prospect from a date window.

    Ranks qualified arms by a simple "stuff" score (whiff% + K%) and
    samples one from the top ``top_n`` so the card rotates day to day.
    ``leagues`` restricts the pool (and percentile baseline); the
    batters-faced floor screens out small-sample flukes.
    """
    import random

    if not start_date or not end_date:
        start_date, end_date = default_window()

    pitchers = get_college_pitchers(start_date, end_date, min_pitches,
                                    leagues=leagues)
    if pitchers.empty:
        return None

    df = pitchers.copy()
    if "batters_faced" in df.columns:
        bf = pd.to_numeric(df["batters_faced"], errors="coerce").fillna(0)
        filtered = df[bf >= min_batters_faced]
        if not filtered.empty:
            df = filtered
    if exclude_ids:
        df = df[~df["player_id"].isin(exclude_ids)]
    if df.empty:
        return None

    # Prefer a curated-ranking prospect who actually has tracked data —
    # feature the highest-ranked such arm. Otherwise fall back to a
    # "stuff"-scored pick (whiff% + K%), sampled from the top ``top_n``.
    df = df.copy()
    df["_norm"] = df["pitcher_name"].map(_norm_name)
    ranked = df[df["_norm"].isin(_PROSPECT_BY_NAME.keys())]

    if not ranked.empty:
        ranked = ranked.copy()
        ranked["_rank"] = ranked["_norm"].map(
            lambda x: _PROSPECT_BY_NAME[x]["rank"])
        row = ranked.sort_values("_rank").iloc[0].to_dict()
    else:
        whiff = pd.to_numeric(df["whiff_rate"], errors="coerce").fillna(0)
        k_pct = pd.to_numeric(df["strike_out_percentage"],
                              errors="coerce").fillna(0)
        df["_score"] = whiff * 100 + k_pct * 100
        top = df.sort_values("_score", ascending=False).head(top_n)
        if top.empty:
            return None
        rng = rng or random
        weights = (top["_score"] - top["_score"].min() + 1.0).tolist()
        row = rng.choices(top.to_dict("records"), weights=weights, k=1)[0]

    league = row.get("league", "") or ""
    meta = match_prospect(row.get("pitcher_name", ""))
    return {
        "name": row.get("pitcher_name", "Unknown"),
        "id": _int(row.get("player_id")),
        "team": row.get("team", ""),
        "team_name": row.get("team_name", "") or row.get("team", ""),
        "league": league,
        "league_label": LEAGUE_LABELS.get(league, "COLLEGE"),
        "rank": meta.get("rank") if meta else None,
        "fv": meta.get("fv") if meta else None,
        "ranked_source": DRAFT_RANKINGS_SOURCE if meta else None,
        "start_date": start_date,
        "end_date": end_date,
    }


# ── Video search (YouTube) ───────────────────────────────────────────

def find_player_video(name: str, team_name: str = "",
                      extra: str = "baseball") -> tuple[str, str] | None:
    """Find a YouTube highlight for an amateur player.

    Scrapes YouTube search results (no API key). Returns ``(url, kind)``
    where ``kind`` is ``"player"`` (the clip names the player) or ``"team"``
    (a clip of the player's team — relevant but not the player themselves),
    or ``None`` if nothing can be confidently tied to the player. The caller
    words the post honestly based on ``kind``.
    """
    if not name or name == "Unknown":
        return None
    query = " ".join(x for x in (name, team_name, extra) if x).strip()
    url = "https://www.youtube.com/results?search_query=" + quote(query)
    try:
        html = _session.get(
            url, timeout=20, headers={"Accept-Language": "en-US"}).text
    except Exception:
        log.warning("YouTube search failed for %s", name, exc_info=True)
        return None

    m = re.search(r'var ytInitialData = (\{.*?\});</script>', html)
    if not m:
        m = re.search(r'ytInitialData"?\]?\s*=\s*(\{.*?\});', html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None

    vids: list[tuple[str, str]] = []

    def walk(o):
        if isinstance(o, dict):
            vr = o.get("videoRenderer")
            if vr and vr.get("videoId"):
                title = "".join(
                    r.get("text", "")
                    for r in vr.get("title", {}).get("runs", []))
                vids.append((vr["videoId"], title))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    if not vids:
        return None

    parts = name.lower().split()
    first, surname = parts[0], parts[-1]
    # On-field / on-mound action words — a returned clip MUST contain one, so
    # what posts is the player (or team) PLAYING, never talking.
    action_words = ("highlight", "highlights", "pitching", "pitches",
                    "strikeout", "strikeouts", "k's", "scoreless", "innings",
                    "gem", "dominant", "dominates", "dealing", "outing",
                    "complete game", "shutout", "no-hitter", "no hitter",
                    "punchout", "punch out", "punches out", " vs ", " vs.",
                    "vs ", "fastball", "slider", "on the mound", "start",
                    "bullpen session", "showcase")
    # Interview / talk / off-field content — reject outright, even on a name
    # match (this is what keeps interviews and podcasts out of the post).
    interview_words = ("interview", "podcast", "press conference", "presser",
                       "sits down", "sit down", "one-on-one", "one on one",
                       "q&a", "q & a", "reaction", "talks", "korner",
                       "media day", "availability", "commit", "commitment",
                       "committed", "signing", "signs with", "draft day",
                       "preview", "previews", "get ready", "gets ready",
                       "catches up", "day in the life", "vlog", "mic'd",
                       "recruiting", "announces")
    # Distinctive team tokens only — drop generic location/qualifier words so
    # the team fallback can't false-match an unrelated club's highlight.
    _generic = {"state", "university", "college", "valley", "city", "county",
                "river", "black", "north", "south", "east", "west", "team",
                "baseball", "club", "the", "los", "san", "new", "blue", "red"}
    team_tokens = [t.lower() for t in team_name.split()
                   if len(t) >= 4 and t.lower() not in _generic]

    def _url(v):
        return f"https://youtu.be/{v}"

    def _words(t):
        # whole words only, so "Berg" can't match inside "Bergman"
        return set(re.findall(r"[a-z0-9']+", t.lower()))

    def _action(tl):
        return any(w in tl for w in action_words)

    # Drop any interview / talk / off-field content up front.
    cand = [(v, t) for v, t in vids
            if not any(w in t.lower() for w in interview_words)]

    # 1) The player's *full* name (first + last as whole words) + an action
    #    word. Full name required — a surname-only match risks a different
    #    player with the same last name (e.g. Cade vs Cal Fisher).
    for v, t in cand:
        if surname in _words(t) and first in _words(t) and _action(t.lower()):
            return _url(v), "player"

    # 2) Fallback: game footage of the player's *team* (distinctive team token
    #    + an action word) — still on-field play, never an interview.
    if team_tokens:
        for v, t in cand:
            tl = t.lower()
            if any(tok in tl for tok in team_tokens) and _action(tl):
                return _url(v), "team"

    # 3) Nothing that's clearly the player/team PLAYING — post no video.
    return None


def download_youtube_clip(url: str, name: str,
                          max_seconds: int = 30) -> "Path | None":
    """Download a YouTube clip and re-encode it to an X-ready MP4.

    yt-dlp grabs the first ``max_seconds`` (X caps native video at 2:20),
    then ffmpeg transcodes to H.264/AAC + faststart with even dimensions
    and 30fps so the tweet upload always accepts it. Returns the MP4 path
    or ``None`` on any failure (caller falls back to posting the link).

    NOTE: this re-hosts third-party highlight footage. Only enable native
    upload for clips you have the right to post — otherwise keep the link.
    """
    import re as _re
    import subprocess

    try:
        import yt_dlp
    except Exception:
        log.warning("yt-dlp not installed — cannot download clip")
        return None

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    safe = _re.sub(r"\W+", "_", name).strip("_").lower() or "clip"
    # duration in the name so re-trimming to a different length can't be
    # short-circuited by a stale cached clip of another length
    raw_tmpl = str(CLIPS_DIR / f"yt_{safe}_{max_seconds}s_raw.%(ext)s")
    final = CLIPS_DIR / f"yt_{safe}_{max_seconds}s.mp4"
    if final.exists():
        return final

    # clear any stale raw parts
    for f in CLIPS_DIR.glob(f"yt_{safe}_{max_seconds}s_raw.*"):
        try:
            f.unlink()
        except OSError:
            pass

    ydl_opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "outtmpl": raw_tmpl,
        "merge_output_format": "mp4",
        "quiet": True, "no_warnings": True, "noprogress": True,
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(0, max_seconds)]),
        "force_keyframes_at_cuts": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        log.warning("yt-dlp download failed for %s", url, exc_info=True)
        return None

    raws = list(CLIPS_DIR.glob(f"yt_{safe}_{max_seconds}s_raw.*"))
    if not raws:
        return None
    raw = raws[0]

    cmd = [
        "ffmpeg", "-y", "-i", str(raw), "-t", str(max_seconds),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0",
        "-vf", "scale='min(1280,iw)':-2,fps=30",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(final),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=240)
    except Exception:
        log.warning("ffmpeg transcode failed for %s", raw.name, exc_info=True)
        return None
    finally:
        try:
            raw.unlink()
        except OSError:
            pass

    return final if final.exists() and final.stat().st_size > 0 else None
