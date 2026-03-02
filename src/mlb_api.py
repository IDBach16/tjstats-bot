"""MLB Stats API client (free, no auth required)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import requests

from .config import MLB_API_BASE

log = logging.getLogger(__name__)


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{MLB_API_BASE}/{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Schedule ──────────────────────────────────────────────────────────

def get_schedule(target_date: date | None = None) -> list[dict]:
    """Return list of game dicts for a given date (default: yesterday)."""
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    ds = target_date.isoformat()
    data = _get("schedule", {"sportId": 1, "date": ds})
    games = []
    for block in data.get("dates", []):
        for g in block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status == "Postponed":
                continue
            teams = g.get("teams", {})
            games.append({
                "game_pk": g["gamePk"],
                "date": ds,
                "status": status,
                "home_team": teams.get("home", {}).get("team", {}).get("name", ""),
                "away_team": teams.get("away", {}).get("team", {}).get("name", ""),
                "home_id": teams.get("home", {}).get("team", {}).get("id"),
                "away_id": teams.get("away", {}).get("team", {}).get("id"),
                "home_score": teams.get("home", {}).get("score"),
                "away_score": teams.get("away", {}).get("score"),
            })
    return games


def get_todays_schedule() -> list[dict]:
    """Games scheduled for today."""
    return get_schedule(date.today())


# ── Players ───────────────────────────────────────────────────────────

def search_player(name: str) -> list[dict]:
    """Search MLB players by name fragment."""
    data = _get("people/search", {"names": name, "sportIds": 1})
    results = []
    for row in data.get("people", []):
        results.append({
            "id": row["id"],
            "full_name": row.get("fullFLName", row.get("fullName", "")),
            "team": row.get("currentTeam", {}).get("name", ""),
            "position": row.get("primaryPosition", {}).get("abbreviation", ""),
        })
    return results


def get_player(player_id: int) -> dict:
    """Get detailed info for a single player."""
    data = _get(f"people/{player_id}", {
        "hydrate": "currentTeam,stats(type=season)"
    })
    people = data.get("people", [])
    return people[0] if people else {}


# ── Game feed ─────────────────────────────────────────────────────────

def get_game_feed(game_pk: int) -> dict:
    """Full live feed for a game (v1.1)."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()
