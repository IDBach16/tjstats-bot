"""Pitch Profiler API client (Patreon-gated Oracle REST endpoint)."""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .config import PITCH_PROFILER_BASE, PITCH_PROFILER_API_KEY, MLB_SEASON

log = logging.getLogger(__name__)

_KEY = PITCH_PROFILER_API_KEY
_BASE = PITCH_PROFILER_BASE


def _fetch(endpoint: str) -> pd.DataFrame:
    """Generic fetch → DataFrame helper."""
    url = f"{_BASE}/{endpoint}"
    log.debug("GET %s", url.replace(_KEY, "***"))
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    items = pd.json_normalize(data).get("items", pd.Series([[]]))[0]
    return pd.json_normalize(items) if items else pd.DataFrame()


# ── Career-level ──────────────────────────────────────────────────────

def get_career_pitchers() -> pd.DataFrame:
    """Aggregated pitcher stats since 2020."""
    return _fetch(f"GET_CAREER_PITCHERS/{_KEY}")


def get_career_pitches() -> pd.DataFrame:
    """Aggregated pitch-type data since 2020."""
    return _fetch(f"GET_CAREER_PITCHES/{_KEY}")


# ── Season-level ──────────────────────────────────────────────────────

def get_season_pitchers(season: int = MLB_SEASON) -> pd.DataFrame:
    """Season aggregated pitcher stats."""
    return _fetch(f"GET_SEASON_PITCHERS/{season}/{_KEY}")


def get_season_pitches(season: int = MLB_SEASON) -> pd.DataFrame:
    """Season aggregated pitch-type data."""
    return _fetch(f"GET_SEASON_PITCHES/{season}/{_KEY}")


def get_team_season_pitchers(season: int = MLB_SEASON) -> pd.DataFrame:
    """Season pitcher data by team."""
    return _fetch(f"GET_TEAM_SEASON_PITCHERS/{season}/{_KEY}")


def get_team_season_pitches(season: int = MLB_SEASON) -> pd.DataFrame:
    """Season pitch-type data by team."""
    return _fetch(f"GET_TEAM_SEASON_PITCHES/{season}/{_KEY}")


# ── Game-level ────────────────────────────────────────────────────────

def get_game_pitchers(season: int = MLB_SEASON) -> pd.DataFrame:
    """Game-level pitcher stats for a season."""
    return _fetch(f"GET_GAME_PITCHERS/{season}/{_KEY}")


def get_game_pitches(season: int = MLB_SEASON) -> pd.DataFrame:
    """Game-level pitch-type data for a season."""
    return _fetch(f"GET_GAME_PITCHES/{season}/{_KEY}")


# ── Pitch-by-pitch ───────────────────────────────────────────────────

def get_pbp_game(game_pk: int) -> pd.DataFrame:
    """Pitch-by-pitch data for a single game."""
    return _fetch(f"GET_PBP_GAME/{game_pk}/{_KEY}")


def get_pbp_season_pitcher(season: int, pitcher_id: int) -> pd.DataFrame:
    """Pitch-by-pitch data for one pitcher's full season."""
    return _fetch(f"GET_PBP_SEASON_PITCHER/{season}/{pitcher_id}/{_KEY}")
