"""Shared helper: pick a player from the watchlist, avoiding recent posts."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .config import DATA_DIR, MLB_SEASON

log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def pick_player(role: str = "pitchers") -> dict:
    """Pick a random player from the watchlist that hasn't been posted recently.

    Returns dict with at minimum {"name": str, "id": int}.
    """
    watchlist = _load_json(DATA_DIR / "players.json")
    history = _load_json(DATA_DIR / "post_history.json")

    players = watchlist.get(role, [])
    if not players:
        # Fallback defaults if no watchlist configured yet
        players = [
            {"name": "Paul Skenes", "id": 694973},
            {"name": "Tarik Skubal", "id": 669373},
            {"name": "Zack Wheeler", "id": 554430},
            {"name": "Dylan Cease", "id": 656302},
            {"name": "Corbin Burns", "id": 669203},
        ]

    # Avoid recently posted players — check ALL history, not just last 7
    # Only count tags that match actual player names in the watchlist
    player_names = {p["name"] for p in players}
    recent_names = set()
    for entry in history.get("posts", [])[-50:]:
        for tag in entry.get("tags", []):
            if tag in player_names:
                recent_names.add(tag)

    candidates = [p for p in players if p["name"] not in recent_names]
    if not candidates:
        candidates = players  # all used recently, reset
        log.info("All players posted recently, resetting pool")

    player = random.choice(candidates)
    player.setdefault("season", MLB_SEASON)
    return player
