"""Text generator: elite pitcher spotlight using Pitch Profiler data."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..config import DATA_DIR, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)


def _load_watchlist() -> list[dict]:
    path = DATA_DIR / "players.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("pitchers", [])
    return []


class PitcherSpotlightGenerator(ContentGenerator):
    name = "pitcher_spotlight"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Try to spotlight a watchlist player first, else pick a top performer
        watchlist = _load_watchlist()
        watchlist_ids = {p["id"] for p in watchlist}

        if watchlist_ids and "pitcher_id" in df.columns:
            candidates = df[df["pitcher_id"].isin(watchlist_ids)]
        else:
            candidates = df

        if candidates.empty:
            candidates = df

        # Pick a strong performer — sort by whiff_rate or k_percent if available
        sort_col = None
        for col in ("whiff_rate", "whiff_percent", "k_percent", "k_rate"):
            if col in candidates.columns:
                sort_col = col
                break

        if sort_col:
            top = candidates.nlargest(20, sort_col)
        else:
            top = candidates.head(20)

        player = top.sample(1).iloc[0]

        # Build tweet — use whatever columns are available
        name = str(player.get("pitcher_name", player.get("player_name", "Unknown")))
        parts = [f"Pitcher Spotlight: {name}\n"]

        stat_map = {
            "era": "ERA",
            "whiff_rate": "Whiff%",
            "whiff_percent": "Whiff%",
            "k_percent": "K%",
            "k_rate": "K%",
            "bb_percent": "BB%",
            "bb_rate": "BB%",
            "xera": "xERA",
            "stuff_plus": "Stuff+",
            "ip": "IP",
        }
        shown = 0
        for col, label in stat_map.items():
            if col in player.index and shown < 5:
                val = player[col]
                if isinstance(val, float):
                    parts.append(f"{label}: {val:.1f}")
                else:
                    parts.append(f"{label}: {val}")
                shown += 1

        parts.append(f"\nData via @mlbpitchprofiler {DEFAULT_HASHTAGS}")
        text = "\n".join(parts)
        return PostContent(text=text, tags=["pitcher_spotlight", name])
