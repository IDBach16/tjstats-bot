"""Text generator: rotating stat-of-the-day leaders."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

# Stats to rotate through, with display names and sort direction
STATS = [
    {"col": "whiff_rate", "label": "Whiff Rate", "fmt": ".1f", "suffix": "%", "top": True},
    {"col": "k_percent", "label": "K%", "fmt": ".1f", "suffix": "%", "top": True},
    {"col": "era", "label": "ERA", "fmt": ".2f", "suffix": "", "top": False},
    {"col": "xera", "label": "xERA", "fmt": ".2f", "suffix": "", "top": False},
    {"col": "stuff_plus", "label": "Stuff+", "fmt": ".0f", "suffix": "", "top": True},
    {"col": "bb_percent", "label": "BB%", "fmt": ".1f", "suffix": "%", "top": False},
    {"col": "chase_rate", "label": "Chase Rate", "fmt": ".1f", "suffix": "%", "top": True},
    {"col": "csw_rate", "label": "CSW%", "fmt": ".1f", "suffix": "%", "top": True},
]


class StatOfDayGenerator(ContentGenerator):
    name = "stat_of_day"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to qualified pitchers if IP column exists
        if "ip" in df.columns:
            df = df[df["ip"] >= 20]

        # Pick a random stat that exists in the data
        available = [s for s in STATS if s["col"] in df.columns]
        if not available:
            return PostContent(text="")

        stat = random.choice(available)
        col = stat["col"]
        ascending = not stat["top"]
        leaders = df.nlargest(5, col) if stat["top"] else df.nsmallest(5, col)

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in leaders.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        header = (
            f"{MLB_SEASON} {stat['label']} Leaders (min 20 IP):\n"
            if "ip" in df.columns
            else f"{MLB_SEASON} {stat['label']} Leaders:\n"
        )
        lines = [header]
        for i, (_, row) in enumerate(leaders.iterrows(), 1):
            val = row[col]
            formatted = f"{val:{stat['fmt']}}{stat['suffix']}"
            lines.append(f"{i}. {row[name_col]} — {formatted}")

        lines.append(f"\nData via @mlbpitchprofiler {DEFAULT_HASHTAGS}")
        text = "\n".join(lines)
        return PostContent(text=text, tags=["stat_of_day", stat["col"]])
