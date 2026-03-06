"""Text generator: rotating stat-of-the-day leaders."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..charts import plot_percentile_rankings
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)

# Stats to rotate through, with display names and sort direction
# pct=True means the raw value is a decimal (0-1) that needs ×100 for display
STATS = [
    {"col": "whiff_rate", "label": "Whiff Rate", "fmt": ".1f", "suffix": "%", "top": True, "pct": True},
    {"col": "strike_out_percentage", "label": "K%", "fmt": ".1f", "suffix": "%", "top": True, "pct": True},
    {"col": "era", "label": "ERA", "fmt": ".2f", "suffix": "", "top": False, "pct": False},
    {"col": "fip", "label": "FIP", "fmt": ".2f", "suffix": "", "top": False, "pct": False},
    {"col": "stuff_plus", "label": "Stuff+", "fmt": ".0f", "suffix": "", "top": True, "pct": False},
    {"col": "walk_percentage", "label": "BB%", "fmt": ".1f", "suffix": "%", "top": False, "pct": True},
    {"col": "chase_percentage", "label": "Chase Rate", "fmt": ".1f", "suffix": "%", "top": True, "pct": True},
    {"col": "called_strikes_plus_whiffs_percentage", "label": "CSW%", "fmt": ".1f", "suffix": "%", "top": True, "pct": True},
    {"col": "barrel_percentage", "label": "Barrel%", "fmt": ".1f", "suffix": "%", "top": False, "pct": True},
    {"col": "strikeouts_per_9", "label": "K/9", "fmt": ".1f", "suffix": "", "top": True, "pct": False},
]


class StatOfDayGenerator(ContentGenerator):
    name = "stat_of_day"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to qualified pitchers if IP column exists
        if "innings_pitched" in df.columns:
            df = df[df["innings_pitched"] >= 50]

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
            if "innings_pitched" in df.columns
            else f"{MLB_SEASON} {stat['label']} Leaders:\n"
        )
        lines = [header]
        for i, (_, row) in enumerate(leaders.iterrows(), 1):
            val = row[col]
            if stat.get("pct"):
                val = val * 100
            formatted = f"{val:{stat['fmt']}}{stat['suffix']}"
            lines.append(f"{i}. {row[name_col]} — {formatted}")

        lines.append(f"\nData via @mlbpitchprofiler {DEFAULT_HASHTAGS}")
        text = "\n".join(lines)

        # Media: try chart + video (image as main tweet, video as reply)
        image_path = None
        video_path = None
        top_row = leaders.iloc[0]
        pitcher_id = top_row.get("pitcher_id") or top_row.get("player_id")
        top_name = str(top_row[name_col]) if name_col else None
        if top_name:
            image_path = plot_percentile_rankings(top_name, df)
        if pitcher_id and top_name:
            try:
                video_path = get_pitcher_clip(int(pitcher_id), top_name)
            except Exception:
                log.warning("Video clip fetch failed for %s", top_name, exc_info=True)

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Percentile rankings chart for {top_name}" if image_path else "",
            tags=["stat_of_day", stat["col"]],
        )
