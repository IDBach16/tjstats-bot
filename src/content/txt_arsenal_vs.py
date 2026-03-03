"""Text generator: Arsenal vs Arsenal — side-by-side pitcher comparison."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from ._helpers import build_stat_block, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..config import DEFAULT_HASHTAGS
from ..charts import plot_pitch_heatmap
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class ArsenalVsGenerator(ContentGenerator):
    name = "arsenal_vs"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to qualified pitchers
        if "innings_pitched" in df.columns:
            qualified = df[df["innings_pitched"] >= 20]
            if not qualified.empty:
                df = qualified

        # Pitcher A: from the watchlist via pick_player
        player_a_info = pick_player()
        name_a = player_a_info["name"]

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in df.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        matches_a = df[df[name_col] == name_a]

        # Pitcher B: pick from top-20 by Stuff+ (excluding Pitcher A)
        sort_col = "stuff_plus" if "stuff_plus" in df.columns else None
        if sort_col:
            top = df.nlargest(20, sort_col)
        else:
            top = df.head(20)

        pool_b = top[top[name_col] != name_a]
        if pool_b.empty:
            pool_b = top

        player_b = pool_b.sample(1).iloc[0]
        name_b = get_name(player_b)

        # If Pitcher A wasn't found in the data, pick another from the pool
        if matches_a.empty:
            remaining = pool_b[pool_b[name_col] != name_b]
            if remaining.empty:
                remaining = pool_b
            player_a = remaining.sample(1).iloc[0]
            name_a = get_name(player_a)
        else:
            player_a = matches_a.iloc[0]

        # Ensure we don't compare the same pitcher
        if name_a == name_b:
            return PostContent(text="")

        block_a = build_stat_block(player_a)
        block_b = build_stat_block(player_b)

        text = (
            f"Which arsenal would you rather have?\n\n"
            f"Pitcher A: {name_a}\n"
            f"{block_a}\n\n"
            f"Pitcher B: {name_b}\n"
            f"{block_b}\n\n"
            f"Reply with A or B!\n\n"
            f"Data via @mlbpitchprofiler {DEFAULT_HASHTAGS}"
        )

        # Media: try chart + video (image as main tweet, video as reply)
        image_path = None
        video_path = None
        pitcher_id = (
            player_a_info.get("id")
            or player_a.get("pitcher_id")
            or player_a.get("player_id")
        )
        if pitcher_id:
            image_path = plot_pitch_heatmap(int(pitcher_id), name_a)
            try:
                video_path = get_pitcher_clip(int(pitcher_id), name_a)
            except Exception:
                log.warning("Video clip fetch failed for %s", name_a, exc_info=True)

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitch heatmap for {name_a}" if image_path else "",
            tags=["arsenal_vs", name_a, name_b],
        )
