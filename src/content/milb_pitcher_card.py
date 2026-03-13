"""Content generator: MiLB pitcher card image."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..milb_statcast import (
    get_milb_season_pitchers, get_milb_season_pitches, pick_milb_player,
    LEVEL_NAMES,
)
from ..charts import plot_milb_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

# Rotate through levels
_LEVELS = ["AA", "A+"]
_level_idx = 0


class MiLBPitcherCardGenerator(ContentGenerator):
    name = "milb_pitcher_card"

    async def generate(self) -> PostContent:
        global _level_idx
        level = _LEVELS[_level_idx % len(_LEVELS)]
        _level_idx += 1

        player = pick_milb_player(level=level, min_pitches=200)
        if not player:
            log.warning("No qualified MiLB pitcher found at %s", level)
            return PostContent(text="")

        name = player["name"]
        # Fix "Last, First" format from Savant
        if "," in name:
            parts = name.split(", ")
            name = f"{parts[1]} {parts[0]}"

        team = player.get("team", "")
        player_id = player.get("id")

        season_df = get_milb_season_pitchers(level=level)
        pitches_df = get_milb_season_pitches(level=level)

        if season_df.empty or pitches_df.empty:
            log.warning("No MiLB data for %s at %s", name, level)
            return PostContent(text="")

        # Fix name in DataFrames
        if player_id:
            pid_col = "player_id"
            season_df = season_df.copy()
            pitches_df = pitches_df.copy()
            season_df.loc[season_df[pid_col] == player_id, "pitcher_name"] = name
            pitches_df.loc[pitches_df[pid_col] == player_id, "pitcher_name"] = name

        level_name = LEVEL_NAMES.get(level, level)
        image_path = plot_milb_pitcher_card(
            name, season_df, pitches_df,
            team=team, player_id=player_id, level=level,
        )
        if not image_path:
            log.warning("MiLB pitcher card rendering failed for %s", name)
            return PostContent(text="")

        text = (
            f"{name}'s {MLB_SEASON} {level_name} Pitcher Card"
            f"\n\n@TJStats {DEFAULT_HASHTAGS} #MiLB"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"MiLB {level} pitcher card for {name}",
            tags=["milb_pitcher_card", name, level],
        )
