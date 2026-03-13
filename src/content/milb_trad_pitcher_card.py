"""Content generator: MiLB traditional pitcher card (non-AAA levels)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..milb_traditional import (
    fetch_level_pitchers, fetch_game_log, get_league_averages,
    pick_traditional_player,
)
from ..milb_statcast import LEVEL_NAMES
from ..charts import plot_traditional_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

# Rotate through non-AAA levels
_LEVELS = ["AA", "A+"]
_level_idx = 0


class MiLBTradPitcherCardGenerator(ContentGenerator):
    name = "milb_trad_pitcher_card"

    async def generate(self) -> PostContent:
        global _level_idx
        level = _LEVELS[_level_idx % len(_LEVELS)]
        _level_idx += 1

        player = pick_traditional_player(level=level, min_ip=30.0)
        if not player:
            log.warning("No qualified traditional pitcher at %s", level)
            return PostContent(text="")

        name = player["name"]
        team = player.get("team", "")
        player_id = player.get("id")

        season_df = fetch_level_pitchers(level=level)
        if season_df.empty:
            log.warning("No traditional data for %s at %s", name, level)
            return PostContent(text="")

        # Fetch game log and league averages for right panel
        game_log = fetch_game_log(player_id, level=level) if player_id else []
        league_avgs = get_league_averages(level=level)

        level_name = LEVEL_NAMES.get(level, level)
        image_path = plot_traditional_pitcher_card(
            name, season_df,
            player_id=player_id, team=team, level=level,
            game_log=game_log, league_avgs=league_avgs,
        )
        if not image_path:
            log.warning("Traditional pitcher card rendering failed for %s", name)
            return PostContent(text="")

        text = (
            f"{name}'s {MLB_SEASON} {level_name} Pitcher Card"
            f"\n\n@TJStats {DEFAULT_HASHTAGS} #MiLB"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"MiLB {level} traditional pitcher card for {name}",
            tags=["milb_trad_pitcher_card", name, level],
        )
