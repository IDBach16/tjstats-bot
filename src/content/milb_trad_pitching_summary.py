"""Content generator: MiLB traditional pitching summary (non-AAA levels)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..milb_traditional import (
    fetch_level_pitchers, fetch_game_log, fetch_monthly_splits,
    get_league_averages, pick_traditional_player,
)
from ..milb_statcast import LEVEL_NAMES
from ..analysis import generate_analysis
from ..charts import plot_traditional_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)

# Rotate through non-AAA levels
_LEVELS = ["AA", "A+"]
_level_idx = 0


class MiLBTradPitchingSummaryGenerator(ContentGenerator):
    name = "milb_trad_pitching_summary"

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

        # Fetch supplementary data
        game_log = fetch_game_log(player_id, level=level) if player_id else []
        monthly = fetch_monthly_splits(player_id, level=level) if player_id else []
        league_avgs = get_league_averages(level=level)

        level_name = LEVEL_NAMES.get(level, level)
        image_path = plot_traditional_pitching_summary(
            name, season_df,
            player_id=player_id, team=team, level=level,
            game_log=game_log, monthly_splits=monthly,
            league_avgs=league_avgs,
        )
        if not image_path:
            log.warning("Traditional summary rendering failed for %s", name)
            return PostContent(text="")

        # AI analysis — build stats dict for the prompt
        analysis_text = None
        try:
            row = season_df[season_df["pitcher_name"] == name]
            if row.empty and player_id:
                row = season_df[season_df["player_id"] == player_id]
            if not row.empty:
                p = row.iloc[0]
                season_stats = {}
                for col, label, fmt in [
                    ("era", "ERA", ".2f"),
                    ("whip", "WHIP", ".2f"),
                    ("inningsPitched", "IP", ".1f"),
                    ("strikeoutsPer9Inn", "K/9", ".2f"),
                    ("walksPer9Inn", "BB/9", ".2f"),
                    ("homeRunsPer9", "HR/9", ".2f"),
                    ("avg", "BAVG", ".3f"),
                    ("strikeoutWalkRatio", "K/BB", ".2f"),
                    ("groundOutsToAirouts", "GO/AO", ".2f"),
                    ("fip", "FIP", ".2f"),
                ]:
                    if col in p.index:
                        try:
                            season_stats[label] = format(float(p[col]), fmt)
                        except (TypeError, ValueError):
                            pass
                if season_stats:
                    analysis_text = generate_analysis(
                        name, season_stats, []
                    )
        except Exception:
            log.warning("AI analysis failed for %s", name, exc_info=True)

        # Video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(player_id, name)
                if video_path:
                    log.info("Got video clip for %s: %s", name, video_path)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        text = (
            f"{name}'s {MLB_SEASON} {level_name} Pitching Summary"
            f"\n\n@TJStats {DEFAULT_HASHTAGS} #MiLB"
        )

        reply_content = None
        if analysis_text:
            reply_content = PostContent(text=analysis_text, tags=["analysis"])

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"MiLB {level} pitching summary for {name}",
            tags=["milb_trad_pitching_summary", name, level],
            reply=reply_content,
        )
