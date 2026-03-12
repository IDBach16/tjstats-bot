"""Generator: Arsenal usage breakdown horizontal bar chart."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_arsenal_usage
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class ArsenalUsageGenerator(ContentGenerator):
    name = "arsenal_usage"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]
        player_id = player.get("id")

        pitches_df = pitch_profiler.get_season_pitches()
        if pitches_df.empty:
            log.warning("No pitch data available")
            return PostContent(text="")

        image = plot_arsenal_usage(name, pitches_df, player_id=player_id)
        if not image:
            log.warning("Arsenal usage chart failed for %s", name)
            return PostContent(text="")

        text = (
            f"{name}'s {MLB_SEASON} Arsenal Usage Breakdown\n\n"
            f"@TJStats {DEFAULT_HASHTAGS}"
        )

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(player_id, name)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        reply_content = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            if not season_df.empty:
                analysis_text = analyze_pitcher(name, season_df, pitches_df)
                if analysis_text:
                    reply_content = PostContent(text=analysis_text, tags=["analysis"])
        except Exception:
            log.warning("Analysis failed for %s", name, exc_info=True)

        return PostContent(
            text=text,
            image_path=image,
            video_path=video_path,
            alt_text=f"Arsenal usage breakdown for {name}",
            tags=["arsenal_usage", name],
            reply=reply_content,
        )
