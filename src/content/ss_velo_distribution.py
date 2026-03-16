"""Generator: Velocity distribution violin chart (Statcast)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_velocity_distribution
from ..config import DEFAULT_HASHTAGS
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class VeloDistributionGenerator(ContentGenerator):
    name = "velo_distribution"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]
        player_id = player.get("id")

        if not player_id:
            log.warning("No player_id for %s", name)
            return PostContent(text="")

        image = plot_velocity_distribution(player_id, name)
        if not image:
            log.warning("Velocity distribution chart failed for %s", name)
            return PostContent(text="")

        # Fetch video clip
        video_path = None
        try:
            video_path = get_pitcher_clip(player_id, name)
        except Exception:
            log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        analysis_text = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            pitches_df = pitch_profiler.get_season_pitches()
            if not season_df.empty:
                analysis_text = analyze_pitcher(name, season_df, pitches_df)
        except Exception:
            log.warning("Analysis failed for %s", name, exc_info=True)

        # Lead with the AI take in the main tweet
        if analysis_text:
            text = (
                f"{analysis_text}\n\n"
                f"{name}'s velocity distribution by pitch type\n\n"
                f"@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name}'s velocity distribution by pitch type\n\n"
                f"@TJStats {DEFAULT_HASHTAGS}"
            )

        # Stats go in reply (graphic already shows them)
        reply_content = None

        return PostContent(
            text=text,
            image_path=image,
            video_path=video_path,
            alt_text=f"Velocity distribution chart for {name}",
            tags=["velo_distribution", name],
            reply=reply_content,
        )
