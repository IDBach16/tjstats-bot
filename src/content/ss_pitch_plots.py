"""Generator: Pitch plots — local movement profile chart from Pitch Profiler data."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_movement_profile
from ..config import DEFAULT_HASHTAGS
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class PitchPlotsScreenshot(ContentGenerator):
    name = "ss_pitch_plots"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]
        player_id = player.get("id")

        pitches_df = pitch_profiler.get_season_pitches()
        if pitches_df.empty:
            log.warning("No pitch data available")
            return PostContent(text="")

        image = plot_movement_profile(name, pitches_df)
        if not image:
            log.warning("Movement profile chart failed for %s", name)
            return PostContent(text="")

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(int(player_id), name)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        analysis_text = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            if not season_df.empty:
                analysis_text = analyze_pitcher(name, season_df, pitches_df)
        except Exception:
            log.warning("Analysis failed for %s", name, exc_info=True)

        # Lead with the AI take in the main tweet
        if analysis_text:
            text = (
                f"{analysis_text}\n\n"
                f"Pitch movement profile for {name}\n\n"
                f"@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"Pitch movement profile for {name}\n\n"
                f"@TJStats {DEFAULT_HASHTAGS}"
            )

        # Stats go in reply (graphic already shows them)
        reply_content = None

        return PostContent(
            text=text,
            image_path=image,
            video_path=video_path,
            alt_text=f"Pitch movement profile for {name}",
            tags=["ss_pitch_plots", name],
            reply=reply_content,
        )
