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

        text = (
            f"Pitch movement profile for {name} "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(int(player_id), name)
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
            alt_text=f"Pitch movement profile for {name}",
            tags=["ss_pitch_plots", name],
            reply=reply_content,
        )
