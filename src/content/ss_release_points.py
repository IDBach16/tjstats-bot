"""Generator: Release point chart from catcher's perspective (Statcast)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_release_points
from ..config import DEFAULT_HASHTAGS
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class ReleasePointGenerator(ContentGenerator):
    name = "release_points"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]
        player_id = player.get("id")

        if not player_id:
            log.warning("No player_id for %s", name)
            return PostContent(text="")

        image = plot_release_points(player_id, name)
        if not image:
            log.warning("Release point chart failed for %s", name)
            return PostContent(text="")

        text = (
            f"Release points for {name} "
            f"— catcher's perspective\n\n"
            f"@TJStats {DEFAULT_HASHTAGS}"
        )

        # Fetch video clip
        video_path = None
        try:
            video_path = get_pitcher_clip(player_id, name)
        except Exception:
            log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        reply_content = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            pitches_df = pitch_profiler.get_season_pitches()
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
            alt_text=f"Release point chart for {name} from catcher perspective",
            tags=["release_points", name],
            reply=reply_content,
        )
