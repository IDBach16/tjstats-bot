"""Text generator: yesterday's hardest pitch."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from pybaseball import statcast

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..config import DEFAULT_HASHTAGS
from ..charts import plot_pitch_movement
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class HardestPitchGenerator(ContentGenerator):
    name = "hardest_pitch"

    async def generate(self) -> PostContent:
        yesterday = date.today() - timedelta(days=1)
        ds = yesterday.strftime("%Y-%m-%d")
        log.info("Pulling Statcast data for %s", ds)

        df = statcast(start_dt=ds, end_dt=ds)
        if df.empty:
            return PostContent(text="")

        # Filter to actual pitches with velocity data
        pitches = df.dropna(subset=["release_speed", "player_name", "pitch_name"])
        if pitches.empty:
            return PostContent(text="")

        fastest = pitches.loc[pitches["release_speed"].idxmax()]
        velo = fastest["release_speed"]
        pitcher = fastest["player_name"]
        pitch_type = fastest["pitch_name"]

        # pybaseball gives "Last, First" — flip to "First Last"
        if ", " in str(pitcher):
            last, first = pitcher.split(", ", 1)
            pitcher = f"{first} {last}"

        # Media: try chart + video (image as main tweet, video as reply)
        image_path = None
        video_path = None
        pitcher_id = fastest.get("pitcher")
        if pitcher_id:
            image_path = plot_pitch_movement(int(pitcher_id), pitcher)
            try:
                video_path = get_pitcher_clip(int(pitcher_id), pitcher)
            except Exception:
                log.warning("Video clip fetch failed for %s", pitcher, exc_info=True)

        # AI analysis using Pitch Profiler season data
        analysis_text = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            if not season_df.empty:
                analysis_text = analyze_pitcher(
                    pitcher, season_df, pitch_profiler.get_season_pitches()
                )
        except Exception:
            log.warning("Analysis generation failed for %s", pitcher, exc_info=True)

        # Lead with the AI take in the main tweet
        stat_line = f"{pitcher} — {velo:.1f} mph {pitch_type}"
        if analysis_text:
            text = (
                f"{analysis_text}\n\n"
                f"Hardest pitch thrown yesterday ({ds}):\n"
                f"{stat_line}\n\n"
                f"Data via @baseaboreball / Statcast {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"Hardest pitch thrown yesterday ({ds}):\n\n"
                f"{stat_line}\n\n"
                f"Data via @baseaboreball / Statcast {DEFAULT_HASHTAGS}"
            )

        # Stats go in reply (graphic already shows them)
        reply_content = None

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitch movement chart for {pitcher}" if image_path else "",
            tags=["hardest_pitch", ds],
            reply=reply_content,
        )
