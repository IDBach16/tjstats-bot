"""Generator: Pitch plots — local movement profile chart from Pitch Profiler data."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..charts import plot_movement_profile
from ..config import DEFAULT_HASHTAGS

log = logging.getLogger(__name__)


class PitchPlotsScreenshot(ContentGenerator):
    name = "ss_pitch_plots"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

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
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"Pitch movement profile for {name}",
            tags=["ss_pitch_plots", name],
        )
