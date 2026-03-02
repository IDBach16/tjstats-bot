"""Screenshot generator: TJStats pitch movement / location plots."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)
SPACE = HF_SPACES["pitch_plots"]


class PitchPlotsScreenshot(ContentGenerator):
    name = "ss_pitch_plots"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

        image = await take_screenshot(
            url=SPACE["url"],
            player_name=name,
            output_name=f"pitch_plots_{name.replace(' ', '_')}",
            full_page=True,
        )

        text = (
            f"Pitch movement & location plots for {name} "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"Pitch movement and location plots for {name}",
            tags=["ss_pitch_plots", name],
        )
