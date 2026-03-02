"""Screenshot generator: TJStats pitching summary card."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)
SPACE = HF_SPACES["pitching_summary"]


class PitchingSummaryScreenshot(ContentGenerator):
    name = "ss_pitching_summary"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

        image = await take_screenshot(
            url=SPACE["url"],
            player_name=name,
            output_name=f"pitching_summary_{name.replace(' ', '_')}",
        )

        text = (
            f"{name}'s {player.get('season', 2025)} pitching summary "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"TJStats pitching summary card for {name}",
            tags=["ss_pitching_summary", name],
        )
