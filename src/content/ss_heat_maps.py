"""Screenshot generator: TJStats pitcher heat maps."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)
SPACE = HF_SPACES["heat_maps"]


class HeatMapsScreenshot(ContentGenerator):
    name = "ss_heat_maps"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

        image = await take_screenshot(
            url=SPACE["url"],
            player_name=name,
            output_name=f"heat_map_{name.replace(' ', '_')}",
        )

        text = (
            f"Pitch heat maps for {name} "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"Pitch heat maps for {name}",
            tags=["ss_heat_maps", name],
        )
