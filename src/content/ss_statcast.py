"""Screenshot generator: TJStats Statcast percentile bar cards."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)
SPACE = HF_SPACES["statcast_cards"]


class StatcastCardsScreenshot(ContentGenerator):
    name = "ss_statcast_cards"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

        image = await take_screenshot(
            url=SPACE["url"],
            player_name=name,
            output_name=f"statcast_card_{name.replace(' ', '_')}",
        )
        if not image:
            log.warning("Screenshot failed for %s — skipping post", name)
            return PostContent(text="")

        text = (
            f"{name}'s Statcast percentile rankings "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"Statcast percentile bars for {name}",
            tags=["ss_statcast_cards", name],
        )
