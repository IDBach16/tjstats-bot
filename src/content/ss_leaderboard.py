"""Screenshot generator: TJStats Statcast leaderboard."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)
SPACE = HF_SPACES["leaderboard"]


class LeaderboardScreenshot(ContentGenerator):
    name = "ss_leaderboard"

    async def generate(self) -> PostContent:
        # Leaderboard doesn't need a player selection — just screenshot the page
        image = await take_screenshot(
            url=SPACE["url"],
            output_name="statcast_leaderboard",
        )
        if not image:
            log.warning("Screenshot failed — skipping post")
            return PostContent(text="")

        text = (
            f"{MLB_SEASON} Statcast Pitching Leaderboard "
            f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
        )
        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"{MLB_SEASON} Statcast pitching leaderboard",
            tags=["ss_leaderboard"],
        )
