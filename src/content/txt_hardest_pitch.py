"""Text generator: yesterday's hardest pitch."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from pybaseball import statcast

from .base import ContentGenerator, PostContent
from ..config import DEFAULT_HASHTAGS

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

        text = (
            f"Hardest pitch thrown yesterday ({ds}):\n\n"
            f"{pitcher} — {velo:.1f} mph {pitch_type}\n\n"
            f"Data via @baseaboreball / Statcast {DEFAULT_HASHTAGS}"
        )
        return PostContent(text=text, tags=["hardest_pitch", ds])
