"""Text generator: Guess the Pitcher — anonymized stat block with delayed reveal."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, safe_stat, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..config import DEFAULT_HASHTAGS
from ..charts import plot_pitch_locations

log = logging.getLogger(__name__)

_TEASER_HOOKS = [
    "Can you guess this pitcher?",
    "Name this arm.",
    "Who is this pitcher?",
    "Guess the pitcher from the stats alone.",
    "Which pitcher posted these numbers?",
    "Do you know who this is?",
]

# Anonymized stat block: all stats, NO name
_ANON_STATS = [
    ("era", "ERA", False),
    ("fip", "FIP", False),
    ("strike_out_percentage", "K%", True),
    ("walk_percentage", "BB%", True),
    ("whiff_rate", "Whiff%", True),
    ("chase_percentage", "Chase%", True),
    ("stuff_plus", "Stuff+", False),
    ("pitching_plus", "Pitching+", False),
]


class GuessThePitcherGenerator(ContentGenerator):
    name = "guess_pitcher"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to qualified pitchers
        if "innings_pitched" in df.columns:
            qualified = df[df["innings_pitched"] >= 20]
            if not qualified.empty:
                df = qualified

        player_info = pick_player()
        player_name = player_info["name"]

        # Find player row in data
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in df.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        matches = df[df[name_col] == player_name]
        if matches.empty:
            # Fallback: pick from top performers
            sort_col = "whiff_rate" if "whiff_rate" in df.columns else None
            if sort_col:
                top = df.nlargest(20, sort_col)
            else:
                top = df.head(20)
            player = top.sample(1).iloc[0]
            player_name = get_name(player)
        else:
            player = matches.iloc[0]

        # Build anonymized stat block
        stat_lines = []
        for col, label, pct in _ANON_STATS:
            if col in player.index:
                stat_lines.append(f"{label}: {fmt_stat(player[col], pct)}")

        # Pair stats two per line
        paired = []
        for i in range(0, len(stat_lines), 2):
            paired.append(" | ".join(stat_lines[i:i + 2]))
        stat_block = "\n".join(paired)

        hook = random.choice(_TEASER_HOOKS)
        text = f"{hook}\n\n{stat_block}\n\nDrop your guess below!\n\n{DEFAULT_HASHTAGS}"

        # Media: anonymized chart only (no video — would reveal the pitcher)
        image_path = None
        pitcher_id = player_info.get("id") or player.get("pitcher_id") or player.get("player_id")
        if pitcher_id:
            image_path = plot_pitch_locations(int(pitcher_id), player_name, anonymize=True)

        # Build reveal reply
        reply = PostContent(
            text=f"The answer is {player_name}!\n\nData via @mlbpitchprofiler {DEFAULT_HASHTAGS}",
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text="Anonymized pitch location chart" if image_path else "",
            tags=["guess_pitcher", player_name],
            reply=reply,
        )
