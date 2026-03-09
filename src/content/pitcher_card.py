"""Content generator: premium pitcher card image (PLV-style)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, safe_stat, build_stat_block, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..charts import plot_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)


class PitcherCardGenerator(ContentGenerator):
    name = "pitcher_card"

    async def generate(self) -> PostContent:
        player_info = pick_player()
        name = player_info["name"]
        team = player_info.get("team")
        player_id = player_info.get("id")

        season_df = pitch_profiler.get_season_pitchers()
        if season_df.empty:
            log.warning("No season pitcher data available")
            return PostContent(text="")

        pitches_df = pitch_profiler.get_season_pitches()

        # Render the card
        image_path = plot_pitcher_card(name, season_df, pitches_df, team=team,
                                       player_id=player_id)
        if not image_path:
            log.warning("Pitcher card rendering failed for %s", name)
            return PostContent(text="")

        # Build tweet text with a compact stat summary
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in season_df.columns:
                name_col = c
                break

        summary_parts: list[str] = []
        if name_col:
            matches = season_df[season_df[name_col] == name]
            if not matches.empty:
                p = matches.iloc[0]
                era = safe_stat(p, "era")
                k_pct = safe_stat(p, "strike_out_percentage", pct=True)
                whiff = safe_stat(p, "whiff_rate", pct=True)
                stuff = safe_stat(p, "stuff_plus")
                if era is not None:
                    summary_parts.append(f"ERA: {era:.2f}")
                if k_pct is not None:
                    summary_parts.append(f"K%: {k_pct:.1f}")
                if whiff is not None:
                    summary_parts.append(f"Whiff%: {whiff:.1f}")
                if stuff is not None:
                    summary_parts.append(f"Stuff+: {stuff:.0f}")

        stat_line = " | ".join(summary_parts) if summary_parts else ""
        stat_section = f"\n\n{stat_line}" if stat_line else ""

        text = (
            f"{name}'s {MLB_SEASON} Pitcher Card"
            f"{stat_section}"
            f"\n\n@TJStats {DEFAULT_HASHTAGS}"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Pitcher card for {name} showing season stats and arsenal breakdown",
            tags=["pitcher_card", name],
        )
