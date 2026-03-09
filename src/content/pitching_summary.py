"""Content generator: TJStats-style pitching summary dashboard."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)


class PitchingSummaryGenerator(ContentGenerator):
    name = "pitching_summary"

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

        image_path = plot_pitching_summary(
            name, season_df, pitches_df,
            team=team, player_id=player_id,
        )
        if not image_path:
            log.warning("Pitching summary rendering failed for %s", name)
            return PostContent(text="")

        # Build tweet text with key stats
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
                for col, label, fmt in [
                    ("era", "ERA", ".2f"),
                    ("fip", "FIP", ".2f"),
                    ("strike_out_percentage", "K%", None),
                    ("whiff_rate", "Whiff%", None),
                ]:
                    if col in p.index:
                        try:
                            val = float(p[col])
                            if fmt:
                                summary_parts.append(
                                    f"{label}: {format(val, fmt)}"
                                )
                            else:
                                summary_parts.append(
                                    f"{label}: {val * 100:.1f}%"
                                )
                        except (TypeError, ValueError):
                            pass

        stat_line = " | ".join(summary_parts) if summary_parts else ""
        stat_section = f"\n\n{stat_line}" if stat_line else ""

        text = (
            f"{name}'s {MLB_SEASON} Pitching Summary"
            f"{stat_section}"
            f"\n\n@TJStats {DEFAULT_HASHTAGS}"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Season pitching summary dashboard for {name}",
            tags=["pitching_summary", name],
        )
