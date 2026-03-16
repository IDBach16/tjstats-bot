"""Content generator: premium pitcher card image (PLV-style)."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, safe_stat, build_stat_block, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..analysis import analyze_pitcher
from ..charts import plot_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

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

        # AI analysis
        analysis_text = analyze_pitcher(name, season_df, pitches_df)

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(player_id, name)
                if video_path:
                    log.info("Got video clip for %s: %s", name, video_path)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        stat_line = " | ".join(summary_parts) if summary_parts else ""

        # Lead with the take, not the title
        if analysis_text:
            text = (
                f"{analysis_text}"
                f"\n\n{name}'s {MLB_SEASON} Pitcher Card"
                f"\n\n@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name}'s {MLB_SEASON} Pitcher Card"
                f"\n\n@TJStats {DEFAULT_HASHTAGS}"
            )

        # Stats go in the reply (the graphic already shows them)
        reply_content = None
        if stat_line:
            reply_content = PostContent(
                text=f"{name} | {stat_line}",
                tags=["stats"],
            )

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitcher card for {name} showing season stats and arsenal breakdown",
            tags=["pitcher_card", name],
            reply=reply_content,
        )
