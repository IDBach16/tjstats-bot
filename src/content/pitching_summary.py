"""Content generator: TJStats-style pitching summary dashboard."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


class PitchingSummaryGenerator(ContentGenerator):
    name = "pitching_summary"

    async def generate(self) -> PostContent:
        # Try up to 3 players if card rendering fails (sparse early-season data)
        for attempt in range(3):
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
            if image_path:
                break
            log.warning("Pitching summary rendering failed for %s (attempt %d), trying another",
                        name, attempt + 1)

        if not image_path:
            log.warning("All pitching summary attempts failed")
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
                                summary_parts.append(f"{label}: {format(val, fmt)}")
                            else:
                                summary_parts.append(f"{label}: {val * 100:.1f}%")
                        except (TypeError, ValueError):
                            pass

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
                f"\n\n{name}'s {MLB_SEASON} Pitching Summary"
                f"\n\n@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name}'s {MLB_SEASON} Pitching Summary"
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
            alt_text=f"Season pitching summary dashboard for {name}",
            tags=["pitching_summary", name],
            reply=reply_content,
        )
