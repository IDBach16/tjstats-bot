"""Generator: Pitch Movement Profile — local matplotlib chart + arsenal text."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..charts import plot_movement_profile
from ..config import DEFAULT_HASHTAGS
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


def _build_arsenal_text(name: str, pitch_data) -> str:
    """Build arsenal breakdown text from pitch-type-level data."""
    import pandas as pd

    if pitch_data is None or pitch_data.empty:
        return ""

    name_col = None
    for c in ("pitcher_name", "player_name", "name"):
        if c in pitch_data.columns:
            name_col = c
            break
    if not name_col:
        return ""

    rows = pitch_data[pitch_data[name_col] == name]
    if rows.empty or "pitch_type" not in rows.columns:
        return ""

    # Coerce + aggregate by pitch type
    for nc in ("velocity", "percentage_thrown"):
        if nc in rows.columns:
            rows = rows.copy()
            rows[nc] = pd.to_numeric(rows[nc], errors="coerce")

    agg = {}
    if "velocity" in rows.columns:
        agg["velocity"] = "mean"
    if "percentage_thrown" in rows.columns:
        agg["percentage_thrown"] = "sum"
    if not agg:
        return ""

    grouped = rows.groupby("pitch_type", as_index=False).agg(agg)
    if "percentage_thrown" in grouped.columns:
        grouped = grouped.sort_values("percentage_thrown", ascending=False)

    from ..charts import PITCH_NAMES

    lines = []
    for _, row in grouped.iterrows():
        pt = str(row["pitch_type"])
        display = PITCH_NAMES.get(pt, pt)
        velo = row.get("velocity")
        if pd.notna(velo):
            lines.append(f"{display} — {float(velo):.1f} mph avg")
        else:
            lines.append(display)

    return "\n".join(lines)


class MovementProfileGenerator(ContentGenerator):
    name = "movement_profile"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]
        player_id = player.get("id")

        pitches_df = pitch_profiler.get_season_pitches()
        if pitches_df.empty:
            log.warning("No pitch data available")
            return PostContent(text="")

        image = plot_movement_profile(name, pitches_df)
        if not image:
            log.warning("Movement profile chart failed for %s", name)
            return PostContent(text="")

        arsenal_text = _build_arsenal_text(name, pitches_df)

        if arsenal_text:
            text = (
                f"{name}'s pitch movement profile:\n\n"
                f"{arsenal_text}\n\n"
                f"@TJStats {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name}'s pitch movement profile "
                f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
            )

        # Fetch video clip
        video_path = None
        if player_id:
            try:
                video_path = get_pitcher_clip(int(player_id), name)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        reply_content = None
        try:
            season_df = pitch_profiler.get_season_pitchers()
            if not season_df.empty:
                analysis_text = analyze_pitcher(name, season_df, pitches_df)
                if analysis_text:
                    reply_content = PostContent(text=analysis_text, tags=["analysis"])
        except Exception:
            log.warning("Analysis failed for %s", name, exc_info=True)

        return PostContent(
            text=text,
            image_path=image,
            video_path=video_path,
            alt_text=f"Pitch movement profile for {name}",
            tags=["movement_profile", name],
            reply=reply_content,
        )
