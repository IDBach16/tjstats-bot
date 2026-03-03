"""Screenshot generator: Pitch Movement Profiles — movement chart + arsenal text."""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from .._player_pick import pick_player
from .. import pitch_profiler
from ..screenshot import take_screenshot
from ..config import HF_SPACES, DEFAULT_HASHTAGS

log = logging.getLogger(__name__)
SPACE = HF_SPACES["pitch_plots"]


def _build_arsenal_text(name: str, pitch_data) -> str:
    """Build arsenal breakdown text from pitch-type-level data.

    Returns lines like "4-Seam — 97.2 mph avg" or a simpler fallback.
    """
    if pitch_data is None or pitch_data.empty:
        return ""

    # Find pitcher rows — try common column names
    name_col = None
    for c in ("pitcher_name", "player_name", "name"):
        if c in pitch_data.columns:
            name_col = c
            break
    if not name_col:
        return ""

    rows = pitch_data[pitch_data[name_col] == name]
    if rows.empty:
        return ""

    # Try to find pitch name and velocity columns
    pitch_col = None
    for c in ("pitch_name", "pitch_type", "tagged_pitch_type"):
        if c in rows.columns:
            pitch_col = c
            break

    velo_col = None
    for c in ("release_speed", "avg_velocity", "velocity", "avg_speed"):
        if c in rows.columns:
            velo_col = c
            break

    if not pitch_col:
        return ""

    lines = []
    for _, row in rows.iterrows():
        pitch_name = str(row[pitch_col])
        if not pitch_name or pitch_name == "nan":
            continue
        if velo_col and row.get(velo_col) is not None:
            try:
                velo = float(row[velo_col])
                lines.append(f"{pitch_name} — {velo:.1f} mph avg")
            except (TypeError, ValueError):
                lines.append(pitch_name)
        else:
            lines.append(pitch_name)

    return "\n".join(lines)


class MovementProfileGenerator(ContentGenerator):
    name = "movement_profile"

    async def generate(self) -> PostContent:
        player = pick_player()
        name = player["name"]

        # Take screenshot of pitch movement plots
        image = await take_screenshot(
            url=SPACE["url"],
            player_name=name,
            output_name=f"movement_profile_{name.replace(' ', '_')}",
            full_page=True,
        )

        # Fetch pitch-type data for arsenal breakdown
        arsenal_text = ""
        try:
            pitch_data = pitch_profiler.get_season_pitches()
            arsenal_text = _build_arsenal_text(name, pitch_data)
        except Exception:
            log.warning("Failed to fetch pitch-type data for arsenal text", exc_info=True)

        if arsenal_text:
            text = (
                f"{name}'s pitch movement profile:\n\n"
                f"{arsenal_text}\n\n"
                f"Movement chart via @TJStats\n\n{DEFAULT_HASHTAGS}"
            )
        else:
            # Simpler fallback when pitch-type data is unavailable
            text = (
                f"{name}'s pitch movement profile "
                f"via @TJStats\n\n{DEFAULT_HASHTAGS}"
            )

        return PostContent(
            text=text,
            image_path=image,
            alt_text=f"Pitch movement profile for {name}",
            tags=["movement_profile", name],
        )
