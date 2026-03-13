"""Text generator: Arsenal vs Arsenal — side-by-side pitcher comparison."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from ._helpers import build_stat_block, get_name
from .. import pitch_profiler
from .._player_pick import pick_player
from ..analysis import generate_analysis
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..charts import plot_pitch_heatmap
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)


def _extract_comparison_stats(player_a, name_a: str, player_b, name_b: str) -> str | None:
    """Generate a comparison analysis for two pitchers."""
    from ..analysis import _get_client

    client = _get_client()
    if not client:
        return None

    def _row_stats(p) -> dict[str, str]:
        stats = {}
        for col, label, fmt in [
            ("era", "ERA", ".2f"), ("fip", "FIP", ".2f"),
            ("strike_out_percentage", "K%", None), ("whiff_rate", "Whiff%", None),
            ("stuff_plus", "Stuff+", ".0f"),
        ]:
            if col in p.index:
                try:
                    val = float(p[col])
                    stats[label] = format(val, fmt) if fmt else f"{val * 100:.1f}%"
                except (TypeError, ValueError):
                    pass
        return stats

    stats_a = _row_stats(player_a)
    stats_b = _row_stats(player_b)

    if not stats_a and not stats_b:
        return None

    stats_text_a = "\n".join(f"  {k}: {v}" for k, v in stats_a.items())
    stats_text_b = "\n".join(f"  {k}: {v}" for k, v in stats_b.items())

    prompt = f"""You are a sharp baseball analyst on Twitter/X. Compare these two {MLB_SEASON} pitchers:

Pitcher A — {name_a}:
{stats_text_a}

Pitcher B — {name_b}:
{stats_text_b}

Write a 2-3 sentence comparison take. Who has the edge and why? Be specific with stats.
Rules:
- Keep it UNDER 260 characters (strict Twitter limit)
- Do NOT use hashtags, emojis, or @ mentions
- Do NOT use dashes or hyphens (use commas, periods, or other punctuation instead)
- Sound like a knowledgeable baseball person"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        if len(text) > 270:
            text = text[:267] + "..."
        log.info("Generated comparison (%d chars): %s", len(text), text[:80])
        return text
    except Exception:
        log.warning("Comparison analysis failed", exc_info=True)
        return None


class ArsenalVsGenerator(ContentGenerator):
    name = "arsenal_vs"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to qualified pitchers
        if "innings_pitched" in df.columns:
            qualified = df[df["innings_pitched"] >= 50]
            if not qualified.empty:
                df = qualified

        # Pitcher A: from the watchlist via pick_player
        player_a_info = pick_player()
        name_a = player_a_info["name"]

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in df.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        matches_a = df[df[name_col] == name_a]

        # Pitcher B: pick from top-20 by Stuff+ (excluding Pitcher A)
        sort_col = "stuff_plus" if "stuff_plus" in df.columns else None
        if sort_col:
            top = df.nlargest(20, sort_col)
        else:
            top = df.head(20)

        pool_b = top[top[name_col] != name_a]
        if pool_b.empty:
            pool_b = top

        player_b = pool_b.sample(1).iloc[0]
        name_b = get_name(player_b)

        # If Pitcher A wasn't found in the data, pick another from the pool
        if matches_a.empty:
            remaining = pool_b[pool_b[name_col] != name_b]
            if remaining.empty:
                remaining = pool_b
            player_a = remaining.sample(1).iloc[0]
            name_a = get_name(player_a)
        else:
            player_a = matches_a.iloc[0]

        # Ensure we don't compare the same pitcher
        if name_a == name_b:
            return PostContent(text="")

        block_a = build_stat_block(player_a)
        block_b = build_stat_block(player_b)

        text = (
            f"Which arsenal would you rather have?\n\n"
            f"Pitcher A: {name_a}\n"
            f"{block_a}\n\n"
            f"Pitcher B: {name_b}\n"
            f"{block_b}\n\n"
            f"Reply with A or B!\n\n"
            f"Data via @mlbpitchprofiler {DEFAULT_HASHTAGS}"
        )

        # Media: try chart + video (image as main tweet, video as reply)
        image_path = None
        video_path = None
        pitcher_id = (
            player_a_info.get("id")
            or player_a.get("pitcher_id")
            or player_a.get("player_id")
        )
        if pitcher_id:
            image_path = plot_pitch_heatmap(int(pitcher_id), name_a)
            try:
                video_path = get_pitcher_clip(int(pitcher_id), name_a)
            except Exception:
                log.warning("Video clip fetch failed for %s", name_a, exc_info=True)

        # AI comparison analysis
        reply_content = None
        analysis_text = _extract_comparison_stats(player_a, name_a, player_b, name_b)
        if analysis_text:
            reply_content = PostContent(text=analysis_text, tags=["analysis"])

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitch heatmap for {name_a}" if image_path else "",
            tags=["arsenal_vs", name_a, name_b],
            reply=reply_content,
        )
