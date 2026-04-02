"""Content generator: Season pitching summary card.

Picks a notable pitcher and generates a full season stats card using
Pitch Profiler data. Rotates through pitchers to avoid repeats.
Runs 3x per week (Tue/Thu/Sat).
"""

from __future__ import annotations

import logging
import os
import re

import anthropic
import pandas as pd

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)


def _generate_season_take(name: str, stats_str: str) -> str | None:
    """Generate 2-sentence season analysis."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content":
                f"Write exactly 2 sentences analyzing {name}'s {MLB_SEASON} season so far. "
                f"Season stats: {stats_str}. "
                f"Be sharp and insightful like a baseball analyst. Reference specific stats. "
                f"No hashtags, emojis, @, dashes, hyphens, or character counts. "
                f"Do NOT start with the pitcher's name. "
                f"Output ONLY the 2 sentences."
            }],
        )
        text = message.content[0].text.strip()
        text = re.sub(r'\s*\(\d+ characters?\)\s*$', '', text)
        return text
    except Exception:
        log.warning("Season take generation failed", exc_info=True)
        return None


class SeasonSummaryGenerator(ContentGenerator):
    name = "season_summary"

    async def generate(self) -> PostContent:
        # Get season-level data
        try:
            season_df = pitch_profiler.get_season_pitchers(MLB_SEASON)
            pitches_df = pitch_profiler.get_season_pitches(MLB_SEASON)
        except Exception:
            log.warning("Failed to fetch Pitch Profiler season data", exc_info=True)
            return PostContent(text="")

        if season_df.empty:
            log.info("No season pitcher data available")
            return PostContent(text="")

        # Filter to qualified pitchers (enough innings/pitches)
        ip_col = None
        for c in ("innings_pitched", "ip", "total_innings"):
            if c in season_df.columns:
                ip_col = c
                break

        candidates = season_df.copy()
        if ip_col:
            candidates[ip_col] = pd.to_numeric(candidates[ip_col], errors="coerce")
            candidates = candidates[candidates[ip_col] >= 5].copy()

        if candidates.empty:
            candidates = season_df.copy()

        # Find name column
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in candidates.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        # Skip recently posted pitchers
        from ..scheduler import was_recently_posted
        candidates = candidates[
            ~candidates[name_col].apply(lambda n: was_recently_posted(n, lookback=12))
        ]

        if candidates.empty:
            log.info("All qualified pitchers were recently posted")
            return PostContent(text="")

        # Score pitchers by stuff+ and pitching+ (pick interesting ones)
        score_cols = []
        for sc in ("stuff_plus", "pitching_plus", "whiff_rate"):
            if sc in candidates.columns:
                candidates[sc] = pd.to_numeric(candidates[sc], errors="coerce")
                score_cols.append(sc)

        if score_cols:
            candidates["_score"] = candidates[score_cols].mean(axis=1)
            candidates = candidates.sort_values("_score", ascending=False)
        else:
            candidates = candidates.sample(frac=1)

        # Pick the top pitcher
        player = candidates.iloc[0]
        name = str(player[name_col])
        pid_col = None
        for c in ("pitcher_id", "player_id", "mlbam_id"):
            if c in player.index:
                pid_col = c
                break
        pid = int(player[pid_col]) if pid_col and pd.notna(player[pid_col]) else None

        # Get team
        team = None
        for tc in ("season_teams", "team", "team_abbreviation"):
            if tc in player.index and player[tc]:
                team = str(player[tc]).split(",")[0].strip().upper()
                break

        log.info("Season summary: %s (pid=%s, team=%s)", name, pid, team)

        # Generate card with season data
        image_path = plot_pitching_summary(
            name, season_df, pitches_df,
            team=team, player_id=pid, level="MLB",
        )
        if not image_path:
            log.warning("Card generation failed for %s", name)
            return PostContent(text="")

        # Build stats string for AI
        stats_parts = []
        for col, label in [
            ("innings_pitched", "IP"), ("era", "ERA"), ("whip", "WHIP"),
            ("strike_outs", "K"), ("walks", "BB"),
            ("strike_out_percentage", "K%"), ("walk_percentage", "BB%"),
            ("whiff_rate", "Whiff%"), ("stuff_plus", "Stuff+"),
            ("pitching_plus", "Pitching+"),
        ]:
            if col in player.index and pd.notna(player[col]):
                val = player[col]
                if "percentage" in col or col == "whiff_rate":
                    stats_parts.append(f"{float(val)*100:.1f}% {label}")
                elif isinstance(val, float):
                    stats_parts.append(f"{val:.2f} {label}" if val < 20 else f"{val:.0f} {label}")
                else:
                    stats_parts.append(f"{val} {label}")

        stats_str = ", ".join(stats_parts) if stats_parts else "season data"

        # Generate analysis
        analysis = _generate_season_take(name, stats_str)

        # Build tweet
        if analysis:
            text = (
                f"{analysis}\n\n"
                f"{name} — {MLB_SEASON} Season Summary\n\n"
                f"@TJStats @PitchProfiler {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name} — {MLB_SEASON} Season Summary\n"
                f"{stats_str}\n\n"
                f"@TJStats @PitchProfiler {DEFAULT_HASHTAGS}"
            )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Season pitching summary for {name}",
            tags=["season_summary", name],
        )
