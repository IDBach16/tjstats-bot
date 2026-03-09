"""Generate AI spark-notes analysis of a pitcher's stats using Claude."""

from __future__ import annotations

import logging
import os

import anthropic
import pandas as pd

from .config import MLB_SEASON

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

# Noise pitch types to skip
_NOISE = {"PO", "IN", "EP", "AB", "FA", "UN", "SC"}


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — skipping AI analysis")
        return None
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=key)
    return _client


def analyze_pitcher(
    name: str,
    season_df: pd.DataFrame,
    pitches_df: pd.DataFrame | None = None,
) -> str | None:
    """High-level helper: extract stats from DataFrames and generate analysis.

    This is the convenience function most generators should call.
    """
    # Find the pitcher row
    name_col = None
    for c in ("pitcher_name", "player_name", "name"):
        if c in season_df.columns:
            name_col = c
            break
    if not name_col:
        return None

    matches = season_df[season_df[name_col] == name]
    if matches.empty:
        return None

    p = matches.iloc[0]

    # Extract season stats
    season_stats: dict[str, str] = {}
    for col, label, fmt in [
        ("era", "ERA", ".2f"),
        ("fip", "FIP", ".2f"),
        ("strike_out_percentage", "K%", None),
        ("walk_percentage", "BB%", None),
        ("whiff_rate", "Whiff%", None),
        ("stuff_plus", "Stuff+", ".0f"),
        ("pitching_plus", "Pitching+", ".0f"),
        ("innings_pitched", "IP", ".1f"),
    ]:
        if col in p.index:
            try:
                val = float(p[col])
                if fmt:
                    season_stats[label] = format(val, fmt)
                else:
                    season_stats[label] = f"{val * 100:.1f}%"
            except (TypeError, ValueError):
                pass

    if not season_stats:
        return None

    # Extract per-pitch stats
    pitch_stats: list[dict] = []
    if pitches_df is not None and not pitches_df.empty:
        pitch_name_col = None
        for c in ("pitch_type", "pitch_name"):
            if c in pitches_df.columns:
                pitch_name_col = c
                break

        if pitch_name_col and name_col in pitches_df.columns:
            pitcher_pitches = pitches_df[pitches_df[name_col] == name]
            if not pitcher_pitches.empty:
                grouped = pitcher_pitches.groupby(pitch_name_col)
                total = pitcher_pitches["percentage_thrown"].sum() if "percentage_thrown" in pitcher_pitches.columns else 1
                for ptype, grp in grouped:
                    if str(ptype) in _NOISE:
                        continue
                    row = grp.iloc[0]
                    ps: dict = {"pitch_name": str(ptype)}
                    if "percentage_thrown" in grp.columns:
                        ps["usage"] = round(grp["percentage_thrown"].sum() / total * 100, 1)
                    for attr, key in [
                        ("velocity", "velocity"),
                        ("whiff_rate", "whiff_rate"),
                        ("stuff_plus", "stuff_plus"),
                        ("run_value_per_100_pitches", "run_value"),
                    ]:
                        if attr in row.index:
                            try:
                                v = float(row[attr])
                                if key == "whiff_rate" and v < 1:
                                    v *= 100
                                ps[key] = v
                            except (TypeError, ValueError):
                                pass
                    pitch_stats.append(ps)

    return generate_analysis(name, season_stats, pitch_stats)


def generate_analysis(
    name: str,
    season_stats: dict,
    pitch_stats: list[dict],
) -> str | None:
    """Generate a concise spark-notes analysis for a pitcher.

    Args:
        name: Pitcher's full name.
        season_stats: Dict of season-level stats (ERA, FIP, K%, etc.).
        pitch_stats: List of per-pitch-type stat dicts.

    Returns:
        Analysis string (under 270 chars) or None on failure.
    """
    client = _get_client()
    if not client:
        return None

    # Build the stats context
    pitch_lines = []
    for p in pitch_stats:
        parts = [f"{p.get('pitch_name', 'Unknown')} ({p.get('usage', '?')}%)"]
        if p.get("velocity"):
            parts.append(f"{p['velocity']:.1f} mph")
        if p.get("whiff_rate") is not None:
            parts.append(f"Whiff: {p['whiff_rate']:.1f}%")
        if p.get("stuff_plus") is not None:
            parts.append(f"Stuff+: {p['stuff_plus']:.0f}")
        if p.get("run_value") is not None:
            parts.append(f"RV/100: {p['run_value']:.1f}")
        pitch_lines.append(" | ".join(parts))

    stats_text = "\n".join(f"  {k}: {v}" for k, v in season_stats.items())
    arsenal_text = "\n".join(f"  - {line}" for line in pitch_lines)

    prompt = f"""You are a sharp baseball analyst writing for Twitter/X. Given these {MLB_SEASON} stats for {name}, write a 2-3 sentence "spark notes" analysis.

Season Stats:
{stats_text}

Arsenal:
{arsenal_text}

Rules:
- Be insightful — highlight what makes this pitcher interesting (elite, underrated, improved, concerning, etc.)
- Reference specific stats to back up your take
- Sound like a knowledgeable baseball person, not a robot
- Keep it UNDER 260 characters total (strict Twitter limit)
- Do NOT use hashtags, emojis, or @ mentions
- Do NOT start with the pitcher's name (the tweet already has it)"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        # Enforce character limit
        if len(text) > 270:
            text = text[:267] + "..."
        log.info("Generated analysis (%d chars): %s", len(text), text[:80])
        return text
    except Exception:
        log.warning("Claude analysis generation failed", exc_info=True)
        return None
