"""Generate AI spark-notes analysis of a pitcher's stats using Claude."""

from __future__ import annotations

import logging
import os

import anthropic

from .config import MLB_SEASON

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — skipping AI analysis")
        return None
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=key)
    return _client


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
