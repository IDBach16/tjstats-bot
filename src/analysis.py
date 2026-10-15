"""Generate AI spark-notes analysis of a pitcher's stats using Claude."""

from __future__ import annotations

import logging
import os

import anthropic
import pandas as pd

from .config import MLB_SEASON
from .pitching_lens import analyst_request, reply_text, separations, whiff_tag

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
    # Rates arrive as 0-1 fractions (Pitch Profiler and savant_pitching both);
    # fmt None means "show as a percentage".
    for col, label, fmt in [
        ("innings_pitched", "IP", ".1f"),
        ("era", "ERA", ".2f"),
        ("fip", "FIP", ".2f"),
        ("strike_out_percentage", "K%", None),
        ("walk_percentage", "BB%", None),
        ("whiff_rate", "Whiff%", None),
        ("chase_percentage", "Chase%", None),
        ("zone_rate", "Zone%", None),
        ("barrel_rate", "Barrel%", None),
        ("hard_hit_rate", "Hard-hit%", None),
        ("avg_exit_velo", "Avg EV", ".1f"),
        ("stuff_plus", "Stuff+", ".0f"),
        ("pitching_plus", "Pitching+", ".0f"),
    ]:
        if col in p.index:
            try:
                val = float(p[col])
                if val != val:          # NaN: a column this source doesn't fill
                    continue
                if fmt:
                    season_stats[label] = format(val, fmt)
                else:
                    season_stats[label] = f"{val * 100:.1f}%"
            except (TypeError, ValueError):
                pass

    if not season_stats:
        return None

    # K-BB% is the lens's headline skill stat; computed here, not by the model.
    try:
        kbb = (float(p["strike_out_percentage"]) - float(p["walk_percentage"])) * 100
        if kbb == kbb:
            season_stats["K-BB%"] = f"{kbb:.1f}%"
    except (KeyError, TypeError, ValueError):
        pass
    hand = str(p.get("p_throws", "") or "").upper()
    if hand in ("R", "L"):
        season_stats["Throws"] = hand

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
                        ("ivb", "ivb"),
                        ("hb", "hb"),
                        ("spin_rate", "spin"),
                        ("whiff_rate", "whiff_rate"),
                        ("put_away", "put_away"),
                        ("est_woba", "xwoba"),
                        ("stuff_plus", "stuff_plus"),
                        ("run_value_per_100_pitches", "run_value"),
                    ]:
                        if attr in row.index:
                            try:
                                v = float(row[attr])
                                if v != v:      # NaN
                                    continue
                                if key in ("whiff_rate", "put_away") and v <= 1:
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
        if p.get("ivb") is not None:
            parts.append(f"IVB {p['ivb']:.1f} in")
        if p.get("hb") is not None:
            parts.append(f"HB {p['hb']:.1f} in")
        if p.get("spin") is not None:
            parts.append(f"{p['spin']:.0f} rpm")
        if p.get("whiff_rate") is not None:
            tag = whiff_tag(str(p.get("pitch_name")), p["whiff_rate"])
            parts.append(f"Whiff: {p['whiff_rate']:.1f}%" + (f" ({tag})" if tag else ""))
        if p.get("put_away") is not None:
            parts.append(f"PutAway: {p['put_away']:.1f}%")
        if p.get("xwoba") is not None:
            parts.append(f"xwOBA: {p['xwoba']:.3f}")
        if p.get("stuff_plus") is not None:
            parts.append(f"Stuff+: {p['stuff_plus']:.0f}")
        if p.get("run_value") is not None:
            parts.append(f"RV/100: {p['run_value']:.1f}")
        pitch_lines.append(" | ".join(parts))

    stats_text = "\n".join(f"  {k}: {v}" for k, v in season_stats.items())
    arsenal_text = "\n".join(f"  - {line}" for line in pitch_lines)
    seps = separations(pitch_stats)
    sep_text = ("\n\nSeparation from his primary fastball:\n"
                + "\n".join(f"  - {line}" for line in seps)) if seps else ""

    prompt = f"""You are a sharp baseball analyst writing for Twitter/X. Given these {MLB_SEASON} stats for {name}, write a 2-3 sentence "spark notes" analysis.

Season Stats:
{stats_text}

Arsenal (HB is pitcher's view: positive = arm side for a RHP, glove side for a LHP):
{arsenal_text}{sep_text}

Not in this data: release height, extension, arm angle, approach angles, location and age. Make no claim about any of them.

Rules:
- Read him through the framework in your instructions and lead with the one thing that makes him interesting (dominant, underrated, improved, concerning, etc.), saying why
- Do NOT use the word "elite" — find more creative ways to describe top performance
- Reference specific stats to back up your take
- Sound like a knowledgeable baseball person, not a robot
- Keep it UNDER 240 characters total: two tight sentences, one idea
- Do NOT use hashtags, emojis, or @ mentions
- Do NOT use dashes or hyphens (use commas, periods, or other punctuation instead)
- Do NOT start with the pitcher's name (the tweet already has it)
- Do NOT include character counts or meta commentary about the text length
- Output ONLY the analysis sentences, nothing else"""

    import re

    def _ask(messages: list[dict]) -> str:
        message = client.messages.create(**analyst_request(), messages=messages)
        out = reply_text(message)
        # Strip any trailing character count like "(259 characters)"
        out = re.sub(r'\s*\(\d+\s*characters?\)\s*$', '', out)
        return re.sub(r'\s*\d+\s*characters?\s*$', '', out)

    try:
        messages = [{"role": "user", "content": prompt}]
        text = _ask(messages)
        # The richer data invites longer takes, and a hard cut ends the tweet
        # mid-word. One rewrite pass first; the cut stays as the last resort.
        if len(text) > 270:
            text = _ask(messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That is {len(text)} characters. Rewrite the same take "
                                            "in under 240 characters. Output only the sentences."},
            ]) or text
        if len(text) > 270:
            text = text[:267] + "..."
        log.info("Generated analysis (%d chars): %s", len(text), text[:80])
        return text
    except Exception:
        log.warning("Claude analysis generation failed", exc_info=True)
        return None
