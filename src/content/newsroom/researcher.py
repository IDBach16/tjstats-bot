"""Beat reporter: turn a Lead into a clean, human-readable FACT SHEET.

The columnist only ever sees this sheet, so every number here is real and
pre-verified. `allowed_numbers` is the set the copy desk (Phase 2) will check the
draft against.
"""

from __future__ import annotations

import re

from .feeds import Lead


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        # 3-decimal rate stats (woba/ba/slg) keep leading dot; others trim
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:g}"
    return str(v)


def _pct(v) -> str:
    """Format a 0-1 rate as a percent value, e.g. 0.204 -> '20.4' (the line
    template appends the trailing % sign). Plain _fmt renders 0.900 as '.900',
    which then reads as 0.9%, not 90% — and the copy desk rejects the mismatch."""
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 100:.1f}"
    except (TypeError, ValueError):
        return str(v)


def _sheet_lines(lead: Lead) -> list[str]:
    f = lead.facts
    k = lead.kind
    if k in ("overperformer", "underperformer"):
        return [
            f"wOBA: {_fmt(f.get('woba'))}   (expected xwOBA: {_fmt(f.get('est_woba'))})",
            f"Gap (xwOBA - wOBA): {_fmt(f.get('gap'))}   |  league avg wOBA: {_fmt(f.get('league_woba'))}",
            f"AVG {_fmt(f.get('ba'))} vs expected {_fmt(f.get('est_ba'))}",
            f"SLG {_fmt(f.get('slg'))} vs expected {_fmt(f.get('est_slg'))}",
            f"Plate appearances: {f.get('pa')}",
        ]
    if k == "nasty_pitch":
        return [
            f"Pitch: {f.get('pitch_name')}",
            f"Whiff rate: {_fmt(f.get('whiff_percent'))}%   (league avg for the pitch pool: {_fmt(f.get('league_whiff'))}%)",
            f"Strikeout rate on it: {_fmt(f.get('k_percent'))}%   |  put-away rate: {_fmt(f.get('put_away'))}%",
            f"Run value / 100: {_fmt(f.get('run_value_per_100'))}   |  opponent AVG: {_fmt(f.get('opp_ba'))}",
            f"Thrown {f.get('pitches')} times ({_fmt(f.get('pitch_usage'))}% of his pitches)",
        ]
    if k == "bat_speed":
        return [
            f"Average bat speed: {_fmt(f.get('avg_bat_speed'))} mph   (league avg: {_fmt(f.get('league_bat_speed'))} mph)",
            f"Blast rate: {_pct(f.get('blast_per_swing'))}%   |  squared-up rate: {_pct(f.get('squared_up_per_swing'))}%",
            f"Hard-swing rate: {_pct(f.get('hard_swing_rate'))}%   |  swing length: {_fmt(f.get('swing_length'))} ft",
        ]
    if k == "hard_hitter":
        return [
            f"Average exit velocity: {_fmt(f.get('avg_hit_speed'))} mph   (league avg: {_fmt(f.get('league_ev'))} mph)",
            f"Hardest hit: {_fmt(f.get('max_hit_speed'))} mph   |  longest batted ball: {_fmt(f.get('max_distance'))} ft",
            f"Barrel rate: {_fmt(f.get('brl_percent'))}%   ({f.get('barrels')} barrels)   |  95+ mph rate: {_fmt(f.get('ev95percent'))}%",
            f"Batted balls tracked: {f.get('attempts')}",
        ]
    if k == "flamethrower":
        lines = [f"Average four-seam fastball: {_fmt(f.get('ff_avg_speed'))} mph   "
                 f"(league-average four-seamer: {_fmt(f.get('league_ff'))} mph)"]
        if f.get("si_avg_speed"):
            lines.append(f"Sinker: {_fmt(f.get('si_avg_speed'))} mph")
        return lines
    if k == "pitcher_luck":
        return [
            f"ERA: {_fmt(f.get('era'))}   vs expected xERA: {_fmt(f.get('xera'))}   "
            f"({'ERA flattering him — regression likely' if f.get('lucky') else 'ERA has been unkind — due to improve'})",
            f"wOBA against: {_fmt(f.get('woba'))}   vs expected {_fmt(f.get('est_woba'))}",
            f"AVG against: {_fmt(f.get('ba'))}   vs expected {_fmt(f.get('est_ba'))}",
            f"Batters faced: {f.get('pa')}",
        ]
    return []


def _article_fact_sheet(lead: Lead) -> dict:
    """Fact sheet for an 'article' lead: an attributed summary to react to."""
    f = lead.facts
    sheet = (
        "REACTING TO A PUBLISHED ARTICLE — write your own take, do NOT copy its text.\n"
        f"OUTLET: {f.get('outlet')}\n"
        f"AUTHOR: {f.get('author') or 'staff'}\n"
        f"HEADLINE: {f.get('title')}\n"
        f"SUMMARY: {f.get('summary')}"
    )
    # numbers that actually appear in the summary — the copy desk holds the
    # writer to these so it can't invent stats the article never stated.
    allowed = set(re.findall(r"\d+\.?\d*", str(f.get("summary", ""))))
    return {
        "subject": lead.subject,
        "kind": "article",
        "is_pitcher": False,
        "is_red": False,
        "sheet": sheet,
        "allowed_numbers": allowed,
        "article": f,
    }


def build_fact_sheet(lead: Lead) -> dict:
    if lead.kind == "article":
        return _article_fact_sheet(lead)
    role = "pitcher" if lead.is_pitcher else "hitter"
    rank_line = (f"Ranks #{lead.rank} of {lead.total} qualified {role}s in this category "
                 f"({MLB_TAG})") if lead.total else ""
    lines = _sheet_lines(lead)
    sheet = (
        f"SUBJECT: {lead.subject} ({role}){'  [Cincinnati Reds]' if lead.is_red else ''}\n"
        f"ANGLE: {lead.angle}\n"
        f"{rank_line}\n"
        f"SEASON STATS (Baseball Savant, {MLB_TAG}):\n- " + "\n- ".join(lines)
    )
    # numbers the writer is allowed to use verbatim (for the Phase-2 fact check)
    allowed = set()
    for v in lead.facts.values():
        allowed.update(re.findall(r"\d+\.?\d*", str(v)))
    allowed.update(str(x) for x in (lead.rank, lead.total))
    return {
        "subject": lead.subject,
        "kind": lead.kind,
        "is_pitcher": lead.is_pitcher,
        "is_red": lead.is_red,
        "sheet": sheet,
        "allowed_numbers": allowed,
    }


# Season label used in the sheet (kept here so it's easy to find/change).
from ...config import MLB_SEASON as _S  # noqa: E402
MLB_TAG = str(_S)
