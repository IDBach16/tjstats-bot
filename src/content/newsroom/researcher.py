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
            f"Blast rate: {_fmt(f.get('blast_per_swing'))}%   |  squared-up rate: {_fmt(f.get('squared_up_per_swing'))}%",
            f"Hard-swing rate: {_fmt(f.get('hard_swing_rate'))}%   |  swing length: {_fmt(f.get('swing_length'))} ft",
        ]
    return []


def build_fact_sheet(lead: Lead) -> dict:
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
