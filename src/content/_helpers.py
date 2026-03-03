"""Shared stat formatting helpers used by multiple content generators."""

from __future__ import annotations


def fmt_stat(val, pct: bool = False) -> str:
    """Format a stat value for display.

    If *pct* is True the raw value is a 0-1 decimal and will be shown as ×100.
    """
    if isinstance(val, float):
        return f"{val * 100:.1f}" if pct else f"{val:.2f}" if val < 10 else f"{val:.1f}"
    return str(val)


def safe_stat(player, col: str, pct: bool = False) -> float | None:
    """Safely extract a numeric stat from a pandas row (Series)."""
    if col not in player.index:
        return None
    val = player[col]
    try:
        return float(val) * 100 if pct else float(val)
    except (TypeError, ValueError):
        return None


def get_name(player) -> str:
    """Extract pitcher display name from a DataFrame row."""
    for col in ("pitcher_name", "player_name", "name"):
        if col in player.index:
            name = player[col]
            if name:
                return str(name)
    return "Unknown"


# Standard stat block definition: (column, label, is_percentage)
STAT_BLOCK = [
    ("era", "ERA", False),
    ("fip", "FIP", False),
    ("innings_pitched", "IP", False),
    ("strike_out_percentage", "K%", True),
    ("walk_percentage", "BB%", True),
    ("whiff_rate", "Whiff%", True),
    ("chase_percentage", "Chase%", True),
    ("strikeouts_per_9", "K/9", False),
    ("stuff_plus", "Stuff+", False),
    ("pitching_plus", "Pitching+", False),
]


def build_stat_block(player, stats: list[tuple[str, str, bool]] | None = None) -> str:
    """Build a pipe-separated stat block from a player row.

    Returns pairs of stats per line, e.g.:
        ERA: 2.45 | FIP: 2.78
        K%: 31.2 | BB%: 5.1
    """
    if stats is None:
        stats = STAT_BLOCK
    available = []
    for col, label, pct in stats:
        if col in player.index:
            available.append(f"{label}: {fmt_stat(player[col], pct)}")
    lines = []
    for i in range(0, len(available), 2):
        lines.append(" | ".join(available[i:i + 2]))
    return "\n".join(lines)
