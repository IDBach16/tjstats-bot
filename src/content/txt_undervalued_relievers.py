"""Text generator: Undervalued Relievers — spotlight low-IP arms with strong peripherals."""

from __future__ import annotations

import logging
import random

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, get_name
from .. import pitch_profiler
from ..config import DEFAULT_HASHTAGS, MLB_SEASON
from ..charts import plot_pitch_locations

log = logging.getLogger(__name__)

_HOOKS = [
    "This reliever deserves more attention.",
    "Under the radar arm alert.",
    "Sleeper reliever worth watching.",
    "This bullpen arm is quietly dominant.",
    "Don't sleep on this reliever.",
    "Hidden gem in the bullpen.",
    "This reliever's numbers are sneaky good.",
]

# Stats to show — tuned for reliever value
_RELIEVER_STATS = [
    ("era", "ERA", False),
    ("fip", "FIP", False),
    ("innings_pitched", "IP", False),
    ("strike_out_percentage", "K%", True),
    ("walk_percentage", "BB%", True),
    ("whiff_rate", "Whiff%", True),
    ("chase_percentage", "Chase%", True),
    ("stuff_plus", "Stuff+", False),
]


def _score_reliever(player) -> float:
    """Score a reliever by peripherals — higher = more undervalued.

    Weights stuff that makes a reliever quietly good: high K%, high whiff,
    low walk rate, good Stuff+, with a decent ERA/FIP.
    """
    score = 0.0
    try:
        # Whiff rate (0-1 scale, higher = better)
        whiff = float(player.get("whiff_rate", 0) or 0)
        score += whiff * 200  # big weight on whiff

        # K% (0-1 scale)
        k_pct = float(player.get("strike_out_percentage", 0) or 0)
        score += k_pct * 150

        # BB% (0-1 scale, lower = better)
        bb_pct = float(player.get("walk_percentage", 0.1) or 0.1)
        score -= bb_pct * 100

        # Stuff+ (100 = avg, higher = better)
        stuff = float(player.get("stuff_plus", 100) or 100)
        score += (stuff - 100) * 1.5

        # ERA bonus for good results
        era = float(player.get("era", 4.5) or 4.5)
        if era < 3.50:
            score += 10
        if era < 2.50:
            score += 10

        # FIP bonus
        fip = float(player.get("fip", 4.5) or 4.5)
        if fip < 3.50:
            score += 10

        # Chase% bonus (higher = pitchers getting chases)
        chase = float(player.get("chase_percentage", 0) or 0)
        score += chase * 50

    except (TypeError, ValueError):
        pass
    return score


class UndervaluedRelieverGenerator(ContentGenerator):
    name = "undervalued_relievers"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Filter to relievers: 20-70 IP, no starts (GS == 0)
        if "innings_pitched" not in df.columns:
            log.warning("No innings_pitched column in data")
            return PostContent(text="")

        ip_mask = (df["innings_pitched"] >= 20) & (df["innings_pitched"] <= 70)
        # Exclude starters: if games_started column exists, require GS == 0
        if "games_started" in df.columns:
            gs_col = pd.to_numeric(df["games_started"], errors="coerce").fillna(0)
            ip_mask = ip_mask & (gs_col == 0)

        relievers = df[ip_mask].copy()

        if relievers.empty:
            log.warning("No relievers found in 20-70 IP range")
            return PostContent(text="")

        log.info("Found %d relievers in 20-70 IP range", len(relievers))

        # Score and pick from the top undervalued arms
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in relievers.columns:
                name_col = c
                break
        if not name_col:
            return PostContent(text="")

        relievers["_score"] = relievers.apply(_score_reliever, axis=1)
        top = relievers.nlargest(15, "_score")

        # Pick randomly from the top 15 to keep it fresh
        player = top.sample(1).iloc[0]
        player_name = get_name(player)

        # Build stat block
        stat_lines = []
        for col, label, pct in _RELIEVER_STATS:
            if col in player.index:
                stat_lines.append(f"{label}: {fmt_stat(player[col], pct)}")

        paired = []
        for i in range(0, len(stat_lines), 2):
            paired.append(" | ".join(stat_lines[i:i + 2]))
        stat_block = "\n".join(paired)

        # Get team
        team = ""
        for tc in ("team", "team_abbreviation", "team_abbrev"):
            if tc in player.index and player[tc]:
                team = f" ({player[tc]})"
                break

        hook = random.choice(_HOOKS)
        text = (
            f"{hook}\n\n"
            f"{player_name}{team}\n"
            f"{stat_block}\n\n"
            f"@TJStats {DEFAULT_HASHTAGS}"
        )

        # Pitch location chart
        image_path = None
        pid = None
        for id_col in ("pitcher_id", "player_id", "mlbam_id"):
            if id_col in player.index:
                try:
                    pid = int(player[id_col])
                    break
                except (TypeError, ValueError):
                    pass
        if pid:
            image_path = plot_pitch_locations(pid, player_name)

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Pitch location chart for {player_name}" if image_path else "",
            tags=["undervalued_relievers", player_name],
        )
