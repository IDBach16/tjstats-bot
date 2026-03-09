"""Text generator: elite pitcher spotlight using Pitch Profiler data."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, safe_stat, build_stat_block
from .. import pitch_profiler
from ..analysis import analyze_pitcher
from ..config import DATA_DIR, DEFAULT_HASHTAGS
from ..charts import plot_pitch_movement
from ..video_clips import get_pitcher_clip

log = logging.getLogger(__name__)

# Aliases for backward compat within this module
_fmt = fmt_stat
_get_stat = safe_stat


def _load_watchlist() -> list[dict]:
    path = DATA_DIR / "players.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("pitchers", [])
    return []


# Each narrative template: (test_func, hook_template, featured_stats)
# test_func(player) -> bool, hook uses {name} and stat placeholders
_NARRATIVES = [
    {
        "id": "elite_whiff",
        "test": lambda p: (_get_stat(p, "whiff_rate", True) or 0) >= 30,
        "hooks": [
            "{name} is making hitters look foolish — {whiff_rate} Whiff% this season.",
            "{name}: Swing and miss machine. {whiff_rate}% whiff rate.",
            "Good luck making contact off {name}. {whiff_rate}% of swings are misses.",
        ],
        "stats": [("era", "ERA", False), ("strike_out_percentage", "K%", True), ("stuff_plus", "Stuff+", False)],
    },
    {
        "id": "elite_k",
        "test": lambda p: (_get_stat(p, "strike_out_percentage", True) or 0) >= 28,
        "hooks": [
            "{name} is punching out batters at an absurd clip — {k_pct} K%.",
            "{name}: {k_pct}% strikeout rate. Hitters are overmatched.",
            "The strikeout numbers for {name} are ridiculous. {k_pct} K%.",
        ],
        "stats": [("era", "ERA", False), ("whiff_rate", "Whiff%", True), ("chase_percentage", "Chase%", True)],
    },
    {
        "id": "elite_era",
        "test": lambda p: (_get_stat(p, "era") or 99) <= 2.80,
        "hooks": [
            "{name} is dealing — {era} ERA across {ip} innings.",
            "{name}: {era} ERA. That's not a typo.",
            "A {era} ERA over {ip} IP. {name} has been untouchable.",
        ],
        "stats": [("strike_out_percentage", "K%", True), ("whiff_rate", "Whiff%", True), ("fip", "FIP", False)],
    },
    {
        "id": "elite_stuff",
        "test": lambda p: (_get_stat(p, "stuff_plus") or 0) >= 115,
        "hooks": [
            "{name}'s stuff is elite — {stuff_plus} Stuff+ this season.",
            "The raw stuff from {name} is off the charts. {stuff_plus} Stuff+.",
            "{name}: {stuff_plus} Stuff+. The arsenal is legit.",
        ],
        "stats": [("era", "ERA", False), ("whiff_rate", "Whiff%", True), ("strike_out_percentage", "K%", True)],
    },
    {
        "id": "command_king",
        "test": lambda p: (_get_stat(p, "walk_percentage", True) or 99) <= 5.0,
        "hooks": [
            "{name} doesn't give anything away — just {bb_pct} BB%.",
            "Pinpoint command from {name}. {bb_pct}% walk rate this season.",
            "{name}: {bb_pct} BB%. Hitters have to earn everything.",
        ],
        "stats": [("era", "ERA", False), ("strike_out_percentage", "K%", True), ("chase_percentage", "Chase%", True)],
    },
]

# Fallback if no narrative matches
_FALLBACK_HOOKS = [
    "{name} has been quietly impressive this season.",
    "Keep an eye on {name} — the numbers are legit.",
    "{name}: The numbers speak for themselves.",
]


def _build_creative_text(player, name: str) -> str:
    """Build a short, creative spotlight tweet based on the pitcher's standout stats."""
    # Collect formatted stat values for template substitution
    stat_vals = {
        "name": name,
        "era": _fmt(player.get("era", 0), False) if "era" in player.index else "",
        "ip": _fmt(player.get("innings_pitched", 0), False) if "innings_pitched" in player.index else "",
        "whiff_rate": _fmt(player.get("whiff_rate", 0), True) if "whiff_rate" in player.index else "",
        "k_pct": _fmt(player.get("strike_out_percentage", 0), True) if "strike_out_percentage" in player.index else "",
        "bb_pct": _fmt(player.get("walk_percentage", 0), True) if "walk_percentage" in player.index else "",
        "chase_pct": _fmt(player.get("chase_percentage", 0), True) if "chase_percentage" in player.index else "",
        "stuff_plus": _fmt(player.get("stuff_plus", 0), False) if "stuff_plus" in player.index else "",
    }

    # Find the first matching narrative
    chosen = None
    for narrative in _NARRATIVES:
        try:
            if narrative["test"](player):
                chosen = narrative
                break
        except Exception:
            continue

    if chosen:
        hook = random.choice(chosen["hooks"]).format(**stat_vals)
    else:
        hook = random.choice(_FALLBACK_HOOKS).format(**stat_vals)

    stat_block = build_stat_block(player)

    text = f"{hook}\n\n{stat_block}\n\nData via @mlbpitchprofiler {DEFAULT_HASHTAGS}"
    return text


class PitcherSpotlightGenerator(ContentGenerator):
    name = "pitcher_spotlight"

    async def generate(self) -> PostContent:
        df = pitch_profiler.get_season_pitchers()
        if df.empty:
            return PostContent(text="")

        # Try to spotlight a watchlist player first, else pick a top performer
        watchlist = _load_watchlist()
        watchlist_ids = {p["id"] for p in watchlist}

        if watchlist_ids and "pitcher_id" in df.columns:
            candidates = df[df["pitcher_id"].isin(watchlist_ids)]
        else:
            candidates = df

        if candidates.empty:
            candidates = df

        # Filter to pitchers with meaningful innings
        if "innings_pitched" in candidates.columns:
            qualified = candidates[candidates["innings_pitched"] >= 50]
            if not qualified.empty:
                candidates = qualified

        # Pick a strong performer — sort by whiff_rate or strike_out_percentage
        sort_col = None
        for col in ("whiff_rate", "strike_out_percentage"):
            if col in candidates.columns:
                sort_col = col
                break

        if sort_col:
            top = candidates.nlargest(20, sort_col)
        else:
            top = candidates.head(20)

        player = top.sample(1).iloc[0]
        name = str(player.get("pitcher_name", player.get("player_name", "Unknown")))

        text = _build_creative_text(player, name)

        # Media: try chart + video (image as main tweet, video as reply)
        image_path = None
        video_path = None
        pitcher_id = player.get("pitcher_id", player.get("player_id"))
        if pitcher_id:
            image_path = plot_pitch_movement(int(pitcher_id), name)
            try:
                video_path = get_pitcher_clip(int(pitcher_id), name)
            except Exception:
                log.warning("Video clip fetch failed for %s", name, exc_info=True)

        # AI analysis
        analysis_text = analyze_pitcher(name, df, pitch_profiler.get_season_pitches())
        reply_content = None
        if analysis_text:
            reply_content = PostContent(text=analysis_text, tags=["analysis"])

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitch movement chart for {name}" if image_path else "",
            tags=["pitcher_spotlight", name],
            reply=reply_content,
        )
