"""Content rotation scheduler + post history tracking."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from .config import DATA_DIR
from .content.base import ContentGenerator

# Screenshot generators
from .content.ss_pitching import PitchingSummaryScreenshot
from .content.ss_statcast import StatcastCardsScreenshot
from .content.ss_pitch_plots import PitchPlotsScreenshot
from .content.ss_leaderboard import LeaderboardScreenshot
from .content.ss_heat_maps import HeatMapsScreenshot

# Text generators
from .content.txt_hardest_pitch import HardestPitchGenerator
from .content.txt_pitcher_spotlight import PitcherSpotlightGenerator
from .content.txt_stat_of_day import StatOfDayGenerator
from .content.txt_guess_pitcher import GuessThePitcherGenerator
from .content.txt_explainer import ExplainerGenerator
from .content.txt_arsenal_vs import ArsenalVsGenerator

# Screenshot generators (new)
from .content.ss_movement_profile import MovementProfileGenerator
from .content.ss_release_points import ReleasePointGenerator
from .content.ss_velo_distribution import VeloDistributionGenerator
from .content.ss_arsenal_usage import ArsenalUsageGenerator

# Card generators — MLB
from .content.pitcher_card import PitcherCardGenerator
from .content.pitching_summary import PitchingSummaryGenerator

# Card generators — MiLB
from .content.milb_pitcher_card import MiLBPitcherCardGenerator
from .content.milb_pitching_summary import MiLBPitchingSummaryGenerator

log = logging.getLogger(__name__)

HISTORY_PATH = DATA_DIR / "post_history.json"

# Registry of all generators by name (for --generator CLI flag)
GENERATORS: dict[str, type[ContentGenerator]] = {
    "pitcher_spotlight": PitcherSpotlightGenerator,
    "stat_of_day": StatOfDayGenerator,
    "hardest_pitch": HardestPitchGenerator,
    "guess_pitcher": GuessThePitcherGenerator,
    "explainer": ExplainerGenerator,
    "arsenal_vs": ArsenalVsGenerator,
    "movement_profile": MovementProfileGenerator,
    "ss_pitching_summary": PitchingSummaryScreenshot,
    "ss_statcast_cards": StatcastCardsScreenshot,
    "ss_pitch_plots": PitchPlotsScreenshot,
    "ss_leaderboard": LeaderboardScreenshot,
    "ss_heat_maps": HeatMapsScreenshot,
    "release_points": ReleasePointGenerator,
    "velo_distribution": VeloDistributionGenerator,
    "arsenal_usage": ArsenalUsageGenerator,
    "pitcher_card": PitcherCardGenerator,
    "pitching_summary": PitchingSummaryGenerator,
    "milb_pitcher_card": MiLBPitcherCardGenerator,
    "milb_pitching_summary": MiLBPitchingSummaryGenerator,
}

# Weekly rotation: day-of-week → (morning_visual, afternoon_text, evening_summary)
# Monday=0 … Sunday=6
# Alternates: MLB days (Mon/Wed/Fri/Sun) vs MiLB days (Tue/Thu/Sat)
SCHEDULE: dict[int, tuple[type[ContentGenerator], type[ContentGenerator], type[ContentGenerator]]] = {
    0: (PitcherCardGenerator, GuessThePitcherGenerator, PitchingSummaryGenerator),              # Mon — MLB
    1: (MiLBPitcherCardGenerator, PitcherSpotlightGenerator, MiLBPitchingSummaryGenerator),     # Tue — MiLB
    2: (VeloDistributionGenerator, ExplainerGenerator, PitchingSummaryGenerator),                # Wed — MLB
    3: (MiLBPitcherCardGenerator, StatOfDayGenerator, MiLBPitchingSummaryGenerator),             # Thu — MiLB
    4: (ReleasePointGenerator, ArsenalVsGenerator, PitchingSummaryGenerator),                    # Fri — MLB
    5: (MiLBPitcherCardGenerator, HardestPitchGenerator, MiLBPitchingSummaryGenerator),          # Sat — MiLB
    6: (MovementProfileGenerator, PitcherSpotlightGenerator, PitchingSummaryGenerator),          # Sun — MLB
}


def get_generators_for_today() -> tuple[ContentGenerator, ContentGenerator, ContentGenerator]:
    """Return (morning, afternoon, evening) generators for today's schedule."""
    dow = date.today().weekday()
    morning_cls, afternoon_cls, evening_cls = SCHEDULE[dow]
    return morning_cls(), afternoon_cls(), evening_cls()


# ── Post history ──────────────────────────────────────────────────────

def _load_history() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return {"posts": []}


def _save_history(data: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(data, indent=2))


def record_post(
    generator_name: str, tweet_id: str, tags: list[str]
) -> None:
    """Append a post entry to the history file."""
    history = _load_history()
    history["posts"].append({
        "date": datetime.utcnow().isoformat(),
        "generator": generator_name,
        "tweet_id": tweet_id,
        "tags": tags,
    })
    # Keep last 200 entries
    history["posts"] = history["posts"][-200:]
    _save_history(history)
    log.info("Recorded post %s from %s", tweet_id, generator_name)


def was_recently_posted(tag: str, lookback: int = 7) -> bool:
    """Check if a tag appeared in the last `lookback` posts."""
    history = _load_history()
    recent = history.get("posts", [])[-lookback:]
    for entry in recent:
        if tag in entry.get("tags", []):
            return True
    return False
