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

# Card generators — MiLB (AAA Statcast)
from .content.milb_pitcher_card import MiLBPitcherCardGenerator
from .content.milb_pitching_summary import MiLBPitchingSummaryGenerator

# Card generators — MiLB Traditional (AA, A+, A, Complex)
from .content.milb_trad_pitcher_card import MiLBTradPitcherCardGenerator
from .content.milb_trad_pitching_summary import MiLBTradPitchingSummaryGenerator

# Biomechanics educational content
from .content.biomechanics_101 import BiomechanicsGenerator

# Daily generators (run every day regardless of rotation)
from .content.reds_summary import RedsSummaryGenerator

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
    "milb_trad_pitcher_card": MiLBTradPitcherCardGenerator,
    "milb_trad_pitching_summary": MiLBTradPitchingSummaryGenerator,
    "biomechanics_101": BiomechanicsGenerator,
    "reds_summary": RedsSummaryGenerator,
}

# Daily generators — these run every day in addition to the rotation schedule
DAILY_GENERATORS: list[type[ContentGenerator]] = [
    RedsSummaryGenerator,
]

# Weekly rotation: day-of-week → (morning, afternoon, evening, *optional_biomech)
# Monday=0 … Sunday=6
# Alternates: MLB (Mon/Wed/Fri/Sun), MiLB AAA Statcast (Tue/Sat),
#             MiLB Traditional AA/A+ (Thu)
# Biomechanics 101 posts daily (4th slot every day)
SCHEDULE: dict[int, tuple[type[ContentGenerator], ...]] = {
    0: (PitcherCardGenerator, GuessThePitcherGenerator, PitchingSummaryGenerator, BiomechanicsGenerator),   # Mon — MLB + Biomech
    1: (MiLBPitcherCardGenerator, PitcherSpotlightGenerator, MiLBPitchingSummaryGenerator, BiomechanicsGenerator),  # Tue — MiLB AAA + Biomech
    2: (VeloDistributionGenerator, ExplainerGenerator, PitchingSummaryGenerator, BiomechanicsGenerator),     # Wed — MLB + Biomech
    3: (MiLBTradPitcherCardGenerator, StatOfDayGenerator, MiLBTradPitchingSummaryGenerator, BiomechanicsGenerator), # Thu — MiLB AA/A+ + Biomech
    4: (ReleasePointGenerator, ArsenalVsGenerator, PitchingSummaryGenerator, BiomechanicsGenerator),         # Fri — MLB + Biomech
    5: (MiLBPitcherCardGenerator, HardestPitchGenerator, MiLBPitchingSummaryGenerator, BiomechanicsGenerator),      # Sat — MiLB AAA + Biomech
    6: (MovementProfileGenerator, PitcherSpotlightGenerator, PitchingSummaryGenerator, BiomechanicsGenerator),       # Sun — MLB + Biomech
}


def get_generators_for_today() -> list[ContentGenerator]:
    """Return generators for today's schedule (3 or 4 depending on day)."""
    dow = date.today().weekday()
    return [cls() for cls in SCHEDULE[dow]]


def get_daily_generators() -> list[ContentGenerator]:
    """Return daily generators that run every day (e.g. Reds summary)."""
    return [cls() for cls in DAILY_GENERATORS]


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
