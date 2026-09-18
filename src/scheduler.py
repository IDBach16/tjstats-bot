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
from .content.best_outing import BestOutingGenerator
from .content.swing_plus_top10 import SwingPlusTop10Generator
from .content.swing_plus_young import SwingPlusYoungGenerator
from .content.best_pitch_week import BestPitchWeekGenerator
from .content.txt_stat_of_day import StatOfDayGenerator
from .content.txt_guess_pitcher import GuessThePitcherGenerator
from .content.txt_explainer import ExplainerGenerator
from .content.txt_arsenal_vs import ArsenalVsGenerator
from .content.txt_undervalued_relievers import UndervaluedRelieverGenerator

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

# Hitter analysis
from .content.hitter_analysis import HitterAnalysisGenerator

# Biomechanics educational content
from .content.biomechanics_101 import BiomechanicsGenerator

# Season summary
from .content.season_summary import SeasonSummaryGenerator

# College (MLB Draft) prospect cards — NCAA D1 Statcast
from .content.draft_prospect import DraftProspectGenerator

# Daily generators (run every day regardless of rotation)
from .content.reds_summary import RedsSummaryGenerator

# Newsroom — multi-agent player-analysis threads (Savant/FanGraphs data)
from .content.newsroom import NewsroomGenerator

log = logging.getLogger(__name__)

HISTORY_PATH = DATA_DIR / "post_history.json"

# How many posts to keep. THIS IS THE REAL CEILING ON DE-DUPLICATION -- a
# generator cannot avoid repeating someone it can no longer see.
#
# It was 200. At the old ~3.4 posts/day that was 58 days of memory, which is how
# Mookie Betts got posted seven times between July and September: the pool had
# hundreds of hitters in it, but the bot could only remember the last two months.
#
# Sizing it: at 2 posts/day, 1000 entries is ~500 days. The pools are 245
# qualified hitters and ~186 unposted qualified pitchers, so each generator can
# work all the way through its field -- twice over -- before the oldest entry
# falls off. Cost is about 225 KB of JSON, committed by the workflow each run.
#
# If the schedule speeds up again, raise this: the number that matters is
# ENTRIES / POSTS-PER-DAY >= the days it takes to exhaust the pool.
HISTORY_LIMIT = 1000

# Registry of all generators by name (for --generator CLI flag)
GENERATORS: dict[str, type[ContentGenerator]] = {
    "pitcher_spotlight": PitcherSpotlightGenerator,
    "stat_of_day": StatOfDayGenerator,
    "hardest_pitch": HardestPitchGenerator,
    "guess_pitcher": GuessThePitcherGenerator,
    "explainer": ExplainerGenerator,
    "arsenal_vs": ArsenalVsGenerator,
    "undervalued_relievers": UndervaluedRelieverGenerator,
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
    "best_outing": BestOutingGenerator,
    "swing_plus_top10": SwingPlusTop10Generator,
    "swing_plus_young": SwingPlusYoungGenerator,
    "best_pitch_week": BestPitchWeekGenerator,
    "season_summary": SeasonSummaryGenerator,
    "hitter_analysis": HitterAnalysisGenerator,
    "newsroom": NewsroomGenerator,
    "draft_prospect": DraftProspectGenerator,
}

# Daily generators — these run every day in addition to the rotation schedule.
# EMPTY since 2026-09-18: the Reds summary came out when their season ended.
# RedsSummaryGenerator stays in GENERATORS for manual --generator runs, so it can
# be put back by adding it here and restoring the 14:00 UTC cron.
DAILY_GENERATORS: list[type[ContentGenerator]] = []

# Daily lineup — cut to 2 posts/day on 2026-09-18 (Ian):
#   1. Pitching Summary — 'screenshot' slot (gens[0]), every day
#   2. Hitter Analysis  — 'text' slot (gens[1]), every day
#
# What came out, and why, so none of it looks like an accident:
#   * Reds Summary  — their season ended.
#   * Pitcher Card  — the screenshot slot used to alternate the season Pitching
#                     Summary (Mon/Wed/Fri/Sun) with the single-game Pitcher Card
#                     (Tue/Thu/Sat). Ian wants the season card every day.
#   * Newsroom      — both daily BachTalk threads, 18:00 and 21:00 UTC.
#
# Every one of them stays in GENERATORS for manual --generator runs; nothing was
# deleted. The workflow's crons were cut to match (14:00 / 18:00 / 21:00 removed),
# because a cron with nothing to run still spends an Actions minute.
# Monday=0 … Sunday=6.
SCHEDULE: dict[int, tuple[type[ContentGenerator], ...]] = {
    d: (PitchingSummaryGenerator, HitterAnalysisGenerator) for d in range(7)
}


def get_generators_for_today() -> list[ContentGenerator]:
    """Return today's rotation generators: Pitching Summary + Hitter Analysis."""
    dow = date.today().weekday()
    return [cls() for cls in SCHEDULE[dow]]


def get_daily_generators() -> list[ContentGenerator]:
    """Daily generators, if any. Empty since the Reds summary came out."""
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
    history["posts"] = history["posts"][-HISTORY_LIMIT:]
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


def recent_generator_tags(
    generator_name: str, index: int = 1, lookback: int = HISTORY_LIMIT
) -> set[str]:
    """Return the tag values at position ``index`` for recent posts from a
    given generator — e.g. every featured player name (tags[1]) the
    ``draft_prospect`` generator has already posted, for cross-run
    de-duplication so it doesn't repeat the same prospect."""
    history = _load_history()
    recent = history.get("posts", [])[-lookback:]
    out: set[str] = set()
    for entry in recent:
        if entry.get("generator") != generator_name:
            continue
        tags = entry.get("tags", [])
        if len(tags) > index and tags[index]:
            out.add(tags[index])
    return out
