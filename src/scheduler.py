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

log = logging.getLogger(__name__)

HISTORY_PATH = DATA_DIR / "post_history.json"

# Weekly rotation: day-of-week → (screenshot_class, text_class)
# Monday=0 … Sunday=6
SCHEDULE: dict[int, tuple[type[ContentGenerator], type[ContentGenerator]]] = {
    0: (PitchingSummaryScreenshot, HardestPitchGenerator),       # Mon
    1: (StatcastCardsScreenshot, PitcherSpotlightGenerator),     # Tue
    2: (PitchPlotsScreenshot, StatOfDayGenerator),               # Wed
    3: (LeaderboardScreenshot, HardestPitchGenerator),           # Thu
    4: (PitchingSummaryScreenshot, PitcherSpotlightGenerator),   # Fri
    5: (StatcastCardsScreenshot, StatOfDayGenerator),            # Sat
    6: (HeatMapsScreenshot, PitcherSpotlightGenerator),         # Sun
}


def get_generators_for_today() -> tuple[ContentGenerator, ContentGenerator]:
    """Return (screenshot_generator, text_generator) for today's schedule."""
    dow = date.today().weekday()
    ss_cls, txt_cls = SCHEDULE[dow]
    return ss_cls(), txt_cls()


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
