"""Text generator: Educational Explainers — rotating stat concepts with real examples."""

from __future__ import annotations

import json
import logging
import random

from .base import ContentGenerator, PostContent
from ._helpers import fmt_stat, get_name
from .. import pitch_profiler
from ..config import DATA_DIR, DEFAULT_HASHTAGS
from ..charts import plot_movement_profile

log = logging.getLogger(__name__)

_HISTORY_PATH = DATA_DIR / "post_history.json"


def _was_recently_posted(tag: str, lookback: int = 7) -> bool:
    """Check post history without importing scheduler (avoids circular import)."""
    if not _HISTORY_PATH.exists():
        return False
    history = json.loads(_HISTORY_PATH.read_text())
    for entry in history.get("posts", [])[-lookback:]:
        if tag in entry.get("tags", []):
            return True
    return False

# Each topic: id, title, body text, optional stat_col for leaderboard example,
# pct flag, and whether lower is better.
_TOPICS = [
    {
        "id": "explainer_stuff_plus",
        "title": "What is Stuff+?",
        "body": (
            "Stuff+ measures pitch quality based on velocity, movement, spin, "
            "and release compared to league avg (100).\n\n"
            "A Stuff+ of 120 = 20% better stuff than average. It explains why "
            "a 92 mph fastball can grade higher than a 96."
        ),
        "stat_col": "stuff_plus",
        "stat_label": "Stuff+",
        "pct": False,
        "ascending": False,
    },
    {
        "id": "explainer_whiff_vs_chase",
        "title": "Whiff% vs Chase%: What's the difference?",
        "body": (
            "Whiff% = swings and misses / total swings. Measures how unhittable "
            "your stuff is.\n\n"
            "Chase% = swings at pitches outside the zone / pitches outside the zone. "
            "Measures how deceptive your approach is.\n\n"
            "Elite pitchers dominate both — they get chases AND whiffs."
        ),
        "stat_col": "whiff_rate",
        "stat_label": "Whiff%",
        "pct": True,
        "ascending": False,
    },
    {
        "id": "explainer_csw",
        "title": "CSW% — The one stat to rule them all?",
        "body": (
            "CSW% = Called Strikes + Whiffs / Total Pitches.\n\n"
            "It combines command (called strikes) with stuff (whiffs) into one number. "
            "League average is around 29%. Elite arms push 33%+."
        ),
        "stat_col": "called_strikes_plus_whiffs_percentage",
        "stat_label": "CSW%",
        "pct": True,
        "ascending": False,
    },
    {
        "id": "explainer_extension",
        "title": "Why extension matters",
        "body": (
            "Extension = how far toward home plate a pitcher releases the ball.\n\n"
            "More extension means the batter has less time to react, making a "
            "93 mph fastball play like 95+. It's one reason why tall pitchers "
            "with long strides have a built-in advantage."
        ),
        "stat_col": None,
        "stat_label": None,
        "pct": False,
        "ascending": False,
    },
    {
        "id": "explainer_ivb",
        "title": "IVB vs Total Break — what's the difference?",
        "body": (
            "Induced Vertical Break (IVB) = movement caused by spin alone, "
            "removing gravity.\n\n"
            "Total Break = the actual path the ball takes, including gravity.\n\n"
            "IVB tells you about spin quality. Total Break tells you what "
            "the hitter actually sees. Both matter, but for different reasons."
        ),
        "stat_col": None,
        "stat_label": None,
        "pct": False,
        "ascending": False,
    },
    {
        "id": "explainer_vaa",
        "title": "VAA: Overhyped or misunderstood?",
        "body": (
            "Vertical Approach Angle (VAA) = the angle the ball enters the "
            "strike zone.\n\n"
            "Flatter VAA (closer to 0) on fastballs makes them harder to "
            "lift. Steeper VAA on breaking balls creates more swing-through.\n\n"
            "It's not overhyped — it's just one piece of the puzzle alongside "
            "velocity, movement, and location."
        ),
        "stat_col": None,
        "stat_label": None,
        "pct": False,
        "ascending": False,
    },
]


class ExplainerGenerator(ContentGenerator):
    name = "explainer"

    async def generate(self) -> PostContent:
        # Pick a topic not recently posted
        available = [t for t in _TOPICS if not _was_recently_posted(t["id"])]
        if not available:
            available = _TOPICS  # all used recently, reset

        topic = random.choice(available)

        # If topic has a stat column, fetch the current leader as a real example
        leader_line = ""
        leader_name = None
        leader_pitcher_id = None
        if topic["stat_col"]:
            try:
                df = pitch_profiler.get_season_pitchers()
                if not df.empty and topic["stat_col"] in df.columns:
                    # Filter to qualified
                    if "innings_pitched" in df.columns:
                        qualified = df[df["innings_pitched"] >= 20]
                        if not qualified.empty:
                            df = qualified

                    if topic["ascending"]:
                        leader_row = df.nsmallest(1, topic["stat_col"]).iloc[0]
                    else:
                        leader_row = df.nlargest(1, topic["stat_col"]).iloc[0]

                    leader_name = get_name(leader_row)
                    leader_pitcher_id = leader_row.get("pitcher_id") or leader_row.get("player_id")
                    leader_val = fmt_stat(leader_row[topic["stat_col"]], topic["pct"])
                    suffix = "%" if topic["pct"] else ""
                    leader_line = (
                        f"\nCurrent leader: {leader_name} at "
                        f"{leader_val}{suffix} {topic['stat_label']}"
                    )
            except Exception:
                log.warning("Failed to fetch leader for explainer topic", exc_info=True)

        text = (
            f"{topic['title']}\n\n"
            f"{topic['body']}"
            f"{leader_line}\n\n"
            f"Data via @mlbpitchprofiler {DEFAULT_HASHTAGS}"
        )

        # Media: movement profile chart from Pitch Profiler data
        image_path = None
        chart_name = leader_name
        try:
            pitches_df = pitch_profiler.get_season_pitches()
            if not pitches_df.empty:
                # If no leader, pick a top pitcher for the visual
                if not chart_name:
                    from .._player_pick import pick_player
                    chart_name = pick_player()["name"]
                image_path = plot_movement_profile(chart_name, pitches_df)
        except Exception:
            log.warning("Movement profile chart failed", exc_info=True)

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Pitch movement profile for {chart_name}" if image_path else "",
            tags=["explainer", topic["id"]],
        )
