"""Configuration: env vars, app catalog, constants."""

import os
from datetime import datetime  # noqa: F401 -- unused while MLB_SEASON is
# pinned, but the commented restore line below needs it. Keeping it makes
# putting the auto-detect back a one-line change.
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)
CLIPS_DIR = ROOT_DIR / "data" / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ── X / Twitter credentials ───────────────────────────────────────────
X_API_KEY = os.environ["X_API_KEY"]
X_API_SECRET = os.environ["X_API_SECRET"]
X_ACCESS_TOKEN = os.environ["X_ACCESS_TOKEN"]
X_ACCESS_TOKEN_SECRET = os.environ["X_ACCESS_TOKEN_SECRET"]
X_BEARER_TOKEN = os.environ["X_BEARER_TOKEN"]

# ── Pitch Profiler ─────────────────────────────────────────────────────
PITCH_PROFILER_API_KEY = os.environ["PITCH_PROFILER_API_KEY"]
PITCH_PROFILER_BASE = (
    "https://g837e5a6fbcb0dd-ch2sockkby63dgzo"
    ".adb.us-chicago-1.oraclecloudapps.com/ords/admin/patreon"
)

# ── MLB Stats API ──────────────────────────────────────────────────────
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ── Season ─────────────────────────────────────────────────────────────
# PINNED TO 2026 (Ian, 2026-09-18). Both scheduled cards read full-season data,
# and a completed season is the right thing to post through the winter.
#
# ⚠ THIS DOES NOT ROLL OVER BY ITSELF. Set it to 2027 once that season has real
# sample -- a few weeks in, not on opening day. Until someone changes it the bot
# will keep posting 2026 cards forever, which is the deliberate trade: silently
# stale beats silently empty.
#
# What it used to do, and why that was the problem: the line below flipped the
# season on 20 March, about a week before opening day. For that week
# get_season_pitchers(2027) returns nothing (Pitching Summary logs a warning and
# posts nothing, which is harmless) and Savant's swing-path board returns a
# handful of swings (Hitter Analysis computes Swing+ off almost no data and
# posts a card that looks real). Posting a bad card is worse than posting none.
#
#   _now = datetime.now()
#   _default_season = _now.year if (_now.month > 3 or (_now.month == 3 and _now.day >= 20)) else _now.year - 1
#
# The env override is kept, so a one-off run can still target another season:
#   MLB_SEASON=2025 python -m src.main --generator pitching_summary
_default_season = 2026
MLB_SEASON = int(os.environ.get("MLB_SEASON", _default_season))

# ── TJStats Hugging Face Spaces catalog ────────────────────────────────
# Each entry: (slug, human label, HF space URL)
HF_SPACES = {
    "pitching_summary": {
        "url": "https://tjstatsapps-pitching-summary-complete.hf.space",
        "label": "Pitching Summary",
        "description": "Season pitching summary cards",
    },
    "statcast_cards": {
        "url": "https://tjstatsapps-2025-mlb-cards.hf.space",
        "label": "Statcast Cards",
        "description": "Percentile bar cards",
    },
    "pitch_plots": {
        "url": "https://tjstatsapps-pitch-plots.hf.space",
        "label": "Pitch Plots",
        "description": "Movement / location plots",
    },
    "leaderboard": {
        "url": "https://tjstatsapps-2025-mlb-statcast-leaderboard.hf.space",
        "label": "Statcast Leaderboard",
        "description": "Stat leader tables",
    },
    "heat_maps": {
        "url": "https://tjstatsapps-pitching-heat-maps.hf.space",
        "label": "Heat Maps",
        "description": "Pitcher heat maps",
    },
    "daily_summary": {
        "url": "https://tjstatsapps-pitching-summary-daily.hf.space",
        "label": "Daily Pitching Summary",
        "description": "Daily pitching summary cards",
    },
}

# ── Hashtags ───────────────────────────────────────────────────────────
DEFAULT_HASHTAGS = "#MLB #Statcast"
