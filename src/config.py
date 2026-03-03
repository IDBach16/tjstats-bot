"""Configuration: env vars, app catalog, constants."""

import os
from datetime import datetime
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
# MLB regular season starts late March; before April, default to prior year
_now = datetime.now()
_default_season = _now.year if _now.month >= 4 else _now.year - 1
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
