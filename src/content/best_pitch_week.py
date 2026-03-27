"""Content generator: Best Pitch of the Week (Fridays).

Finds the best single pitch type performance from the past week using
Pitch Profiler game data, creates a visualization with all metrics,
and fetches the Savant video of that pitch in action.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image as PILImage

import anthropic
import pandas as pd
import requests

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..charts import plot_best_pitch_card
from ..config import SCREENSHOTS_DIR, MLB_SEASON, MLB_API_BASE, DEFAULT_HASHTAGS
from ..video_clips import get_game_strikeout_clip

log = logging.getLogger(__name__)

WATERMARK_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "BachTalk.png"

# Theme
CARD_BG = "#0d1117"
CARD_SURFACE = "#161b22"
CARD_BORDER = "#30363d"
CARD_TEXT = "#f0f6fc"
CARD_TEXT_MUTED = "#8b949e"
GOLD = "#ffbe0b"

PITCH_COLORS = {
    "FF": "#d62828", "SI": "#f77f00", "FC": "#8338ec",
    "SL": "#3a86ff", "SV": "#00b4d8", "ST": "#00b4d8",
    "CU": "#2ec4b6", "KC": "#06d6a0", "CH": "#ffbe0b",
    "FS": "#fb5607", "KN": "#9d4edd", "CT": "#8338ec",
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "SV": "Sweeper", "ST": "Sweeper",
    "CU": "Curveball", "KC": "Knuckle Curve", "CH": "Changeup",
    "FS": "Splitter", "KN": "Knuckleball", "CT": "Cutter",
}


def _get_week_game_pks() -> list[int]:
    """Get game PKs from the past 7 days."""
    pks = []
    for days_ago in range(1, 8):
        d = date.today() - timedelta(days=days_ago)
        url = f"{MLB_API_BASE}/schedule?sportId=1&date={d.strftime('%Y-%m-%d')}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            for dd in data.get("dates", []):
                for g in dd.get("games", []):
                    if "Final" in g.get("status", {}).get("detailedState", ""):
                        pks.append(g["gamePk"])
        except Exception:
            continue
    return pks


def _build_best_pitch_card(row, pitcher_name, pitch_name_display, game_date):
    """Build visualization card for the best pitch of the week."""
    fig = plt.figure(figsize=(12, 8), dpi=150)
    fig.set_facecolor(CARD_BG)

    # Background
    noise = np.random.default_rng(42).uniform(0.04, 0.07, (68, 120))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.imshow(noise, aspect="auto", cmap="gray", alpha=0.03, extent=[0, 1, 0, 1])
    bg.axis("off")

    pt = str(row.get("pitch_type", "FF"))
    accent = PITCH_COLORS.get(pt, "#3a86ff")

    # Accent stripe
    stripe = fig.add_axes([0, 0.97, 1, 0.03])
    stripe.set_xlim(0, 1); stripe.set_ylim(0, 1)
    stripe.add_patch(Rectangle((0, 0), 1, 1, color=accent))
    stripe.axis("off")

    # Title
    shadow = [patheffects.withStroke(linewidth=4, foreground=CARD_BG)]
    fig.text(0.5, 0.93, "Best Pitch of the Week", fontsize=22,
             fontweight="bold", color=CARD_TEXT, ha="center", va="top", path_effects=shadow)
    fig.text(0.5, 0.89,
             f"{pitcher_name}  |  {pitch_name_display}  |  Week of {game_date}",
             fontsize=11, color=accent, ha="center", va="top", fontweight="bold")

    # Main stats grid
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.74])
    ax.set_facecolor(CARD_SURFACE)
    for s in ax.spines.values():
        s.set_color(CARD_BORDER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Stats to display
    stats = [
        ("Velocity", f"{row.get('velocity', '?'):.1f} mph" if row.get('velocity') else "—", 0),
        ("Spin Rate", f"{row.get('spin_rate', '?'):.0f} rpm" if row.get('spin_rate') else "—", 1),
        ("IVB", f'{row.get("ivb", "?"):.1f}"' if row.get("ivb") else "—", 2),
        ("HB", f'{row.get("hb", "?"):.1f}"' if row.get("hb") else "—", 3),
        ("Stuff+", f"{row.get('stuff_plus', '?'):.0f}" if row.get("stuff_plus") else "—", 4),
        ("Whiff%", f"{float(row.get('whiff_rate', 0))*100:.1f}%" if row.get("whiff_rate") else "—", 5),
        ("Chase%", f"{float(row.get('chase_percentage', 0))*100:.1f}%" if row.get("chase_percentage") else "—", 6),
        ("CSW%", f"{float(row.get('called_strikes_plus_whiffs_percentage', 0))*100:.1f}%" if row.get("called_strikes_plus_whiffs_percentage") else "—", 7),
        ("Zone%", f"{float(row.get('zone_percentage', 0))*100:.1f}%" if row.get("zone_percentage") else "—", 8),
        ("Run Value", f"{float(row.get('run_value_per_100_pitches', 0)):.1f}" if row.get("run_value_per_100_pitches") else "—", 9),
        ("wOBA", f"{row.get('woba', '?'):.3f}" if row.get("woba") else "—", 10),
        ("Pitches", f"{int(row.get('thrown', 0))}", 11),
    ]

    # 4x3 grid layout
    for i, (label, value, idx) in enumerate(stats):
        col = idx % 4
        grid_row = idx // 4
        x = 0.8 + col * 2.3
        y = 6.5 - grid_row * 2.5

        # Value
        val_color = accent if "Stuff+" in label and row.get("stuff_plus", 0) and float(row.get("stuff_plus", 0)) >= 105 else CARD_TEXT
        ax.text(x, y, value, fontsize=16, fontweight="bold", color=val_color,
                va="center", ha="center")
        # Label
        ax.text(x, y - 0.6, label, fontsize=9, color=CARD_TEXT_MUTED,
                va="center", ha="center")

    # Pitch type dot
    ax.scatter(5.0, 0.5, c=accent, s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax.text(5.0, 0.5, pt, fontsize=8, fontweight="bold", color="white",
            va="center", ha="center", zorder=6)

    # Footer
    foot = fig.add_axes([0.03, 0.0, 0.94, 0.003])
    foot.set_xlim(0, 1); foot.set_ylim(0, 1)
    foot.add_patch(Rectangle((0, 0), 1, 1, color=CARD_BORDER))
    foot.axis("off")
    fig.text(0.04, 0.025, "@BachTalk1", fontsize=10, color=accent,
             ha="left", va="center", fontweight="bold")
    fig.text(0.5, 0.025, "Data: Pitch Profiler", fontsize=9,
             color=CARD_TEXT_MUTED, ha="center", va="center")
    fig.text(0.96, 0.025, "Best Pitch of the Week", fontsize=9,
             color=CARD_TEXT_MUTED, ha="right", va="center")

    # Watermark
    if WATERMARK_PATH.exists():
        try:
            img = PILImage.open(WATERMARK_PATH).convert("RGBA")
            arr = np.array(img, dtype=np.float32)
            is_white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
            arr[is_white, 3] = 0
            not_t = arr[:, :, 3] > 0
            arr[not_t, 0] = 255; arr[not_t, 1] = 255; arr[not_t, 2] = 255
            arr = arr.astype(np.uint8)
            ax_wm = fig.add_axes([0.375, 0.3, 0.25, 0.25], zorder=10)
            ax_wm.imshow(arr, alpha=0.1)
            ax_wm.set_facecolor("none")
            ax_wm.axis("off")
        except Exception:
            pass

    out = SCREENSHOTS_DIR / "best_pitch_week.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def _find_savant_video(pitcher_id, pitch_type, game_pk):
    """Find a Savant video for a specific pitch type from a specific game."""
    from ..video_clips import _download_mp4
    try:
        pbp = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay", timeout=15
        ).json()

        for play in pbp.get("allPlays", []):
            if play.get("matchup", {}).get("pitcher", {}).get("id") != pitcher_id:
                continue
            event = (play.get("result", {}).get("event") or "").lower()
            if "strikeout" not in event:
                continue
            for pe in reversed(play.get("playEvents", [])):
                play_id = pe.get("playId")
                if play_id:
                    # Get video
                    surl = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                    resp = requests.get(surl, timeout=10)
                    mp4s = re.findall(r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', resp.text)
                    if mp4s:
                        clip_path = SCREENSHOTS_DIR.parent / "data" / "clips" / f"best_pitch_{pitcher_id}.mp4"
                        clip_path.parent.mkdir(parents=True, exist_ok=True)
                        if _download_mp4(mp4s[0], clip_path):
                            return clip_path
                    break
            break
    except Exception:
        log.warning("Savant video fetch failed", exc_info=True)
    return None


class BestPitchWeekGenerator(ContentGenerator):
    name = "best_pitch_week"

    async def generate(self) -> PostContent:
        # Get this week's game data
        game_pks = _get_week_game_pks()
        if not game_pks:
            return PostContent(text="")

        try:
            game_pitches = pitch_profiler.get_game_pitches(MLB_SEASON)
        except Exception:
            log.warning("Failed to fetch game pitches", exc_info=True)
            return PostContent(text="")

        # Filter to regular season + this week
        reg = game_pitches[game_pitches["game_type"] == "R"]
        week = reg[reg["game_pk"].isin(game_pks)]

        if week.empty:
            return PostContent(text="")

        # Score each pitch type performance
        # Must have thrown at least 10 pitches of this type
        week["thrown"] = pd.to_numeric(week["thrown"], errors="coerce")
        qualified = week[week["thrown"] >= 10].copy()

        if qualified.empty:
            return PostContent(text="")

        # Score: stuff+ * 0.4 + whiff_rate * 100 * 0.3 + csw% * 100 * 0.3
        for col in ["stuff_plus", "whiff_rate", "called_strikes_plus_whiffs_percentage"]:
            qualified[col] = pd.to_numeric(qualified[col], errors="coerce")

        qualified["score"] = (
            qualified["stuff_plus"].fillna(100) * 0.4
            + qualified["whiff_rate"].fillna(0) * 100 * 0.3
            + qualified["called_strikes_plus_whiffs_percentage"].fillna(0) * 100 * 0.3
        )

        # Skip recently posted
        from ..scheduler import was_recently_posted
        qualified = qualified[
            ~qualified["pitcher_name"].apply(lambda n: was_recently_posted(str(n), lookback=10))
        ]

        if qualified.empty:
            return PostContent(text="")

        best = qualified.sort_values("score", ascending=False).iloc[0]
        pitcher_name = str(best["pitcher_name"])
        pitch_type = str(best["pitch_type"])
        pitcher_id = int(best["pitcher_id"]) if pd.notna(best.get("pitcher_id")) else None
        game_pk = int(best["game_pk"]) if pd.notna(best.get("game_pk")) else None
        pitch_display = PITCH_NAMES.get(pitch_type, pitch_type)

        week_start = (date.today() - timedelta(days=7)).strftime("%m/%d")
        game_date = week_start

        log.info("Best pitch: %s's %s (stuff+=%.0f, whiff=%.1f%%)",
                 pitcher_name, pitch_display,
                 best.get("stuff_plus", 0) or 0,
                 (best.get("whiff_rate", 0) or 0) * 100)

        # Get team from MLB API
        team = None
        if pitcher_id:
            try:
                mlb_resp = requests.get(
                    f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=currentTeam",
                    timeout=10)
                team_name = mlb_resp.json()["people"][0].get("currentTeam", {}).get("name", "")
                _N2A = {"Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
                    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
                    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
                    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
                    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
                    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
                    "New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI",
                    "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
                    "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
                    "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH"}
                team = _N2A.get(team_name)
            except Exception:
                pass

        # Build card
        pitch_dict = best.to_dict()
        image_path = plot_best_pitch_card(
            pitcher_name=pitcher_name,
            pitch_type=pitch_type,
            team=team or "",
            player_id=pitcher_id,
            game_date=game_date,
            pitch_data=pitch_dict,
            title="Best Pitch of the Week",
        )

        # Get video
        video_path = None
        if pitcher_id and game_pk:
            video_path = _find_savant_video(pitcher_id, pitch_type, game_pk)

        # Build stats string
        stuff = best.get("stuff_plus", "")
        whiff = best.get("whiff_rate", "")
        velo = best.get("velocity", "")
        stats_parts = []
        if velo: stats_parts.append(f"{float(velo):.1f} mph")
        if stuff: stats_parts.append(f"Stuff+ {float(stuff):.0f}")
        if whiff: stats_parts.append(f"{float(whiff)*100:.1f}% whiff rate")
        stats_str = ", ".join(stats_parts)

        text = (
            f"Best Pitch of the Week\n\n"
            f"{pitcher_name}'s {pitch_display}\n"
            f"{stats_str}\n\n"
            f"@TJStats @PitchProfiler {DEFAULT_HASHTAGS}"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Best pitch of the week: {pitcher_name}'s {pitch_display}",
            tags=["best_pitch_week", pitcher_name, pitch_type],
        )
