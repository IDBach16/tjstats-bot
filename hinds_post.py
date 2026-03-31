#!/usr/bin/env python3
"""One-off: Rece Hinds 2025 hitter card — MLB + MiLB stats side by side."""

import sys
import os
from pathlib import Path
from io import BytesIO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
from PIL import Image

from src.poster import post_with_image
from src.config import SCREENSHOTS_DIR

# ── Constants ──
NAVY = "#1a1a2e"
GOLD = "#c8a951"
WHITE = "#ffffff"
LIGHT_BG = "#f8f9fa"
REDS_RED = "#C6011F"
REDS_ACCENT = "#000000"

PLAYER_ID = 677956
PLAYER_NAME = "Rece Hinds"


def fetch_headshot(pid):
    url = f"https://img.mlbstatic.com/mlb-photos/image/upload/w_400,q_100/v1/people/{pid}/headshot/silo/current"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


def fetch_logo(team_abbrev="cin"):
    url = f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/{team_abbrev}.png&h=200&w=200"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


def fetch_stats(pid, season, sport_id):
    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
           f"?stats=season&season={season}&group=hitting&sportId={sport_id}")
    resp = requests.get(url, timeout=10)
    for sg in resp.json().get("stats", []):
        for s in sg.get("splits", []):
            stat = s["stat"]
            if stat.get("gamesPlayed", 0) > 0:
                return {
                    "team": s.get("team", {}).get("name", ""),
                    "league": s.get("sport", {}).get("name", ""),
                    "G": stat.get("gamesPlayed", 0),
                    "AB": stat.get("atBats", 0),
                    "H": stat.get("hits", 0),
                    "2B": stat.get("doubles", 0),
                    "3B": stat.get("triples", 0),
                    "HR": stat.get("homeRuns", 0),
                    "RBI": stat.get("rbi", 0),
                    "BB": stat.get("baseOnBalls", 0),
                    "K": stat.get("strikeOuts", 0),
                    "SB": stat.get("stolenBases", 0),
                    "AVG": stat.get("avg", ".000"),
                    "OBP": stat.get("obp", ".000"),
                    "SLG": stat.get("slg", ".000"),
                    "OPS": stat.get("ops", ".000"),
                }
    return None


def draw_stat_table(ax, title, stats, accent_color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 0.85), 1, 0.15, facecolor=accent_color,
                                edgecolor="none", zorder=2))
    ax.text(0.5, 0.925, title, ha="center", va="center", fontsize=16,
            fontweight="bold", color=WHITE, zorder=3)

    # Team / League subtitle
    subtitle = f"{stats['team']}"
    ax.text(0.5, 0.78, subtitle, ha="center", va="center", fontsize=10,
            color="#888888")

    # Main slash line
    ax.text(0.5, 0.65, f"{stats['AVG']} / {stats['OBP']} / {stats['SLG']}",
            ha="center", va="center", fontsize=28, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.56, "AVG / OBP / SLG", ha="center", va="center",
            fontsize=9, color="#888888")

    # OPS highlight
    ax.text(0.5, 0.46, f"OPS: {stats['OPS']}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=accent_color)

    # Stat grid
    stat_items = [
        ("G", stats["G"]), ("AB", stats["AB"]), ("H", stats["H"]),
        ("HR", stats["HR"]), ("RBI", stats["RBI"]), ("2B", stats["2B"]),
        ("BB", stats["BB"]), ("K", stats["K"]), ("SB", stats["SB"]),
    ]

    cols = 3
    rows = 3
    for i, (label, val) in enumerate(stat_items):
        col = i % cols
        row = i // cols
        x = 0.17 + col * 0.33
        y = 0.32 - row * 0.12
        ax.text(x, y, str(val), ha="center", va="center", fontsize=16,
                fontweight="bold", color=NAVY)
        ax.text(x, y - 0.04, label, ha="center", va="center", fontsize=8,
                color="#888888")


def generate_card():
    print("Fetching stats...")
    mlb_stats = fetch_stats(PLAYER_ID, 2025, 1)
    milb_stats = fetch_stats(PLAYER_ID, 2025, 11)  # AAA

    if not mlb_stats:
        print("No MLB stats found")
        return None
    if not milb_stats:
        print("No MiLB stats found")
        return None

    print(f"MLB: {mlb_stats['G']}G, {mlb_stats['AVG']} AVG, {mlb_stats['HR']} HR")
    print(f"MiLB: {milb_stats['G']}G, {milb_stats['AVG']} AVG, {milb_stats['HR']} HR")

    headshot = fetch_headshot(PLAYER_ID)
    logo = fetch_logo("cin")

    # ── Figure ──
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.set_facecolor(WHITE)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.2, 0.1, 2.5],
                           hspace=0.15, wspace=0.12)

    # ── Header ──
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")

    # Background
    ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NAVY,
                                    edgecolor="none", zorder=0))
    # Gold accent line
    ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 0.04, facecolor=GOLD,
                                    edgecolor="none", zorder=1))

    # Headshot
    if headshot:
        headshot.thumbnail((300, 300))
        img_arr = np.array(headshot)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        imagebox = OffsetImage(img_arr, zoom=0.45)
        ab = AnnotationBbox(imagebox, (0.08, 0.55), xycoords=ax_hdr.transAxes,
                            frameon=False, zorder=5)
        ax_hdr.add_artist(ab)

    # Name and info
    ax_hdr.text(0.22, 0.72, PLAYER_NAME, fontsize=36, fontweight="bold",
                color=WHITE, va="center", zorder=3)
    ax_hdr.text(0.22, 0.42, "OF  |  Cincinnati Reds  |  2025 Season",
                fontsize=14, color="#aaaaaa", va="center", zorder=3)
    ax_hdr.text(0.22, 0.18, "MLB + MiLB (AAA Louisville Bats)",
                fontsize=11, color=GOLD, va="center", fontweight="bold", zorder=3)

    # Logo
    if logo:
        logo.thumbnail((150, 150))
        logo_arr = np.array(logo)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        logobox = OffsetImage(logo_arr, zoom=0.5)
        lab = AnnotationBbox(logobox, (0.92, 0.55), xycoords=ax_hdr.transAxes,
                             frameon=False, zorder=5)
        ax_hdr.add_artist(lab)

    # ── Divider ──
    ax_div = fig.add_subplot(gs[1, :])
    ax_div.axis("off")

    # ── MLB Stats (left) ──
    ax_mlb = fig.add_subplot(gs[2, 0])
    draw_stat_table(ax_mlb, "MLB (Cincinnati Reds)", mlb_stats, REDS_RED)

    # ── MiLB Stats (right) ──
    ax_milb = fig.add_subplot(gs[2, 1])
    draw_stat_table(ax_milb, "AAA (Louisville Bats)", milb_stats, "#2a4a8e")

    # ── Footer ──
    fig.text(0.02, 0.01, "@BachTalk1", fontsize=10, color="#aaaaaa",
             fontweight="bold")
    fig.text(0.98, 0.01, "Data: MLB Stats API | 2025 Season", fontsize=8,
             color="#cccccc", ha="right")

    # Save
    filepath = SCREENSHOTS_DIR / "hinds_2025_card.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=WHITE,
                edgecolor="none", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def main():
    card_path = generate_card()
    if not card_path:
        print("Card generation failed")
        return

    text = (
        'IDK if he is a franchise player Barry...\n\n'
        'Rece Hinds 2025 Hitting Summary\n'
        'MLB vs AAA — side by side\n\n'
        '@TJStats #Reds #MLB'
    )

    tweet_id = post_with_image(
        text=text,
        image_path=card_path,
        alt_text="Rece Hinds 2025 hitting summary — MLB and MiLB stats",
    )
    print(f"Posted: {tweet_id}")


if __name__ == "__main__":
    main()
