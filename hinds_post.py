#!/usr/bin/env python3
"""One-off: Rece Hinds 2025 hitter card — MLB + MiLB with Statcast plate discipline."""

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
import matplotlib.colors as mcolors
import numpy as np
import requests
from PIL import Image
from pybaseball import statcast_batter

from src.poster import post_with_image
from src.config import SCREENSHOTS_DIR

NAVY = "#1a1a2e"
GOLD = "#c8a951"
WHITE = "#ffffff"
REDS_RED = "#C6011F"

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


def fetch_logo(team="cin"):
    url = f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/{team}.png&h=200&w=200"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


def fetch_season_stats(pid, season, sport_id):
    url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
           f"?stats=season&season={season}&group=hitting&sportId={sport_id}")
    resp = requests.get(url, timeout=10)
    for sg in resp.json().get("stats", []):
        for s in sg.get("splits", []):
            stat = s["stat"]
            if stat.get("gamesPlayed", 0) > 0:
                return {
                    "team": s.get("team", {}).get("name", ""),
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
                    "PA": stat.get("plateAppearances", 0),
                }
    return None


def get_statcast_discipline(pid):
    """Get plate discipline metrics from Statcast."""
    print("Pulling Statcast data...")
    df = statcast_batter("2025-03-20", "2025-09-30", pid)
    if df.empty:
        return None, None

    total = len(df)
    swing_descs = ["hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
                   "foul_tip", "hit_into_play_no_out", "hit_into_play_score",
                   "foul_bunt", "missed_bunt"]
    whiff_descs = ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]

    swings = df[df["description"].isin(swing_descs)].shape[0]
    whiffs = df[df["description"].isin(whiff_descs)].shape[0]
    called_k = df[df["description"] == "called_strike"].shape[0]

    in_zone = df[df["zone"].between(1, 9)]
    out_zone = df[~df["zone"].between(1, 9) & df["zone"].notna()]

    z_swings = in_zone[in_zone["description"].isin(swing_descs)].shape[0]
    z_whiffs = in_zone[in_zone["description"].isin(whiff_descs)].shape[0]
    o_swings = out_zone[out_zone["description"].isin(swing_descs)].shape[0]

    # xStats
    xba = df["estimated_ba_using_speedangle"].dropna().mean()
    xwoba = df["estimated_woba_using_speedangle"].dropna().mean()
    ev = df["launch_speed"].dropna().mean()
    la = df["launch_angle"].dropna().mean()

    discipline = {
        "Pitches": total,
        "Swing%": swings / total * 100 if total else 0,
        "Whiff%": whiffs / swings * 100 if swings else 0,
        "K%": 47.7,  # from season stats (21/44)
        "BB%": 2.3,
        "Zone%": len(in_zone) / total * 100 if total else 0,
        "Z-Swing%": z_swings / len(in_zone) * 100 if len(in_zone) else 0,
        "Z-Whiff%": z_whiffs / z_swings * 100 if z_swings else 0,
        "Chase%": o_swings / len(out_zone) * 100 if len(out_zone) else 0,
        "CSW%": (whiffs + called_k) / total * 100 if total else 0,
        "xBA": xba,
        "xwOBA": xwoba,
        "Avg EV": ev,
        "Avg LA": la,
    }

    # Zone heatmap (whiff% per zone 1-9)
    zone_whiffs = {}
    for z in range(1, 10):
        zdf = df[df["zone"] == z]
        zs = zdf[zdf["description"].isin(swing_descs)].shape[0]
        zw = zdf[zdf["description"].isin(whiff_descs)].shape[0]
        zone_whiffs[z] = (zw / zs * 100) if zs > 0 else None

    return discipline, zone_whiffs


def draw_zone_heatmap(ax, zone_data, title="Whiff% by Zone"):
    """Draw 3x3 zone heatmap."""
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=10)

    cmap = mcolors.LinearSegmentedColormap.from_list("whiff",
        ["#2e7d32", "#66bb6a", "#fff9c4", "#ff7043", "#c62828"])

    # Zone layout (catcher's view): 1=top-left, 3=top-right, 7=bot-left, 9=bot-right
    grid = [
        [1, 2, 3],  # top
        [4, 5, 6],  # mid
        [7, 8, 9],  # low
    ]

    for row in range(3):
        for col in range(3):
            z = grid[row][col]
            val = zone_data.get(z)
            x, y = col, 2 - row

            if val is not None:
                norm_val = min(val / 80, 1.0)  # normalize to 0-80% range
                color = cmap(norm_val)
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                             facecolor=color, edgecolor=NAVY, linewidth=2, zorder=2))
                # Determine text color
                tc = WHITE if norm_val > 0.5 else NAVY
                ax.text(x, y + 0.05, f"{val:.0f}%", ha="center", va="center",
                        fontsize=16, fontweight="bold", color=tc, zorder=3)
            else:
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                             facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=1, zorder=2))
                ax.text(x, y, "-", ha="center", va="center", fontsize=14, color="#999")

    ax.text(1, -0.9, "Catcher's View", ha="center", fontsize=8, color="#888", fontstyle="italic")


def draw_discipline_table(ax, disc):
    """Draw plate discipline metrics table."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Plate Discipline", fontsize=14, fontweight="bold", color=NAVY, pad=10)

    metrics = [
        ("Swing%", f"{disc['Swing%']:.1f}%"),
        ("Whiff%", f"{disc['Whiff%']:.1f}%"),
        ("K%", f"{disc['K%']:.1f}%"),
        ("BB%", f"{disc['BB%']:.1f}%"),
        ("Zone%", f"{disc['Zone%']:.1f}%"),
        ("Z-Swing%", f"{disc['Z-Swing%']:.1f}%"),
        ("Z-Whiff%", f"{disc['Z-Whiff%']:.1f}%"),
        ("Chase%", f"{disc['Chase%']:.1f}%"),
        ("CSW%", f"{disc['CSW%']:.1f}%"),
    ]

    # Color thresholds (from hitter perspective: lower whiff/K/chase is better)
    def get_color(label, val_str):
        try:
            v = float(val_str.replace("%", ""))
        except:
            return "#f0f0f0"
        if "Whiff" in label or label == "K%":
            if v <= 20: return "#a8e6a3"
            if v <= 30: return "#fff9c4"
            return "#ef9a9a"
        if label == "BB%":
            if v >= 10: return "#a8e6a3"
            if v >= 7: return "#fff9c4"
            return "#ef9a9a"
        if label == "Chase%":
            if v <= 25: return "#a8e6a3"
            if v <= 32: return "#fff9c4"
            return "#ef9a9a"
        return "#f0f2f5"

    for i, (label, val) in enumerate(metrics):
        y = 0.92 - i * 0.1
        bg = get_color(label, val)
        ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.96, 0.08,
                     facecolor=bg, edgecolor="#dee2e6", linewidth=0.5, zorder=1))
        ax.text(0.06, y, label, fontsize=11, fontweight="600", color=NAVY,
                va="center", zorder=2)
        ax.text(0.94, y, val, fontsize=12, fontweight="bold", color=NAVY,
                va="center", ha="right", zorder=2)


def draw_xstats(ax, disc):
    """Draw expected stats section."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Statcast Values", fontsize=14, fontweight="bold", color=NAVY, pad=10)

    items = [
        ("xBA", f"{disc['xBA']:.3f}"),
        ("xwOBA", f"{disc['xwOBA']:.3f}"),
        ("Avg Exit Velo", f"{disc['Avg EV']:.1f} mph"),
        ("Avg Launch Angle", f"{disc['Avg LA']:.1f}\u00b0"),
    ]

    for i, (label, val) in enumerate(items):
        y = 0.82 - i * 0.18
        ax.text(0.5, y + 0.04, val, ha="center", va="center",
                fontsize=22, fontweight="bold", color=NAVY)
        ax.text(0.5, y - 0.05, label, ha="center", va="center",
                fontsize=10, color="#888")


def draw_stat_table(ax, title, stats, accent_color):
    """Draw traditional stats summary."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, facecolor=accent_color,
                                edgecolor="none", zorder=2))
    ax.text(0.5, 0.94, title, ha="center", va="center", fontsize=14,
            fontweight="bold", color=WHITE, zorder=3)

    ax.text(0.5, 0.76, f"{stats['AVG']} / {stats['OBP']} / {stats['SLG']}",
            ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.69, "AVG / OBP / SLG", ha="center", fontsize=8, color="#888")

    ax.text(0.5, 0.58, f"OPS: {stats['OPS']}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=accent_color)

    items = [("G", stats["G"]), ("AB", stats["AB"]), ("H", stats["H"]),
             ("HR", stats["HR"]), ("RBI", stats["RBI"]), ("BB", stats["BB"]),
             ("K", stats["K"]), ("SB", stats["SB"]), ("2B", stats["2B"])]
    for i, (l, v) in enumerate(items):
        col, row = i % 3, i // 3
        x = 0.17 + col * 0.33
        y = 0.42 - row * 0.14
        ax.text(x, y, str(v), ha="center", fontsize=14, fontweight="bold", color=NAVY)
        ax.text(x, y - 0.045, l, ha="center", fontsize=7, color="#888")


def generate_card():
    print("Fetching stats...")
    mlb = fetch_season_stats(PLAYER_ID, 2025, 1)
    milb = fetch_season_stats(PLAYER_ID, 2025, 11)
    disc, zones = get_statcast_discipline(PLAYER_ID)
    headshot = fetch_headshot(PLAYER_ID)
    logo = fetch_logo("cin")

    if not mlb or not milb or not disc:
        print("Missing data")
        return None

    print(f"MLB: {mlb['AVG']} AVG, {mlb['HR']} HR | MiLB: {milb['AVG']} AVG, {milb['HR']} HR")
    print(f"Whiff%: {disc['Whiff%']:.1f}%, Chase%: {disc['Chase%']:.1f}%")

    # ── Figure: 5 rows ──
    fig = plt.figure(figsize=(18, 22), dpi=150)
    fig.set_facecolor(WHITE)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1.0, 1.8, 1.8, 1.4],
                           hspace=0.25, wspace=0.2)

    # ── Row 0: Header ──
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NAVY, edgecolor="none"))
    ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 0.03, facecolor=GOLD, edgecolor="none", zorder=1))

    if headshot:
        headshot.thumbnail((300, 300))
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        ib = OffsetImage(np.array(headshot), zoom=0.45)
        ax_hdr.add_artist(AnnotationBbox(ib, (0.07, 0.55), xycoords=ax_hdr.transAxes, frameon=False))

    ax_hdr.text(0.20, 0.72, PLAYER_NAME, fontsize=34, fontweight="bold", color=WHITE, va="center")
    ax_hdr.text(0.20, 0.42, "OF  |  Cincinnati Reds  |  2025 Season", fontsize=13, color="#aaa", va="center")
    ax_hdr.text(0.20, 0.18, "MLB + MiLB (AAA) with Statcast Plate Discipline", fontsize=11, color=GOLD, fontweight="bold", va="center")

    if logo:
        logo.thumbnail((150, 150))
        lb = OffsetImage(np.array(logo), zoom=0.5)
        ax_hdr.add_artist(AnnotationBbox(lb, (0.93, 0.55), xycoords=ax_hdr.transAxes, frameon=False))

    # ── Row 1: MLB stats (left) + MiLB stats (right) ──
    ax_mlb = fig.add_subplot(gs[1, 0:2])
    draw_stat_table(ax_mlb, "MLB (Cincinnati Reds)", mlb, REDS_RED)

    ax_milb = fig.add_subplot(gs[1, 2])
    draw_stat_table(ax_milb, "AAA (Louisville Bats)", milb, "#2a4a8e")

    # ── Row 2: Zone heatmap (left) + Discipline (center) + xStats (right) ──
    ax_zone = fig.add_subplot(gs[2, 0])
    draw_zone_heatmap(ax_zone, zones, "Whiff% by Zone (MLB)")

    ax_disc = fig.add_subplot(gs[2, 1])
    draw_discipline_table(ax_disc, disc)

    ax_xstats = fig.add_subplot(gs[2, 2])
    draw_xstats(ax_xstats, disc)

    # ── Row 3: Footer ──
    ax_foot = fig.add_subplot(gs[3, :])
    ax_foot.set_xlim(0, 1)
    ax_foot.set_ylim(0, 1)
    ax_foot.axis("off")
    ax_foot.add_patch(plt.Rectangle((0, 0.8), 1, 0.02, facecolor=NAVY, edgecolor="none"))
    ax_foot.text(0.02, 0.5, "@BachTalk1", fontsize=12, fontweight="bold", color=NAVY, va="center")
    ax_foot.text(0.98, 0.5, "Data: Statcast / MLB Stats API | 2025", fontsize=9, color="#aaa", ha="right", va="center")
    ax_foot.text(0.5, 0.5, "171 MLB pitches tracked", fontsize=10, color="#888", ha="center", va="center")

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
        'MLB vs AAA with Statcast plate discipline\n\n'
        '@TJStats #Reds #MLB'
    )

    tweet_id = post_with_image(
        text=text,
        image_path=card_path,
        alt_text="Rece Hinds 2025 — MLB vs MiLB stats with Statcast whiff zones and plate discipline",
    )
    print(f"Posted: {tweet_id}")


if __name__ == "__main__":
    main()
