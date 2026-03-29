#!/usr/bin/env python
"""Reds Opening Day 2025 example post — generates game summary cards and
posts a thread to X/Twitter.

Run with:
    C:\\Users\\IDBac\\AppData\\Local\\Programs\\Python\\Python313\\python.exe reds_example.py
"""

from __future__ import annotations

import logging
import sys
import time
from io import BytesIO
from pathlib import Path

# Ensure project root is on path so `src` package is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from src.config import SCREENSHOTS_DIR, MLB_SEASON
from src.pitch_profiler import get_game_pitchers, get_game_pitches, get_pbp_game
from src.poster import post_with_image, post_reply, post_video_reply
from src.video_clips import get_pitcher_clip

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

GAME_PK = 778561
GAME_DATE = "03/27/2025"
OPPONENT = "SF Giants"

REDS_PITCHERS = [
    (668881, "Hunter Greene"),
    (663574, "Tony Santillan"),
    (605130, "Scott Barlow"),
    (641941, "Emilio Pagán"),
    (594580, "Sam Moll"),
    (664139, "Ian Gibaut"),
]

# Noise pitch types to filter
_NOISE_PITCHES = {"PO", "IN", "EP", "AB", "AS", "UN", "XX", "NP", "SC"}

# MLB team colours
TEAM_COLORS = {
    "ARI": "#A71930", "ATL": "#CE1141", "BAL": "#DF4601", "BOS": "#BD3039",
    "CHC": "#0E3386", "CWS": "#27251F", "CIN": "#C6011F", "CLE": "#00385D",
    "COL": "#333366", "DET": "#0C2340", "HOU": "#002D62", "KC":  "#004687",
    "LAA": "#BA0021", "LAD": "#005A9C", "MIA": "#00A3E0", "MIL": "#FFC52F",
    "MIN": "#002B5C", "NYM": "#002D72", "NYY": "#003087", "OAK": "#003831",
    "PHI": "#E81828", "PIT": "#FDB827", "SD":  "#2F241D", "SF":  "#FD5A1E",
    "SEA": "#0C2C56", "STL": "#C41E3A", "TB":  "#092C5C", "TEX": "#003278",
    "TOR": "#134A8E", "WSH": "#AB0003",
}

# TJStats pitch colour palette (from charts.py)
_TJ_PITCH_COLOURS = {
    'FF': {'colour': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'colour': '#FF007D', 'name': 'Fastball'},
    'SI': {'colour': '#98165D', 'name': 'Sinker'},
    'FC': {'colour': '#BE5FA0', 'name': 'Cutter'},
    'CH': {'colour': '#F79E70', 'name': 'Changeup'},
    'FS': {'colour': '#FE6100', 'name': 'Splitter'},
    'SC': {'colour': '#F08223', 'name': 'Screwball'},
    'FO': {'colour': '#FFB000', 'name': 'Forkball'},
    'SL': {'colour': '#67E18D', 'name': 'Slider'},
    'ST': {'colour': '#1BB999', 'name': 'Sweeper'},
    'SV': {'colour': '#376748', 'name': 'Slurve'},
    'KC': {'colour': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'colour': '#3025CE', 'name': 'Curveball'},
    'CS': {'colour': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'colour': '#648FFF', 'name': 'Eephus'},
    'KN': {'colour': '#867A08', 'name': 'Knuckleball'},
}
_TJ_COLOUR = {k: v['colour'] for k, v in _TJ_PITCH_COLOURS.items()}
_TJ_NAME = {k: v['name'] for k, v in _TJ_PITCH_COLOURS.items()}

# Colormaps for table cell color-coding
_CMAP_GOOD = LinearSegmentedColormap.from_list("tj", ['#FFB000', '#FFFFFF', '#648FFF'])
_CMAP_BAD = LinearSegmentedColormap.from_list("tj_r", ['#648FFF', '#FFFFFF', '#FFB000'])

# Pitch stats table columns
_PITCH_TABLE_COLS = [
    ("velocity", "$\\bf{Velo}$", ".1f", True),
    ("ivb", "$\\bf{iVB}$", ".1f", None),
    ("hb", "$\\bf{HB}$", ".1f", None),
    ("spin_rate", "$\\bf{Spin}$", ".0f", None),
    ("release_extension", "$\\bf{Ext.}$", ".1f", True),
    ("stuff_plus", "$\\bf{Stf+}$", ".0f", True),
    ("whiff_rate", "$\\bf{Whiff\\%}$", ".1%", True),
    ("chase_percentage", "$\\bf{Chase\\%}$", ".1%", True),
    ("run_value_per_100_pitches", "$\\bf{RV\\/100}$", ".1f", True),
    ("woba", "$\\bf{wOBA}$", ".3f", False),
]

# Game stats row — columns from get_game_pitchers
_GAME_STATS = [
    ("innings_pitched", "$\\bf{IP}$", ".1f"),
    ("hits", "$\\bf{H}$", ".0f"),
    ("runs", "$\\bf{R}$", ".0f"),
    ("earned_runs", "$\\bf{ER}$", ".0f"),
    ("walks", "$\\bf{BB}$", ".0f"),
    ("strike_outs", "$\\bf{K}$", ".0f"),
    ("pitches_thrown", "$\\bf{NP}$", ".0f"),
    ("strike_percentage", "$\\bf{Strk\\%}$", ".1%"),
    ("era", "$\\bf{ERA}$", ".2f"),
    ("fip", "$\\bf{FIP}$", ".2f"),
    ("whiff_rate", "$\\bf{Whiff\\%}$", ".1%"),
    ("stuff_plus", "$\\bf{Stf+}$", ".0f"),
]

# Watermark
_WATERMARK_PATH = ROOT / "assets" / "BachTalk.png"

# Logo map for ESPN CDN
_LOGO_MAP = {
    "AZ": "ari", "ARI": "ari", "ATL": "atl", "BAL": "bal",
    "BOS": "bos", "CHC": "chc", "CWS": "chw", "CIN": "cin",
    "CLE": "cle", "COL": "col", "DET": "det", "HOU": "hou",
    "KC": "kc", "LAA": "laa", "LAD": "lad", "MIA": "mia",
    "MIL": "mil", "MIN": "min", "NYM": "nym", "NYY": "nyy",
    "OAK": "oak", "PHI": "phi", "PIT": "pit", "SD": "sd",
    "SF": "sf", "SEA": "sea", "STL": "stl", "TB": "tb",
    "TEX": "tex", "TOR": "tor", "WSH": "wsh",
}


def _get_table_cell_color(value: float, league_mean: float,
                          cmap, spread: float = 0.3) -> str:
    lo = league_mean * (1 - spread)
    hi = league_mean * (1 + spread)
    if lo >= hi:
        return "#ffffff"
    try:
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        return mcolors.to_hex(cmap(norm(value)))
    except (ValueError, ZeroDivisionError):
        return "#ffffff"


def _pctile_color(p):
    """Return a colour for a percentile value."""
    if p >= 90: return "#c0392b"   # elite red
    if p >= 70: return "#e67e22"   # above avg orange
    if p >= 40: return "#f1c40f"   # average yellow
    if p >= 20: return "#3498db"   # below avg blue
    return "#2c3e50"               # poor dark


def _draw_watermark(fig, alpha=0.12, scale=0.45):
    if not _WATERMARK_PATH.exists():
        return
    try:
        img = Image.open(_WATERMARK_PATH).convert("RGBA")
        arr = np.array(img)
        ax_wm = fig.add_axes([0.5 - scale / 2, 0.5 - scale / 2, scale, scale],
                             zorder=-1)
        ax_wm.imshow(arr, alpha=alpha)
        ax_wm.axis("off")
    except Exception:
        pass


# ── Chart: Game Pitching Summary (forked from plot_pitching_summary) ─────

def plot_game_pitching_summary(
    name: str,
    player_row: pd.Series,
    pitch_rows_df: pd.DataFrame,
    all_pitches_df: pd.DataFrame | None = None,
    team: str = "CIN",
    player_id: int | None = None,
    pbp_df: pd.DataFrame | None = None,
    season_df: pd.DataFrame | None = None,
) -> Path | None:
    """Render a game-day pitching summary card.

    Like plot_pitching_summary but:
      - Movement chart (left) + percentile bars (right)
      - Pitch location scatter plot
      - Subtitle: "Game Summary — {date} vs {opponent}"
      - Game stats row instead of season stats
    """
    try:
        accent = TEAM_COLORS.get(team, "#3a86ff")

        # Pitcher hand
        p_throws = "R"
        for hc in ("p_throws", "throws", "hand"):
            if hc in player_row.index and player_row[hc]:
                p_throws = str(player_row[hc])[0].upper()
                break

        # ── Pitch-type aggregation ───────────────────────────────────
        pitch_rows = pd.DataFrame()
        if pitch_rows_df is not None and not pitch_rows_df.empty:
            prows = pitch_rows_df.copy()
            if "pitch_type" in prows.columns:
                prows = prows[~prows["pitch_type"].isin(_NOISE_PITCHES)]

            # Try to get pitcher hand from pitch data
            if "p_throws" in prows.columns and not prows.empty:
                pt_val = prows["p_throws"].dropna()
                if not pt_val.empty:
                    p_throws = str(pt_val.iloc[0])[0].upper()

            if not prows.empty and "pitch_type" in prows.columns:
                num_cols = [
                    "velocity", "ivb", "hb", "spin_rate",
                    "release_extension", "stuff_plus",
                    "whiff_rate", "chase_percentage",
                    "percentage_thrown", "woba",
                    "run_value_per_100_pitches",
                ]
                for nc in num_cols:
                    if nc in prows.columns:
                        prows[nc] = pd.to_numeric(prows[nc], errors="coerce")

                agg = {}
                for nc in num_cols:
                    if nc in prows.columns:
                        agg[nc] = "sum" if nc == "percentage_thrown" else "mean"
                pitch_rows = prows.groupby("pitch_type", as_index=False).agg(agg)

                # Normalize usage
                if "percentage_thrown" in pitch_rows.columns:
                    total = pitch_rows["percentage_thrown"].sum()
                    if total > 0:
                        pitch_rows["percentage_thrown"] = (
                            pitch_rows["percentage_thrown"] / total
                        )
                pitch_rows = pitch_rows.sort_values(
                    "percentage_thrown", ascending=False
                ).reset_index(drop=True)

        # ── Figure + GridSpec ────────────────────────────────────────
        # 8 rows: border, header, game stats, movement+percentile,
        # pitch locations, pitch table, footer, border
        fig = plt.figure(figsize=(20, 28), dpi=150)
        fig.set_facecolor("white")

        gs = gridspec.GridSpec(
            8, 8,
            height_ratios=[2, 18, 8, 30, 25, 30, 5, 2],
            width_ratios=[1, 18, 18, 18, 18, 18, 18, 1],
            hspace=0.3, wspace=0.3,
        )

        # Border axes (hidden)
        for pos in [gs[0, 1:7], gs[7, 1:7], gs[:, 0], gs[:, -1]]:
            bax = fig.add_subplot(pos)
            bax.axis("off")

        # ── Row 1: Header — headshot / bio / logo ────────────────────
        ax_headshot = fig.add_subplot(gs[1, 1:2])
        ax_bio = fig.add_subplot(gs[1, 2:6])
        ax_logo = fig.add_subplot(gs[1, 6:7])

        # Headshot
        ax_headshot.axis("off")
        if player_id:
            try:
                hs_url = (
                    f"https://img.mlbstatic.com/mlb-photos/image/upload/"
                    f"d_people:generic:headshot:67:current.png/"
                    f"w_640,q_auto:best/v1/people/{player_id}"
                    f"/headshot/silo/current.png"
                )
                resp = requests.get(hs_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                ax_headshot.set_xlim(0, 1.3)
                ax_headshot.set_ylim(0, 1)
                ax_headshot.imshow(img, extent=[0, 1, 0, 1], origin="upper")
            except Exception:
                log.debug("Headshot failed for %s", player_id)

        # Bio text
        ax_bio.axis("off")
        hand_str = f"{p_throws}HP"
        ax_bio.text(0.5, 1.0, name, va="top", ha="center",
                    fontsize=42, fontweight="bold")
        ax_bio.text(0.5, 0.62, hand_str, va="top", ha="center",
                    fontsize=22, color="#555555")
        ax_bio.text(0.5, 0.38, f"Game Summary \u2014 {GAME_DATE} vs {OPPONENT}",
                    va="top", ha="center", fontsize=26, fontweight="bold")
        ax_bio.text(0.5, 0.12, "2025 MLB Season", va="top",
                    ha="center", fontsize=20, fontstyle="italic",
                    color="#666666")

        # Team logo
        ax_logo.axis("off")
        slug = _LOGO_MAP.get(team)
        if slug:
            try:
                logo_url = (
                    f"https://a.espncdn.com/combiner/i?img="
                    f"/i/teamlogos/mlb/500/scoreboard/{slug}.png"
                    f"&h=500&w=500"
                )
                resp = requests.get(logo_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                ax_logo.set_xlim(0, 1.3)
                ax_logo.set_ylim(0, 1)
                ax_logo.imshow(img, extent=[0.3, 1.3, 0, 1], origin="upper")
            except Exception:
                log.debug("Logo failed for %s", team)

        # ── Row 2: Game Stats Table ──────────────────────────────────
        ax_season = fig.add_subplot(gs[2, 1:7])
        ax_season.axis("off")

        game_headers = []
        game_values = []
        for col, header, fmt in _GAME_STATS:
            if col in player_row.index:
                try:
                    val = float(player_row[col])
                    game_values.append(format(val, fmt))
                    game_headers.append(header)
                except (TypeError, ValueError):
                    pass

        if game_values:
            tbl = ax_season.table(
                cellText=[game_values],
                colLabels=game_headers,
                cellLoc="center",
                bbox=[0.0, 0.0, 1, 1],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(20)
            tbl.scale(1, 2.5)
            for key, cell in tbl.get_celld().items():
                cell.set_edgecolor("#cccccc")
                if key[0] == 0:
                    cell.set_facecolor("#f0f0f0")
                    cell.set_text_props(fontweight="bold")

        # ── Row 3: Movement chart (LEFT) + Percentile bars (RIGHT) ──
        ax_break = fig.add_subplot(gs[3, 1:4])

        if not pitch_rows.empty and "hb" in pitch_rows.columns and "ivb" in pitch_rows.columns:
            for _, row in pitch_rows.iterrows():
                pt = str(row["pitch_type"])
                hb_val = float(row.get("hb", 0) or 0)
                ivb_val = float(row.get("ivb", 0) or 0)
                color = _TJ_COLOUR.get(pt, "#888888")
                pname = _TJ_NAME.get(pt, pt)
                usage = float(row.get("percentage_thrown", 0.1) or 0.1)
                size = max(100, min(600, usage * 1200))

                # Flip HB for RHP
                hb_plot = -hb_val if p_throws == "R" else hb_val

                ax_break.scatter(hb_plot, ivb_val, c=color, s=size,
                                 edgecolors="black", linewidths=0.8,
                                 alpha=1, zorder=2)
                ax_break.annotate(
                    pname, (hb_plot, ivb_val),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=10, fontweight="bold", color=color,
                )

            ax_break.axhline(0, color="#808080", alpha=0.5,
                             linestyle="--", zorder=1)
            ax_break.axvline(0, color="#808080", alpha=0.5,
                             linestyle="--", zorder=1)
            ax_break.set_xlabel("Horizontal Break (in)", fontsize=16)
            ax_break.set_ylabel("Induced Vertical Break (in)", fontsize=16)
            ax_break.set_title("Pitch Movement", fontsize=20,
                               fontweight="bold")
            ax_break.set_xlim(-25, 25)
            ax_break.set_ylim(-25, 25)
            ax_break.set_aspect("equal", adjustable="box")
            ax_break.grid(True, alpha=0.3)

            if p_throws == "R":
                ax_break.text(-24, -24, "\u2190 Glove Side",
                              fontstyle="italic", fontsize=10,
                              bbox=dict(facecolor="white", edgecolor="black"),
                              ha="left", va="bottom", zorder=3)
                ax_break.text(24, -24, "Arm Side \u2192",
                              fontstyle="italic", fontsize=10,
                              bbox=dict(facecolor="white", edgecolor="black"),
                              ha="right", va="bottom", zorder=3)
            else:
                ax_break.invert_xaxis()
                ax_break.text(24, -24, "\u2190 Arm Side",
                              fontstyle="italic", fontsize=10,
                              bbox=dict(facecolor="white", edgecolor="black"),
                              ha="left", va="bottom", zorder=3)
                ax_break.text(-24, -24, "Glove Side \u2192",
                              fontstyle="italic", fontsize=10,
                              bbox=dict(facecolor="white", edgecolor="black"),
                              ha="right", va="bottom", zorder=3)
        else:
            ax_break.axis("off")
            ax_break.text(0.5, 0.5, "No movement data",
                          ha="center", va="center", fontsize=16)

        # ── Row 3 (right): Percentile bars ──────────────────────────
        ax_pctile = fig.add_subplot(gs[3, 4:7])

        pctile_stats = [
            ("era", "ERA", True),
            ("fip", "FIP", True),
            ("strike_out_percentage", "K%", False),
            ("walk_percentage", "BB%", True),
            ("whiff_rate", "Whiff%", False),
            ("chase_percentage", "Chase%", False),
            ("stuff_plus", "Stuff+", False),
            ("pitching_plus", "Pitching+", False),
        ]

        labels_p = []
        values_p = []
        pctiles_p = []
        compare_df = season_df if season_df is not None else pd.DataFrame()
        for col, label, ascending in pctile_stats:
            if col not in compare_df.columns or col not in player_row.index:
                continue
            vals = pd.to_numeric(compare_df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            try:
                raw_f = float(player_row[col])
            except (TypeError, ValueError):
                continue
            pctile = (vals < raw_f).sum() / len(vals) * 100
            if ascending:
                pctile = 100 - pctile
            labels_p.append(label)
            pctiles_p.append(pctile)
            if col in ("strike_out_percentage", "walk_percentage",
                        "whiff_rate", "chase_percentage"):
                values_p.append(f"{raw_f * 100:.1f}%")
            elif col in ("era", "fip"):
                values_p.append(f"{raw_f:.2f}")
            else:
                values_p.append(f"{raw_f:.0f}")

        if labels_p:
            y_pos = np.arange(len(labels_p))
            colors_p = [_pctile_color(p) for p in pctiles_p]
            bars = ax_pctile.barh(y_pos, pctiles_p, color=colors_p,
                                  height=0.6, edgecolor="none")
            ax_pctile.set_yticks(y_pos)
            ax_pctile.set_yticklabels(labels_p, fontsize=14)
            ax_pctile.set_xlim(0, 108)
            ax_pctile.invert_yaxis()
            ax_pctile.set_title("Percentile Rankings", fontsize=20,
                                fontweight="bold")
            ax_pctile.grid(True, axis="x", alpha=0.3)
            for spine in ax_pctile.spines.values():
                spine.set_visible(False)
            ax_pctile.tick_params(left=False, bottom=False,
                                  labelbottom=False)

            for bar, val, pct, color in zip(bars, values_p, pctiles_p,
                                             colors_p):
                ax_pctile.text(
                    3, bar.get_y() + bar.get_height() / 2,
                    val, va="center", ha="left",
                    fontsize=12, fontweight="bold",
                    color="white" if pct > 25 else "#333333",
                )
                ax_pctile.text(
                    bar.get_width() + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}th", va="center", ha="left",
                    fontsize=12, fontweight="bold", color=color,
                )
        else:
            ax_pctile.axis("off")

        # ── Row 4: Pitch location scatter plot ────────────────────────
        ax_loc = fig.add_subplot(gs[4, 1:7])

        if pbp_df is not None and not pbp_df.empty and "plate_x" in pbp_df.columns and "plate_z" in pbp_df.columns:
            loc_df = pbp_df.dropna(subset=["plate_x", "plate_z"]).copy()
            loc_df["plate_x"] = pd.to_numeric(loc_df["plate_x"], errors="coerce")
            loc_df["plate_z"] = pd.to_numeric(loc_df["plate_z"], errors="coerce")
            loc_df = loc_df.dropna(subset=["plate_x", "plate_z"])

            # Strike zone boundaries
            sz_top = 3.5
            sz_bot = 1.5
            if "sz_top" in loc_df.columns:
                _st = pd.to_numeric(loc_df["sz_top"], errors="coerce").dropna()
                if not _st.empty:
                    sz_top = _st.mean()
            if "sz_bot" in loc_df.columns:
                _sb = pd.to_numeric(loc_df["sz_bot"], errors="coerce").dropna()
                if not _sb.empty:
                    sz_bot = _sb.mean()

            # Draw strike zone rectangle
            from matplotlib.patches import Rectangle
            sz_rect = Rectangle((-0.83, sz_bot), 1.66, sz_top - sz_bot,
                                linewidth=2, edgecolor="black",
                                facecolor="none", zorder=1)
            ax_loc.add_patch(sz_rect)

            # Plot each pitch coloured by pitch type
            pt_col = "pitch_type" if "pitch_type" in loc_df.columns else None
            plotted_types = set()
            if pt_col:
                for pt_code in loc_df[pt_col].unique():
                    if pt_code in _NOISE_PITCHES:
                        continue
                    subset = loc_df[loc_df[pt_col] == pt_code]
                    color = _TJ_COLOUR.get(pt_code, "#888888")
                    label = _TJ_NAME.get(pt_code, pt_code)
                    ax_loc.scatter(subset["plate_x"], subset["plate_z"],
                                   c=color, s=60, alpha=0.85,
                                   edgecolors="black", linewidths=0.4,
                                   label=label, zorder=2)
                    plotted_types.add(pt_code)
            else:
                ax_loc.scatter(loc_df["plate_x"], loc_df["plate_z"],
                               c="#3a86ff", s=60, alpha=0.85,
                               edgecolors="black", linewidths=0.4,
                               zorder=2)

            ax_loc.set_xlim(-2.5, 2.5)
            ax_loc.set_ylim(0, 5)
            ax_loc.set_aspect("equal", adjustable="box")
            ax_loc.set_title("Pitch Locations (Catcher's View)", fontsize=20,
                             fontweight="bold")
            ax_loc.set_xlabel("Horizontal Position (ft)", fontsize=14)
            ax_loc.set_ylabel("Vertical Position (ft)", fontsize=14)
            ax_loc.grid(True, alpha=0.2)
            if plotted_types:
                ax_loc.legend(loc="upper right", fontsize=10, framealpha=0.9)
        else:
            ax_loc.axis("off")
            ax_loc.text(0.5, 0.5, "No pitch location data",
                        ha="center", va="center", fontsize=16)

        # ── Row 5: Color-coded pitch stats table ─────────────────────
        ax_table = fig.add_subplot(gs[5, 1:7])
        ax_table.axis("off")

        if not pitch_rows.empty:
            league_df = all_pitches_df if all_pitches_df is not None else pitch_rows_df

            cell_text = []
            cell_colors = []
            row_label_colors = []

            for _, row in pitch_rows.iterrows():
                pt = str(row["pitch_type"])
                pname = _TJ_NAME.get(pt, pt)
                usage = float(row.get("percentage_thrown", 0) or 0)
                row_data = [pname, f"{usage:.1%}"]
                row_colors = ["#ffffff", "#ffffff"]
                row_label_colors.append(_TJ_COLOUR.get(pt, "#888888"))

                for pp_col, _, fmt, higher_good in _PITCH_TABLE_COLS:
                    if pp_col not in row.index:
                        row_data.append("\u2014")
                        row_colors.append("#ffffff")
                        continue
                    try:
                        val = float(row[pp_col])
                    except (TypeError, ValueError):
                        row_data.append("\u2014")
                        row_colors.append("#ffffff")
                        continue
                    if pd.isna(val):
                        row_data.append("\u2014")
                        row_colors.append("#ffffff")
                        continue

                    row_data.append(format(val, fmt))

                    # Color coding vs league
                    if higher_good is not None and league_df is not None and not league_df.empty:
                        league_vals = (
                            league_df[league_df["pitch_type"] == pt][pp_col]
                            if pp_col in league_df.columns
                            else pd.Series(dtype=float)
                        )
                        league_vals = pd.to_numeric(
                            league_vals, errors="coerce"
                        ).dropna()
                        if not league_vals.empty:
                            lmean = league_vals.mean()
                            cmap = _CMAP_GOOD if higher_good else _CMAP_BAD
                            row_colors.append(
                                _get_table_cell_color(val, lmean, cmap))
                        else:
                            row_colors.append("#ffffff")
                    else:
                        row_colors.append("#ffffff")

                cell_text.append(row_data)
                cell_colors.append(row_colors)

            # Headers
            actual_headers = ["$\\bf{Pitch\\ Name}$", "$\\bf{Pitch\\%}$"]
            for pp_col, header, _, _ in _PITCH_TABLE_COLS:
                actual_headers.append(header)

            n_cols = len(actual_headers)
            col_widths = [2.5] + [1] * (n_cols - 1)

            tbl = ax_table.table(
                cellText=cell_text,
                colLabels=actual_headers,
                cellLoc="center",
                bbox=[0, -0.05, 1, 1],
                colWidths=col_widths,
                cellColours=cell_colors,
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(16)
            tbl.scale(1, 2.2)

            # Style header row
            for i in range(n_cols):
                cell = tbl.get_celld()[(0, i)]
                cell.set_facecolor("#f0f0f0")
                cell.set_edgecolor("#cccccc")
                cell.set_text_props(fontweight="bold")

            # Color pitch name cells
            for i in range(len(cell_text)):
                cell = tbl.get_celld()[(i + 1, 0)]
                cell.set_facecolor(row_label_colors[i])
                r, g, b = mcolors.to_rgb(row_label_colors[i])
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                cell.set_text_props(
                    color="white" if luma < 0.5 else "black",
                    fontweight="bold",
                )
                for j in range(n_cols):
                    tbl.get_celld()[(i + 1, j)].set_edgecolor("#cccccc")

        # ── Footer ───────────────────────────────────────────────────
        ax_footer = fig.add_subplot(gs[6, 1:7])
        ax_footer.axis("off")
        ax_footer.text(0, 1, "By: @BachTalk1", ha="left", va="top",
                       fontsize=22, fontweight="bold")
        ax_footer.text(0.5, 1,
                       "Colour Coding Compares to League Average By Pitch",
                       ha="center", va="top", fontsize=14,
                       color="#666666")
        ax_footer.text(1, 1, "Data: Pitch Profiler\nImages: MLB, ESPN",
                       ha="right", va="top", fontsize=22)

        # ── Save ─────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"game_summary_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, facecolor="white", dpi=150,
                    bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        log.info("Saved game summary card: %s", out)
        return out

    except Exception:
        log.warning("plot_game_pitching_summary failed for %s", name,
                    exc_info=True)
        return None


# ── Helper: build tweet summary line for a pitcher ───────────────────────

def _summary_line(row: pd.Series, name: str) -> str:
    """Build a one-line text summary like 'Hunter Greene: 5.0 IP, 2 ER, 8 K'."""
    parts = [name + ":"]
    for col, label in [
        ("innings_pitched", "IP"),
        ("earned_runs", "ER"),
        ("strike_outs", "K"),
        ("walks", "BB"),
        ("pitches_thrown", "P"),
    ]:
        if col in row.index:
            try:
                val = float(row[col])
                if col == "innings_pitched":
                    parts.append(f"{val:.1f} {label}")
                else:
                    parts.append(f"{int(val)} {label}")
            except (TypeError, ValueError):
                pass
    return ", ".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    log.info("Fetching game pitcher stats from Pitch Profiler...")
    game_pitchers_df = get_game_pitchers(2025)
    log.info("Game pitchers rows: %d", len(game_pitchers_df))

    log.info("Fetching game pitch data from Pitch Profiler...")
    game_pitches_df = get_game_pitches(2025)
    log.info("Game pitches rows: %d", len(game_pitches_df))

    log.info("Fetching pitch-by-pitch data for game %d...", GAME_PK)
    pbp_df = get_pbp_game(GAME_PK)
    log.info("PBP rows: %d", len(pbp_df))

    # Filter to our game
    gp_col = "game_pk" if "game_pk" in game_pitchers_df.columns else None
    if gp_col is None:
        # Try alternative column names
        for c in game_pitchers_df.columns:
            if "game" in c.lower() and "pk" in c.lower():
                gp_col = c
                break
    if gp_col is None:
        log.error("Cannot find game_pk column in game_pitchers. Columns: %s",
                  list(game_pitchers_df.columns))
        return

    game_pitchers_df[gp_col] = pd.to_numeric(game_pitchers_df[gp_col], errors="coerce")
    reds_pitchers = game_pitchers_df[
        (game_pitchers_df[gp_col] == GAME_PK)
    ]

    # Try to filter by team
    team_col = None
    for c in ("team", "team_abbreviation", "team_abbrev"):
        if c in reds_pitchers.columns:
            team_col = c
            break
    if team_col:
        reds_pitchers = reds_pitchers[
            reds_pitchers[team_col].str.upper() == "CIN"
        ]

    log.info("Reds pitchers found: %d", len(reds_pitchers))
    if reds_pitchers.empty:
        log.warning("No Reds pitchers found for game_pk=%d. "
                    "Available game_pks: %s",
                    GAME_PK,
                    game_pitchers_df[gp_col].unique()[:20])
        return

    # Find pitcher name column
    name_col = None
    for c in ("pitcher_name", "player_name", "name"):
        if c in reds_pitchers.columns:
            name_col = c
            break
    if name_col is None:
        log.error("No name column found. Columns: %s", list(reds_pitchers.columns))
        return

    # Find pitcher ID column
    id_col = None
    for c in ("pitcher_id", "player_id", "mlbam_id"):
        if c in reds_pitchers.columns:
            id_col = c
            break

    # Filter game pitches
    pitcher_ids = set(p[0] for p in REDS_PITCHERS)
    gp_pitch_col = "game_pk" if "game_pk" in game_pitches_df.columns else None
    if gp_pitch_col is None:
        for c in game_pitches_df.columns:
            if "game" in c.lower() and "pk" in c.lower():
                gp_pitch_col = c
                break

    game_pitch_id_col = None
    for c in ("pitcher_id", "player_id", "mlbam_id"):
        if c in game_pitches_df.columns:
            game_pitch_id_col = c
            break

    reds_pitches = game_pitches_df.copy()
    if gp_pitch_col:
        reds_pitches[gp_pitch_col] = pd.to_numeric(reds_pitches[gp_pitch_col], errors="coerce")
        reds_pitches = reds_pitches[reds_pitches[gp_pitch_col] == GAME_PK]
    if game_pitch_id_col:
        reds_pitches[game_pitch_id_col] = pd.to_numeric(reds_pitches[game_pitch_id_col], errors="coerce")
        reds_pitches = reds_pitches[reds_pitches[game_pitch_id_col].isin(pitcher_ids)]

    log.info("Reds pitch rows for this game: %d", len(reds_pitches))

    # ── Generate cards ───────────────────────────────────────────────
    cards: list[tuple[str, int, Path, str]] = []  # (name, pid, card_path, summary)

    for pid, pname in REDS_PITCHERS:
        log.info("Generating card for %s (%d)...", pname, pid)

        # Find this pitcher's row in game_pitchers
        if id_col:
            mask = reds_pitchers[id_col].astype(str).str.strip() == str(pid)
            player_matches = reds_pitchers[mask]
        else:
            player_matches = reds_pitchers[reds_pitchers[name_col] == pname]

        if player_matches.empty:
            # Try fuzzy name match
            player_matches = reds_pitchers[
                reds_pitchers[name_col].str.contains(pname.split()[-1], case=False, na=False)
            ]

        if player_matches.empty:
            log.warning("Could not find pitcher row for %s (id=%d)", pname, pid)
            continue

        player_row = player_matches.iloc[0]

        # Get this pitcher's pitch-type data
        if game_pitch_id_col and not reds_pitches.empty:
            pitcher_pitches = reds_pitches[
                reds_pitches[game_pitch_id_col] == pid
            ]
        else:
            pitch_name_col = None
            for c in ("pitcher_name", "player_name", "name"):
                if c in reds_pitches.columns:
                    pitch_name_col = c
                    break
            if pitch_name_col:
                pitcher_pitches = reds_pitches[reds_pitches[pitch_name_col] == pname]
            else:
                pitcher_pitches = pd.DataFrame()

        # Filter PBP data for this pitcher
        pitcher_pbp = pd.DataFrame()
        if not pbp_df.empty:
            pbp_pid_col = None
            for c in ("pitcher_id", "pitcher", "player_id", "mlbam_id"):
                if c in pbp_df.columns:
                    pbp_pid_col = c
                    break
            if pbp_pid_col:
                pbp_df[pbp_pid_col] = pd.to_numeric(pbp_df[pbp_pid_col], errors="coerce")
                pitcher_pbp = pbp_df[pbp_df[pbp_pid_col] == pid]
            else:
                # Try name-based matching
                pbp_name_col = None
                for c in ("pitcher_name", "player_name", "name"):
                    if c in pbp_df.columns:
                        pbp_name_col = c
                        break
                if pbp_name_col:
                    pitcher_pbp = pbp_df[pbp_df[pbp_name_col].str.contains(
                        pname.split()[-1], case=False, na=False)]

        card_path = plot_game_pitching_summary(
            name=pname,
            player_row=player_row,
            pitch_rows_df=pitcher_pitches,
            all_pitches_df=game_pitches_df,
            team="CIN",
            player_id=pid,
            pbp_df=pitcher_pbp,
            season_df=game_pitchers_df,
        )

        summary = _summary_line(player_row, pname)

        if card_path:
            cards.append((pname, pid, card_path, summary))
            log.info("Card saved: %s", card_path)
        else:
            log.warning("Card generation failed for %s", pname)

    if not cards:
        log.error("No cards were generated. Exiting.")
        return

    log.info("Generated %d cards. Posting thread to X...", len(cards))

    # ── Post thread ──────────────────────────────────────────────────
    main_text = (
        "\U0001f534 Reds Pitching Recap \u2014 Opening Day 2025\n"
        "SF Giants 6, Reds 4\n\n"
        "Example of our new daily Reds Summary \u2014 launching next Friday!\n\n"
        "Each day a Reds pitcher takes the mound, you\u2019ll get a full game "
        "breakdown with cards & video.\n\n"
        "Thread below \u2b07\ufe0f\n"
        "@TJStats #Reds #MLB #OpeningDay"
    )

    main_tweet_id = post_with_image(
        text=main_text,
        image_path=cards[0][2],
        alt_text=f"Game summary card for {cards[0][0]}",
    )
    log.info("Main tweet posted: %s", main_tweet_id)

    reply_to = main_tweet_id
    for pname, pid, card_path, summary in cards[1:]:
        # Post card as reply
        reply_id = post_reply(
            text=summary,
            in_reply_to=reply_to,
            image_path=card_path,
            alt_text=f"Game summary card for {pname}",
        )
        log.info("Reply posted for %s: %s", pname, reply_id)
        reply_to = reply_id

        # Try to get a video clip for this pitcher
        try:
            clip_path = get_pitcher_clip(pid, pname)
            if clip_path:
                vid_id = post_video_reply(
                    in_reply_to=reply_to,
                    video_path=clip_path,
                    text=f"\U0001f3ac {pname} strikeout clip",
                )
                log.info("Video reply posted for %s: %s", pname, vid_id)
                reply_to = vid_id
        except Exception:
            log.warning("Video clip failed for %s", pname, exc_info=True)

    # Also try video for the first pitcher (who was in the main tweet)
    try:
        clip_path = get_pitcher_clip(cards[0][1], cards[0][0])
        if clip_path:
            vid_id = post_video_reply(
                in_reply_to=main_tweet_id,
                video_path=clip_path,
                text=f"\U0001f3ac {cards[0][0]} strikeout clip",
            )
            log.info("Video reply posted for %s: %s", cards[0][0], vid_id)
    except Exception:
        log.warning("Video clip failed for %s", cards[0][0], exc_info=True)

    log.info("Thread posted successfully!")


if __name__ == "__main__":
    main()
