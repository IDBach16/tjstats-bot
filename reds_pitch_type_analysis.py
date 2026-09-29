"""Reds Starter Pitch Type Analysis — ERA vs Arsenal Size & Pitch Type Comparison.

Chart 1: Scatter — # of pitch types vs ERA for all MLB starters (Reds highlighted)
Chart 2: Grouped bar — Reds starters' avg RV/100 per pitch type vs league avg
"""

import io
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MLB_SEASON = 2026
MIN_IP = 10  # minimum innings to qualify as a starter (early-season friendly)
SCREENSHOTS_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# ── Theme (matching BachTalk style) ──────────────────────────────────
BG_COLOR = "#1a1a2e"
SURFACE_COLOR = "#0d1117"
TEXT_COLOR = "#e0e0e0"
MUTED_COLOR = "#8b949e"
GRID_COLOR = "#2a2a4a"
REDS_COLOR = "#C6011F"
LEAGUE_COLOR = "#3a86ff"

PITCH_COLORS = {
    "FF": "#d62828", "SI": "#f77f00", "FC": "#8338ec", "SL": "#3a86ff",
    "SV": "#00b4d8", "ST": "#00b4d8", "CU": "#2ec4b6", "KC": "#06d6a0",
    "CH": "#ffbe0b", "FS": "#fb5607", "KN": "#9d4edd",
}
PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "SV": "Sweeper", "ST": "Sweeper", "CU": "Curveball", "KC": "K-Curve",
    "CH": "Changeup", "FS": "Splitter", "KN": "Knuckle",
}
NOISE_PITCHES = {"PO", "IN", "EP", "AB", "AS", "UN", "XX", "NP", "SC"}


def _add_watermark(fig, size_pct=0.35, alpha=0.06):
    """Add BachTalk logo as a large, subtle centered background watermark."""
    wm_path = ASSETS_DIR / "BachTalk.png"
    if not wm_path.exists():
        return
    from PIL import Image, ImageEnhance

    wm = Image.open(wm_path).convert("RGBA")

    # Scale to a percentage of figure height
    fig_h_px = int(fig.get_figheight() * fig.dpi)
    fig_w_px = int(fig.get_figwidth() * fig.dpi)
    target_h = int(fig_h_px * size_pct)
    scale = target_h / wm.size[1]
    target_w = int(wm.size[0] * scale)
    wm = wm.resize((target_w, target_h), Image.LANCZOS)

    # Reduce opacity via alpha channel
    r, g, b, a = wm.split()
    a = a.point(lambda p: int(p * alpha))
    wm = Image.merge("RGBA", (r, g, b, a))

    wm_arr = np.array(wm)
    # Center it
    xo = (fig_w_px - target_w) // 2
    yo = (fig_h_px - target_h) // 2
    fig.figimage(wm_arr, xo=xo, yo=yo, zorder=1)


TEAM_ID_TO_ABBREV = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def fetch_pitcher_stats() -> pd.DataFrame:
    """Fetch season pitcher stats from MLB Stats API (ERA, IP, team, etc.)."""
    log.info("Fetching pitcher stats from MLB Stats API...")
    all_rows = []
    for offset in range(0, 1000, 500):
        url = (
            f"https://statsapi.mlb.com/api/v1/stats?stats=season&season={MLB_SEASON}"
            f"&group=pitching&sportId=1&gameType=R&limit=500&offset={offset}"
            f"&sortStat=inningsPitched&order=desc"
        )
        resp = requests.get(url, timeout=30)
        splits = resp.json()["stats"][0]["splits"]
        if not splits:
            break
        for s in splits:
            stat = s["stat"]
            gs = int(stat.get("gamesStarted", 0))
            ip = float(stat.get("inningsPitched", "0") or "0")
            if gs < 2 or ip < MIN_IP:
                continue
            team_id = s["team"]["id"]
            k = int(stat.get("strikeOuts", 0))
            bb = int(stat.get("baseOnBalls", 0))
            hr = int(stat.get("homeRuns", 0))
            bf = int(stat.get("battersFaced", 0) or 0)
            er = int(stat.get("earnedRuns", 0))
            h = int(stat.get("hits", 0))
            # FIP = ((13*HR + 3*BB - 2*K) / IP) + 3.2
            fip = ((13 * hr + 3 * bb - 2 * k) / ip + 3.2) if ip > 0 else 0
            k_pct = k / bf * 100 if bf > 0 else 0
            bb_pct = bb / bf * 100 if bf > 0 else 0
            k_bb = k_pct - bb_pct
            all_rows.append({
                "player_id": s["player"]["id"],
                "player_name": s["player"]["fullName"],
                "team": TEAM_ID_TO_ABBREV.get(team_id, "???"),
                "era": float(stat.get("era", "0") or "0"),
                "fip": round(fip, 2),
                "ip": ip,
                "gs": gs,
                "k": k,
                "bb": bb,
                "hr": hr,
                "bf": bf,
                "h": h,
                "whip": float(stat.get("whip", "0") or "0"),
                "k_pct": round(k_pct, 1),
                "bb_pct": round(bb_pct, 1),
                "k_bb": round(k_bb, 1),
                "k_per_9": float(stat.get("strikeoutsPer9Inn", "0") or "0"),
                "bb_per_9": float(stat.get("walksPer9Inn", "0") or "0"),
                "hr_per_9": float(stat.get("homeRunsPer9", "0") or "0"),
            })
    df = pd.DataFrame(all_rows)
    log.info("Got %d starters with %d+ IP and 2+ GS", len(df), MIN_IP)
    return df


def fetch_arsenal_stats() -> pd.DataFrame:
    """Fetch per-pitcher, per-pitch-type arsenal stats from Savant (min=1 PA)."""
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats?"
        f"type=pitcher&pitchType=&year={MLB_SEASON}&position=SP&team=&min=1&csv=true"
    )
    log.info("Fetching pitch arsenal stats from Savant (min=1 PA)...")
    resp = requests.get(url, timeout=30)
    df = pd.read_csv(io.StringIO(resp.text))
    log.info("Got %d pitch-type rows", len(df))
    return df


def fetch_pitch_velocities(arsenal_df: pd.DataFrame = None) -> pd.DataFrame:
    """Fetch pitch arsenal velocity data for pitch type inventory.

    If arsenal_df is provided, only count pitch types that also appear
    in arsenal stats (filters out phantom types with no PA data).
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?"
        f"type=avg_speed&year={MLB_SEASON}&min=10&csv=true"
    )
    log.info("Fetching pitch velocity data from Savant...")
    resp = requests.get(url, timeout=30)
    df = pd.read_csv(io.StringIO(resp.text))

    # Build set of valid (player_id, pitch_type) pairs from arsenal data
    valid_pairs = None
    if arsenal_df is not None and not arsenal_df.empty:
        valid_pairs = set(
            zip(arsenal_df["player_id"].astype(int),
                arsenal_df["pitch_type"].str.upper())
        )

    # Convert wide format to pitch type counts per pitcher
    pitch_cols = [c for c in df.columns if c.endswith("_avg_speed")]
    rows = []
    for _, row in df.iterrows():
        pid = row.get("pitcher")
        if pd.isna(pid):
            continue
        pid = int(pid)
        types = []
        for c in pitch_cols:
            if pd.notna(row[c]) and row[c] > 0:
                pt = c.replace("_avg_speed", "").upper()
                # Only include if it has actual arsenal stats
                if valid_pairs is not None and (pid, pt) not in valid_pairs:
                    continue
                types.append(pt)
        rows.append({"player_id": pid, "num_pitch_types": len(types), "pitch_types": types})

    result = pd.DataFrame(rows)
    log.info("Got pitch type counts for %d pitchers from velocity data", len(result))
    return result


def plot_arsenal_size_vs_era(pitcher_df: pd.DataFrame, arsenal_df: pd.DataFrame) -> Path:
    """Chart 1: Scatter — # pitch types vs ERA, Reds highlighted."""

    # Count distinct pitch types per pitcher from arsenal data
    # Filter noise pitch types
    clean_arsenal = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()

    pitch_counts = clean_arsenal.groupby("player_id")["pitch_type"].nunique().reset_index()
    pitch_counts.columns = ["player_id", "num_pitch_types"]

    # Merge with pitcher stats
    merged = pitcher_df.merge(pitch_counts, on="player_id", how="inner")

    merged = merged.dropna(subset=["era", "num_pitch_types"])
    merged = merged[merged["era"] <= 8.0]

    # Identify Reds pitchers
    reds = merged[merged["team"] == "CIN"]
    others = merged[merged["team"] != "CIN"]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # League scatter
    ax.scatter(others["num_pitch_types"], others["era"],
               c=LEAGUE_COLOR, alpha=0.35, s=50, edgecolors="none",
               label="MLB Starters", zorder=2)

    # Reds scatter (larger, highlighted)
    if not reds.empty:
        ax.scatter(reds["num_pitch_types"], reds["era"],
                   c=REDS_COLOR, alpha=0.95, s=120, edgecolors="white",
                   linewidths=1.5, label="Reds Starters", zorder=4)

        # Label Reds pitchers
        for _, row in reds.iterrows():
            ax.annotate(
                row["player_name"],
                (row["num_pitch_types"], row["era"]),
                fontsize=9, color="white", fontweight="bold",
                xytext=(8, 6), textcoords="offset points",
                path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)],
                zorder=5,
            )

    # Trend line
    x_all = merged["num_pitch_types"].values
    y_all = merged["era"].values
    if len(x_all) > 5:
        z = np.polyfit(x_all, y_all, 1)
        p = np.poly1d(z)
        x_range = np.linspace(x_all.min() - 0.2, x_all.max() + 0.2, 100)
        ax.plot(x_range, p(x_range), color="#ffbe0b", linewidth=2,
                linestyle="--", alpha=0.6, label=f"Trend (slope: {z[0]:+.2f})", zorder=3)

    # Averages by pitch type count
    avg_by_count = merged.groupby("num_pitch_types")["era"].mean()
    ax.plot(avg_by_count.index, avg_by_count.values, color="#2ec4b6",
            linewidth=2.5, marker="o", markersize=8, alpha=0.8,
            label="Avg ERA by # Types", zorder=3)

    # Styling
    ax.set_xlabel("Number of Pitch Types", fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax.set_ylabel("ERA", fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax.set_title(f"MLB Starters: Arsenal Size vs ERA ({MLB_SEASON})",
                 fontsize=18, color="white", fontweight="bold", pad=15)

    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linestyle="--")
    ax.set_xticks(range(int(x_all.min()), int(x_all.max()) + 1))
    ax.invert_yaxis()  # Lower ERA = better = top

    legend = ax.legend(loc="upper right", fontsize=10, facecolor=BG_COLOR,
                       edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    legend.get_frame().set_alpha(0.8)

    # Footer
    fig.text(0.02, 0.01, "By: @BachTalk1", fontsize=9, color=MUTED_COLOR)
    fig.text(0.98, 0.01, f"Data: Baseball Savant | Min {MIN_IP} IP",
             fontsize=9, color=MUTED_COLOR, ha="right")

    _add_watermark(fig)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = SCREENSHOTS_DIR / "arsenal_size_vs_era.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 1: %s", path)
    return path


def plot_pitch_type_comparison(pitcher_df: pd.DataFrame, arsenal_df: pd.DataFrame) -> Path:
    """Chart 2: Reds starters vs league avg — run value per 100 by pitch type."""

    # Filter noise
    clean = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()

    rv_col = "run_value_per_100"
    if rv_col not in clean.columns:
        log.error("No run_value_per_100 column! Cols: %s", list(clean.columns))
        return None

    clean[rv_col] = pd.to_numeric(clean[rv_col], errors="coerce")

    # Identify Reds pitcher IDs from pitcher_df
    reds_ids = set(pitcher_df[pitcher_df["team"] == "CIN"]["player_id"])

    # Split into Reds vs league
    reds_arsenal = clean[clean["player_id"].isin(reds_ids)].copy()
    league_arsenal = clean[~clean["player_id"].isin(reds_ids)].copy()

    log.info("Reds pitchers in arsenal data: %d, league: %d",
             reds_arsenal["player_id"].nunique(), league_arsenal["player_id"].nunique())

    # Aggregate by pitch type
    reds_agg = reds_arsenal.groupby("pitch_type")[rv_col].mean().sort_index()
    league_agg = league_arsenal.groupby("pitch_type")[rv_col].mean().sort_index()

    # Get pitch types that Reds actually throw
    reds_types = set(reds_agg.index)
    # Include all common types for comparison
    common_types = sorted(reds_types & set(league_agg.index),
                          key=lambda x: reds_agg.get(x, 0))

    if not common_types:
        log.error("No common pitch types between Reds and league!")
        return None

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(common_types))
    bar_width = 0.35

    reds_vals = [reds_agg.get(pt, 0) for pt in common_types]
    league_vals = [league_agg.get(pt, 0) for pt in common_types]

    # Color each pitch type bar with its color
    reds_bars = ax.bar(x - bar_width/2, reds_vals, bar_width,
                       label="Reds Starters", edgecolor="white", linewidth=0.8, zorder=3)
    league_bars = ax.bar(x + bar_width/2, league_vals, bar_width,
                         label="MLB Avg", alpha=0.6, edgecolor="white", linewidth=0.8, zorder=3)

    # Color bars by pitch type
    for i, pt in enumerate(common_types):
        color = PITCH_COLORS.get(pt, "#888888")
        reds_bars[i].set_facecolor(color)
        league_bars[i].set_facecolor(color)
        league_bars[i].set_alpha(0.4)

    # Add value labels on bars
    for bars, vals in [(reds_bars, reds_vals), (league_bars, league_vals)]:
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height()
            offset = -0.15 if val > 0 else 0.15
            ax.text(bar.get_x() + bar.get_width()/2, y_pos + offset,
                    f"{val:.1f}", ha="center", va="bottom" if val < 0 else "top",
                    fontsize=9, color=TEXT_COLOR, fontweight="bold",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)])

    # Zero line
    ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5, zorder=2)

    # Labels
    pitch_labels = [PITCH_NAMES.get(pt, pt) for pt in common_types]
    ax.set_xticks(x)
    ax.set_xticklabels(pitch_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Run Value per 100 Pitches", fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax.set_title(f"Reds Starters vs MLB — Pitch Type Effectiveness ({MLB_SEASON})",
                 fontsize=18, color="white", fontweight="bold", pad=15)

    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3, linestyle="--")

    # Note: negative run value = better for pitcher
    ax.text(0.02, 0.97, "↓ Negative = Better for Pitcher",
            transform=ax.transAxes, fontsize=10, color="#2ec4b6",
            va="top", fontweight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)])

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REDS_COLOR, edgecolor="white", label="Reds Starters"),
        Patch(facecolor=LEAGUE_COLOR, alpha=0.4, edgecolor="white", label="MLB Avg"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
                       facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    legend.get_frame().set_alpha(0.8)

    # Footer
    fig.text(0.02, 0.01, "By: @BachTalk1", fontsize=9, color=MUTED_COLOR)
    fig.text(0.98, 0.01, f"Data: Baseball Savant | Min {MIN_IP} IP",
             fontsize=9, color=MUTED_COLOR, ha="right")

    _add_watermark(fig)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = SCREENSHOTS_DIR / "reds_vs_league_pitch_types.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 2: %s", path)
    return path


def plot_reds_pitcher_breakdown(pitcher_df: pd.DataFrame, arsenal_df: pd.DataFrame) -> Path:
    """Chart 3: Each Reds starter — pitch type usage (%) + RV/100 per pitch type."""

    clean = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()

    rv_col = "run_value_per_100"
    pct_col = "pitch_usage"
    if rv_col not in clean.columns or pct_col not in clean.columns:
        log.warning("Missing columns. Cols: %s", list(clean.columns))
        return None

    clean[rv_col] = pd.to_numeric(clean[rv_col], errors="coerce")
    clean[pct_col] = pd.to_numeric(clean[pct_col], errors="coerce")

    reds_ids = set(pitcher_df[pitcher_df["team"] == "CIN"]["player_id"])
    reds_data = clean[clean["player_id"].isin(reds_ids)].copy()

    if reds_data.empty:
        log.warning("No Reds data for breakdown chart")
        return None

    # Get Reds pitchers sorted by ERA
    reds_pitchers_df = pitcher_df[pitcher_df["player_id"].isin(reds_ids)].copy()
    reds_pitchers_df = reds_pitchers_df.sort_values("era")

    pitcher_ids = reds_pitchers_df["player_id"].tolist()
    n_pitchers = len(pitcher_ids)

    if n_pitchers == 0:
        return None

    # ── Plot: one row per pitcher ──
    fig, axes = plt.subplots(n_pitchers, 1, figsize=(14, 2.5 * n_pitchers + 2),
                             squeeze=False)
    fig.set_facecolor(BG_COLOR)

    for i, pid in enumerate(pitcher_ids):
        ax = axes[i, 0]
        ax.set_facecolor(BG_COLOR)

        p_data = reds_data[reds_data["player_id"] == pid].copy()
        if p_data.empty:
            continue

        # Get pitcher name from pitcher_df
        name_row = reds_pitchers_df[reds_pitchers_df["player_id"] == pid]
        if not name_row.empty:
            pname = name_row.iloc[0]["player_name"]
            p_era = f" (ERA: {name_row.iloc[0]['era']:.2f})"
        else:
            # Try arsenal data name
            name_val = p_data.iloc[0].get("last_name, first_name", "")
            if name_val and ", " in str(name_val):
                parts = str(name_val).split(", ")
                pname = f"{parts[1]} {parts[0]}"
            else:
                pname = f"Pitcher {pid}"
            p_era = ""

        # Sort by usage
        p_data = p_data.sort_values(pct_col, ascending=False)
        pitch_types = p_data["pitch_type"].tolist()
        usage = p_data[pct_col].tolist()
        rv = p_data[rv_col].tolist()

        # Horizontal bars for usage %
        colors = [PITCH_COLORS.get(pt, "#888888") for pt in pitch_types]
        labels = [f"{PITCH_NAMES.get(pt, pt)}" for pt in pitch_types]
        y_pos = np.arange(len(pitch_types))

        bars = ax.barh(y_pos, usage, color=colors, edgecolor="white",
                       linewidth=0.5, height=0.7, zorder=3)

        # Add usage % and RV/100 labels
        for j, (bar, u, r) in enumerate(zip(bars, usage, rv)):
            # Usage label inside bar
            u_pct = u * 100 if u < 1 else u  # handle if already percentage
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{u_pct:.1f}%  |  RV/100: {r:+.1f}",
                    va="center", fontsize=10, color=TEXT_COLOR, fontweight="bold",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.set_title(f"{pname}{p_era}", fontsize=14, color="white",
                     fontweight="bold", loc="left", pad=8)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.3, linestyle="--")
        ax.invert_yaxis()

    fig.suptitle(f"Reds Starters — Pitch Arsenal Breakdown ({MLB_SEASON})",
                 fontsize=20, color="white", fontweight="bold", y=0.99)

    fig.text(0.02, 0.005, "By: @BachTalk1", fontsize=9, color=MUTED_COLOR)
    fig.text(0.98, 0.005, f"Data: Baseball Savant | Min {MIN_IP} IP",
             fontsize=9, color=MUTED_COLOR, ha="right")

    _add_watermark(fig)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    path = SCREENSHOTS_DIR / "reds_starter_arsenal_breakdown.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 3: %s", path)
    return path


def plot_mean_pitch_types_vs_era(pitcher_df: pd.DataFrame, arsenal_df: pd.DataFrame) -> Path:
    """Chart 4: Reds mean # pitch types & ERA vs league mean, grouped by arsenal size."""

    # Count distinct pitch types per pitcher
    clean_arsenal = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()
    pitch_counts = clean_arsenal.groupby("player_id")["pitch_type"].nunique().reset_index()
    pitch_counts.columns = ["player_id", "num_pitch_types"]

    merged = pitcher_df.merge(pitch_counts, on="player_id", how="inner")
    merged = merged.dropna(subset=["era", "num_pitch_types"])
    merged = merged[merged["era"] <= 8.0]

    reds = merged[merged["team"] == "CIN"].copy()
    league = merged[merged["team"] != "CIN"].copy()

    # ── Overall means ──
    reds_mean_types = reds["num_pitch_types"].mean()
    reds_mean_era = reds["era"].mean()
    league_mean_types = league["num_pitch_types"].mean()
    league_mean_era = league["era"].mean()

    log.info("Reds mean: %.1f types, %.2f ERA | League mean: %.1f types, %.2f ERA",
             reds_mean_types, reds_mean_era, league_mean_types, league_mean_era)

    # ── Group by # pitch types: avg ERA for Reds vs league ──
    league_by_count = league.groupby("num_pitch_types").agg(
        era_mean=("era", "mean"), count=("era", "count")
    ).reset_index()
    reds_by_count = reds.groupby("num_pitch_types").agg(
        era_mean=("era", "mean"), count=("era", "count")
    ).reset_index()

    # All pitch type counts that exist
    all_counts = sorted(set(league_by_count["num_pitch_types"]) | set(reds_by_count["num_pitch_types"]))

    # ── Plot ──
    fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(16, 8),
                                           gridspec_kw={"width_ratios": [1, 1.3]})
    fig.set_facecolor(BG_COLOR)

    # ── Left panel: Overall comparison (big numbers) ──
    ax_main.set_facecolor(BG_COLOR)
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 10)
    ax_main.axis("off")

    # Title
    ax_main.text(5, 9.3, "Mean Arsenal Size & ERA", fontsize=18, color="white",
                 fontweight="bold", ha="center", va="center")

    # Reds box
    reds_box = plt.Rectangle((0.5, 4.5), 4, 4.2, facecolor=REDS_COLOR, alpha=0.15,
                              edgecolor=REDS_COLOR, linewidth=2, transform=ax_main.transData)
    ax_main.add_patch(reds_box)
    ax_main.text(2.5, 8.2, "REDS", fontsize=16, color=REDS_COLOR,
                 fontweight="bold", ha="center")
    ax_main.text(2.5, 7.0, f"{reds_mean_types:.1f}", fontsize=42, color="white",
                 fontweight="bold", ha="center", va="center")
    ax_main.text(2.5, 6.0, "pitch types", fontsize=12, color=MUTED_COLOR,
                 ha="center")
    ax_main.text(2.5, 5.1, f"{reds_mean_era:.2f} ERA", fontsize=22, color=REDS_COLOR,
                 fontweight="bold", ha="center")

    # League box
    league_box = plt.Rectangle((5.5, 4.5), 4, 4.2, facecolor=LEAGUE_COLOR, alpha=0.15,
                                edgecolor=LEAGUE_COLOR, linewidth=2, transform=ax_main.transData)
    ax_main.add_patch(league_box)
    ax_main.text(7.5, 8.2, "MLB", fontsize=16, color=LEAGUE_COLOR,
                 fontweight="bold", ha="center")
    ax_main.text(7.5, 7.0, f"{league_mean_types:.1f}", fontsize=42, color="white",
                 fontweight="bold", ha="center", va="center")
    ax_main.text(7.5, 6.0, "pitch types", fontsize=12, color=MUTED_COLOR,
                 ha="center")
    ax_main.text(7.5, 5.1, f"{league_mean_era:.2f} ERA", fontsize=22, color=LEAGUE_COLOR,
                 fontweight="bold", ha="center")

    # Difference callout
    era_diff = reds_mean_era - league_mean_era
    diff_color = "#d62828" if era_diff > 0 else "#2ec4b6"
    diff_sign = "+" if era_diff > 0 else ""
    ax_main.text(5, 3.5, f"Reds {diff_sign}{era_diff:.2f} ERA vs League",
                 fontsize=14, color=diff_color, fontweight="bold", ha="center")

    type_diff = reds_mean_types - league_mean_types
    type_sign = "+" if type_diff > 0 else ""
    ax_main.text(5, 2.7, f"{type_sign}{type_diff:.1f} pitch types vs League",
                 fontsize=12, color=MUTED_COLOR, ha="center")

    # Individual Reds pitchers listed
    ax_main.text(5, 1.6, "Reds Starters:", fontsize=11, color=MUTED_COLOR,
                 ha="center", fontstyle="italic")
    reds_sorted = reds.sort_values("era")
    y_pos = 1.0
    for _, row in reds_sorted.iterrows():
        ax_main.text(5, y_pos,
                     f"{row['player_name']}  —  {int(row['num_pitch_types'])} types, {row['era']:.2f} ERA",
                     fontsize=9, color=TEXT_COLOR, ha="center", family="monospace")
        y_pos -= 0.5

    # ── Right panel: Grouped bar — ERA by # pitch types ──
    ax_bar.set_facecolor(BG_COLOR)

    x = np.arange(len(all_counts))
    bar_width = 0.35

    # Get values (NaN where a group doesn't exist)
    reds_eras = []
    league_eras = []
    reds_ns = []
    league_ns = []
    for c in all_counts:
        r_row = reds_by_count[reds_by_count["num_pitch_types"] == c]
        l_row = league_by_count[league_by_count["num_pitch_types"] == c]
        reds_eras.append(r_row["era_mean"].iloc[0] if not r_row.empty else None)
        league_eras.append(l_row["era_mean"].iloc[0] if not l_row.empty else None)
        reds_ns.append(int(r_row["count"].iloc[0]) if not r_row.empty else 0)
        league_ns.append(int(l_row["count"].iloc[0]) if not l_row.empty else 0)

    # Plot league bars
    for i, (era, n) in enumerate(zip(league_eras, league_ns)):
        if era is not None:
            bar = ax_bar.bar(x[i] + bar_width/2, era, bar_width,
                             color=LEAGUE_COLOR, alpha=0.5, edgecolor="white",
                             linewidth=0.8, zorder=3)
            ax_bar.text(x[i] + bar_width/2, era + 0.1, f"{era:.2f}",
                        ha="center", fontsize=10, color=LEAGUE_COLOR, fontweight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)])
            ax_bar.text(x[i] + bar_width/2, 0.15, f"n={n}",
                        ha="center", fontsize=8, color=MUTED_COLOR)

    # Plot Reds bars
    for i, (era, n) in enumerate(zip(reds_eras, reds_ns)):
        if era is not None:
            bar = ax_bar.bar(x[i] - bar_width/2, era, bar_width,
                             color=REDS_COLOR, edgecolor="white",
                             linewidth=0.8, zorder=3)
            ax_bar.text(x[i] - bar_width/2, era + 0.1, f"{era:.2f}",
                        ha="center", fontsize=10, color=REDS_COLOR, fontweight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_COLOR)])
            ax_bar.text(x[i] - bar_width/2, 0.15, f"n={n}",
                        ha="center", fontsize=8, color=MUTED_COLOR)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{c} types" for c in all_counts], fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("ERA", fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax_bar.set_title("Avg ERA by Arsenal Size", fontsize=16, color="white",
                     fontweight="bold", pad=10)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=11)
    for spine in ax_bar.spines.values():
        spine.set_color(GRID_COLOR)
    ax_bar.grid(True, axis="y", color=GRID_COLOR, alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REDS_COLOR, edgecolor="white", label="Reds"),
        Patch(facecolor=LEAGUE_COLOR, alpha=0.5, edgecolor="white", label="MLB"),
    ]
    legend = ax_bar.legend(handles=legend_elements, loc="upper right", fontsize=11,
                           facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    legend.get_frame().set_alpha(0.8)

    # Footer
    fig.text(0.02, 0.01, "By: @BachTalk1", fontsize=9, color=MUTED_COLOR)
    fig.text(0.98, 0.01, f"Data: Baseball Savant + MLB Stats API | Min {MIN_IP} IP",
             fontsize=9, color=MUTED_COLOR, ha="right")

    fig.suptitle(f"Reds vs MLB Starters — Pitch Type Count & ERA ({MLB_SEASON})",
                 fontsize=20, color="white", fontweight="bold", y=1.01)

    _add_watermark(fig)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = SCREENSHOTS_DIR / "reds_mean_pitch_types_vs_era.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 4: %s", path)
    return path


def plot_reds_arsenal_post(pitcher_df: pd.DataFrame, arsenal_df: pd.DataFrame,
                          velo_df: pd.DataFrame = None) -> Path:
    """Post-ready card: Reds starters vs MLB — arsenal size, ERA, FIP, K%, BB%, WHIP."""

    # Use velocity data for pitch type counts (most complete inventory)
    if velo_df is not None and not velo_df.empty:
        pitch_counts = velo_df[["player_id", "num_pitch_types"]].copy()
        # Build pitch type name lookup from velocity data
        pitcher_pitches = {row["player_id"]: row["pitch_types"] for _, row in velo_df.iterrows()}
    else:
        # Fallback to arsenal stats
        clean_arsenal = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()
        pitch_counts = clean_arsenal.groupby("player_id")["pitch_type"].nunique().reset_index()
        pitch_counts.columns = ["player_id", "num_pitch_types"]
        pitcher_pitches = clean_arsenal.groupby("player_id")["pitch_type"].apply(list).to_dict()

    merged = pitcher_df.merge(pitch_counts, on="player_id", how="inner")
    merged = merged.dropna(subset=["era", "num_pitch_types"])
    merged = merged[merged["era"] <= 10.0]

    reds = merged[merged["team"] == "CIN"].copy().sort_values("era")
    league = merged[merged["team"] != "CIN"].copy()

    # ── Aggregate stats ──
    stat_cols = ["era", "fip", "whip", "k_pct", "bb_pct", "k_bb", "k_per_9", "bb_per_9", "hr_per_9"]
    reds_means = {c: reds[c].mean() for c in stat_cols}
    reds_means["num_pitch_types"] = reds["num_pitch_types"].mean()
    reds_means["ip"] = reds["ip"].sum()
    league_means = {c: league[c].mean() for c in stat_cols}
    league_means["num_pitch_types"] = league["num_pitch_types"].mean()
    league_means["ip"] = league["ip"].sum()

    # ── Build the card ──
    fig = plt.figure(figsize=(14, 20))
    fig.set_facecolor(SURFACE_COLOR)

    # No axes — draw everything manually
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(SURFACE_COLOR)

    # ── Title banner ──
    ax.fill_between([0, 100], 95, 100, color=REDS_COLOR, zorder=2)
    ax.text(50, 97.5, f"REDS STARTING PITCHING ARSENAL REPORT — {MLB_SEASON}",
            fontsize=18, color="white", fontweight="bold", ha="center", va="center", zorder=3)

    # ── Section 1: Reds vs MLB Summary ──
    ax.text(50, 93.5, "REDS STARTERS vs MLB AVERAGE", fontsize=14, color="white",
            fontweight="bold", ha="center", va="center")

    # Stat comparison table
    compare_stats = [
        ("Arsenal", f"{reds_means['num_pitch_types']:.1f} types", f"{league_means['num_pitch_types']:.1f} types"),
        ("ERA", f"{reds_means['era']:.2f}", f"{league_means['era']:.2f}"),
        ("FIP", f"{reds_means['fip']:.2f}", f"{league_means['fip']:.2f}"),
        ("WHIP", f"{reds_means['whip']:.2f}", f"{league_means['whip']:.2f}"),
        ("K%", f"{reds_means['k_pct']:.1f}%", f"{league_means['k_pct']:.1f}%"),
        ("BB%", f"{reds_means['bb_pct']:.1f}%", f"{league_means['bb_pct']:.1f}%"),
        ("K-BB%", f"{reds_means['k_bb']:.1f}%", f"{league_means['k_bb']:.1f}%"),
        ("K/9", f"{reds_means['k_per_9']:.2f}", f"{league_means['k_per_9']:.2f}"),
        ("HR/9", f"{reds_means['hr_per_9']:.2f}", f"{league_means['hr_per_9']:.2f}"),
    ]

    # Table header
    y_start = 91
    ax.text(30, y_start, "STAT", fontsize=10, color=MUTED_COLOR,
            fontweight="bold", ha="center", va="center")
    ax.text(55, y_start, "REDS", fontsize=10, color=REDS_COLOR,
            fontweight="bold", ha="center", va="center")
    ax.text(75, y_start, "MLB", fontsize=10, color=LEAGUE_COLOR,
            fontweight="bold", ha="center", va="center")
    ax.plot([15, 90], [y_start - 0.6, y_start - 0.6], color=GRID_COLOR, linewidth=0.8)

    # "Better" direction: lower is better for ERA, FIP, WHIP, BB%, HR/9; higher for K%, K-BB%, K/9
    higher_better = {"K%", "K-BB%", "K/9", "Arsenal"}
    lower_better = {"ERA", "FIP", "WHIP", "BB%", "HR/9"}

    for i, (label, reds_val, mlb_val) in enumerate(compare_stats):
        y = y_start - 1.5 - i * 2.0
        # Alternating row background
        if i % 2 == 0:
            ax.fill_between([15, 90], y - 0.8, y + 0.8, color="#1a1f2e", alpha=0.5)

        ax.text(30, y, label, fontsize=11, color=TEXT_COLOR,
                fontweight="bold", ha="center", va="center", family="monospace")

        # Color code: green if Reds better, red if worse
        try:
            rv = float(reds_val.replace("%", "").replace(" types", ""))
            mv = float(mlb_val.replace("%", "").replace(" types", ""))
            if label in higher_better:
                reds_color = "#2ec4b6" if rv > mv else "#d62828" if rv < mv else TEXT_COLOR
            elif label in lower_better:
                reds_color = "#2ec4b6" if rv < mv else "#d62828" if rv > mv else TEXT_COLOR
            else:
                reds_color = TEXT_COLOR
        except ValueError:
            reds_color = TEXT_COLOR

        ax.text(55, y, reds_val, fontsize=12, color=reds_color,
                fontweight="bold", ha="center", va="center", family="monospace")
        ax.text(75, y, mlb_val, fontsize=12, color=MUTED_COLOR,
                ha="center", va="center", family="monospace")

    # ── Section 2: Individual Reds Starters ──
    section2_y = y_start - 1.5 - len(compare_stats) * 2.0 - 2.5
    ax.plot([5, 95], [section2_y + 1.5, section2_y + 1.5], color=REDS_COLOR, linewidth=2)
    ax.text(50, section2_y + 2.5, "INDIVIDUAL STARTER BREAKDOWN",
            fontsize=14, color="white", fontweight="bold", ha="center", va="center")

    # Column headers
    headers = ["PITCHER", "#PT", "ERA", "FIP", "WHIP", "K%", "BB%", "K-BB%", "ARSENAL"]
    header_x = [12, 25, 32, 39, 46, 53, 60, 68, 85]
    for hx, ht in zip(header_x, headers):
        ax.text(hx, section2_y, ht, fontsize=9, color=MUTED_COLOR,
                fontweight="bold", ha="center", va="center")
    ax.plot([5, 95], [section2_y - 0.6, section2_y - 0.6], color=GRID_COLOR, linewidth=0.8)

    for i, (_, row) in enumerate(reds.iterrows()):
        y = section2_y - 1.8 - i * 3.8
        pid = row["player_id"]

        # Alternating row background
        if i % 2 == 0:
            ax.fill_between([5, 95], y - 1.3, y + 1.0, color="#1a1f2e", alpha=0.5)

        # Name
        ax.text(12, y, row["player_name"], fontsize=11, color="white",
                fontweight="bold", ha="center", va="center")
        # IP under name
        ax.text(12, y - 0.9, f"{row['ip']:.1f} IP, {int(row['gs'])} GS",
                fontsize=8, color=MUTED_COLOR, ha="center", va="center")

        # Stats
        n_types = int(row["num_pitch_types"])
        vals = [
            str(n_types),
            f"{row['era']:.2f}",
            f"{row['fip']:.2f}",
            f"{row['whip']:.2f}",
            f"{row['k_pct']:.1f}",
            f"{row['bb_pct']:.1f}",
            f"{row['k_bb']:.1f}",
        ]
        val_x = header_x[1:8]

        for vx, v, stat_label in zip(val_x, vals, headers[1:8]):
            # Color code vs league mean
            try:
                fv = float(v)
                stat_key_map = {"ERA": "era", "FIP": "fip", "WHIP": "whip",
                                "K%": "k_pct", "BB%": "bb_pct", "K-BB%": "k_bb"}
                lk = stat_key_map.get(stat_label)
                if lk and lk in league_means:
                    lm = league_means[lk]
                    if stat_label in ("K%", "K-BB%"):
                        color = "#2ec4b6" if fv > lm else "#d62828" if fv < lm * 0.85 else "#ffbe0b"
                    elif stat_label in ("ERA", "FIP", "WHIP", "BB%"):
                        color = "#2ec4b6" if fv < lm else "#d62828" if fv > lm * 1.15 else "#ffbe0b"
                    else:
                        color = TEXT_COLOR
                else:
                    color = TEXT_COLOR
            except ValueError:
                color = TEXT_COLOR

            ax.text(vx, y, v, fontsize=11, color=color,
                    fontweight="bold", ha="center", va="center", family="monospace")

        # Pitch type arsenal (colored dots/labels)
        pitches = pitcher_pitches.get(pid, [])
        arsenal_x_start = 78
        for j, pt in enumerate(sorted(pitches)):
            px = arsenal_x_start + j * 4.5
            color = PITCH_COLORS.get(pt, "#888888")
            ax.text(px, y, PITCH_NAMES.get(pt, pt), fontsize=8, color=color,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=0.8))

    # ── Section 3: ERA & FIP by Arsenal Size ──
    bar_y_top = section2_y - 1.8 - len(reds) * 3.8 - 2.0
    ax.plot([5, 95], [bar_y_top + 1.0, bar_y_top + 1.0], color=REDS_COLOR, linewidth=2)
    ax.text(50, bar_y_top + 2.0, "ERA & FIP BY ARSENAL SIZE",
            fontsize=14, color="white", fontweight="bold", ha="center", va="center")

    # Group by arsenal size
    league_by_count = league.groupby("num_pitch_types").agg(
        era_mean=("era", "mean"), fip_mean=("fip", "mean"), count=("era", "count")
    ).reset_index()
    reds_by_count = reds.groupby("num_pitch_types").agg(
        era_mean=("era", "mean"), fip_mean=("fip", "mean"), count=("era", "count")
    ).reset_index()
    all_counts = sorted(set(league_by_count["num_pitch_types"]) | set(reds_by_count["num_pitch_types"]))

    # Mini embedded axes for the bar chart
    bar_ax = fig.add_axes([0.10, 0.03, 0.80, (bar_y_top - 1) / 100 * 0.75])
    bar_ax.set_facecolor(SURFACE_COLOR)

    x = np.arange(len(all_counts))
    bw = 0.18  # narrower bars for 4 per group

    for i, c in enumerate(all_counts):
        l_row = league_by_count[league_by_count["num_pitch_types"] == c]
        r_row = reds_by_count[reds_by_count["num_pitch_types"] == c]

        if not l_row.empty:
            era = l_row["era_mean"].iloc[0]
            fip = l_row["fip_mean"].iloc[0]
            n = int(l_row["count"].iloc[0])
            # MLB ERA bar
            bar_ax.bar(x[i] + bw * 0.5, era, bw, color=LEAGUE_COLOR, alpha=0.5,
                       edgecolor="white", linewidth=0.8, zorder=3)
            bar_ax.text(x[i] + bw * 0.5, era + 0.12, f"{era:.2f}", ha="center",
                        fontsize=8, color=LEAGUE_COLOR, fontweight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=SURFACE_COLOR)])
            # MLB FIP bar
            bar_ax.bar(x[i] + bw * 1.5, fip, bw, color=LEAGUE_COLOR, alpha=0.25,
                       edgecolor=LEAGUE_COLOR, linewidth=0.8, linestyle="--", zorder=3)
            bar_ax.text(x[i] + bw * 1.5, fip + 0.12, f"{fip:.2f}", ha="center",
                        fontsize=8, color=LEAGUE_COLOR, alpha=0.7,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=SURFACE_COLOR)])
            bar_ax.text(x[i] + bw, -0.35, f"n={n}", ha="center",
                        fontsize=7, color=MUTED_COLOR)

        if not r_row.empty:
            era = r_row["era_mean"].iloc[0]
            fip = r_row["fip_mean"].iloc[0]
            n = int(r_row["count"].iloc[0])
            # Reds ERA bar
            bar_ax.bar(x[i] - bw * 1.5, era, bw, color=REDS_COLOR,
                       edgecolor="white", linewidth=0.8, zorder=3)
            bar_ax.text(x[i] - bw * 1.5, era + 0.12, f"{era:.2f}", ha="center",
                        fontsize=8, color=REDS_COLOR, fontweight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=SURFACE_COLOR)])
            # Reds FIP bar
            bar_ax.bar(x[i] - bw * 0.5, fip, bw, color="#ffbe0b",
                       edgecolor="white", linewidth=0.8, zorder=3, alpha=0.7)
            bar_ax.text(x[i] - bw * 0.5, fip + 0.12, f"{fip:.2f}", ha="center",
                        fontsize=8, color="#ffbe0b",
                        path_effects=[patheffects.withStroke(linewidth=2, foreground=SURFACE_COLOR)])
            bar_ax.text(x[i] - bw, -0.35, f"n={n}", ha="center",
                        fontsize=7, color=MUTED_COLOR)

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([f"{c} Pitch Types" for c in all_counts], fontsize=11, fontweight="bold")
    bar_ax.set_ylabel("ERA / FIP", fontsize=12, color=TEXT_COLOR, fontweight="bold")
    bar_ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in bar_ax.spines.values():
        spine.set_color(GRID_COLOR)
    bar_ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.3, linestyle="--")
    bar_ax.set_ylim(bottom=-0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REDS_COLOR, edgecolor="white", label="Reds ERA"),
        Patch(facecolor="#ffbe0b", alpha=0.7, edgecolor="white", label="Reds FIP"),
        Patch(facecolor=LEAGUE_COLOR, alpha=0.5, edgecolor="white", label="MLB ERA"),
        Patch(facecolor=LEAGUE_COLOR, alpha=0.25, edgecolor=LEAGUE_COLOR, label="MLB FIP"),
    ]
    legend = bar_ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
                           facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                           ncol=2)
    legend.get_frame().set_alpha(0.8)

    # Footer
    ax.text(2, 0.5, "By: @BachTalk1", fontsize=9, color=MUTED_COLOR, va="bottom")
    ax.text(98, 0.5, f"Data: Baseball Savant + MLB Stats API | Min {MIN_IP} IP",
            fontsize=9, color=MUTED_COLOR, ha="right", va="bottom")

    _add_watermark(fig)

    path = SCREENSHOTS_DIR / "reds_arsenal_report.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved post card: %s", path)
    return path


def plot_pitcher_pitch_card(pitcher_row: pd.Series, arsenal_df: pd.DataFrame,
                           league_arsenal: pd.DataFrame) -> Path | None:
    """Generate a per-pitcher card showing pitch type stats with ERA & FIP."""

    pid = int(pitcher_row["player_id"])
    name = pitcher_row["player_name"]
    era = pitcher_row["era"]
    fip = pitcher_row["fip"]
    ip = pitcher_row["ip"]
    gs = int(pitcher_row["gs"])
    whip = pitcher_row["whip"]
    k_pct = pitcher_row["k_pct"]
    bb_pct = pitcher_row["bb_pct"]
    k_bb = pitcher_row["k_bb"]

    # Get this pitcher's pitch types from arsenal data
    p_arsenal = arsenal_df[
        (arsenal_df["player_id"] == pid) & (~arsenal_df["pitch_type"].isin(NOISE_PITCHES))
    ].copy()

    if p_arsenal.empty:
        log.warning("No arsenal data for %s", name)
        return None

    p_arsenal = p_arsenal.sort_values("pitch_usage", ascending=False)

    # League averages per pitch type for comparison
    league_avgs = league_arsenal.groupby("pitch_type").agg({
        "run_value_per_100": "mean", "woba": "mean", "whiff_percent": "mean",
        "k_percent": "mean", "hard_hit_percent": "mean", "pitch_usage": "mean",
    }).to_dict("index")

    # ── Build card ──
    n_pitches = len(p_arsenal)
    fig_h = max(4.5, 2.8 + n_pitches * 0.9)
    fig = plt.figure(figsize=(12, fig_h))
    fig.set_facecolor(SURFACE_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(SURFACE_COLOR)

    # ── Header ──
    ax.fill_between([0, 100], 90, 100, color=REDS_COLOR, zorder=2)
    ax.text(50, 95, name.upper(), fontsize=20, color="white",
            fontweight="bold", ha="center", va="center", zorder=3)

    # Stats row under name
    stats_y = 86
    stat_items = [
        (f"{era:.2f}", "ERA"), (f"{fip:.2f}", "FIP"), (f"{whip:.2f}", "WHIP"),
        (f"{k_pct:.1f}%", "K%"), (f"{bb_pct:.1f}%", "BB%"), (f"{k_bb:+.1f}%", "K-BB%"),
        (f"{ip:.1f}", "IP"), (f"{gs}", "GS"),
    ]
    stat_spacing = 100 / (len(stat_items) + 1)
    for j, (val, label) in enumerate(stat_items):
        sx = stat_spacing * (j + 1)
        ax.text(sx, stats_y + 1.5, val, fontsize=14, color="white",
                fontweight="bold", ha="center", va="center")
        ax.text(sx, stats_y - 1.5, label, fontsize=9, color=MUTED_COLOR,
                ha="center", va="center")

    # ── Pitch type table ──
    table_y = 80
    headers = ["PITCH", "USAGE", "VELO", "WHIFF%", "K%", "wOBA", "HH%", "RV/100"]
    hdr_x = [8, 20, 30, 41, 52, 63, 74, 88]

    ax.plot([3, 97], [table_y + 0.5, table_y + 0.5], color=GRID_COLOR, linewidth=1)
    for hx, ht in zip(hdr_x, headers):
        ax.text(hx, table_y + 2, ht, fontsize=9, color=MUTED_COLOR,
                fontweight="bold", ha="center", va="center")

    # Get velocity from arsenal data or pitch_name
    for i, (_, prow) in enumerate(p_arsenal.iterrows()):
        pt = prow["pitch_type"]
        y = table_y - 2 - i * (70 / max(n_pitches, 1))
        color = PITCH_COLORS.get(pt, "#888888")
        pt_name = PITCH_NAMES.get(pt, pt)

        # Alternating row bg
        if i % 2 == 0:
            ax.fill_between([3, 97], y - (30 / max(n_pitches, 1)), y + (30 / max(n_pitches, 1)),
                            color="#1a1f2e", alpha=0.5)

        # Pitch name with color dot
        ax.text(2, y, "\u25cf", fontsize=14, color=color, ha="center", va="center")
        ax.text(8, y, pt_name, fontsize=11, color="white", fontweight="bold",
                ha="center", va="center")

        # Usage
        usage = prow.get("pitch_usage", 0)
        usage_pct = usage * 100 if usage < 1 else usage
        ax.text(20, y, f"{usage_pct:.1f}%", fontsize=11, color=TEXT_COLOR,
                ha="center", va="center", family="monospace")

        # Velocity — try to get from pitch name or estimate
        # We don't have velo in arsenal-stats, show "-"
        ax.text(30, y, "—", fontsize=11, color=MUTED_COLOR,
                ha="center", va="center", family="monospace")

        # Whiff%
        whiff = prow.get("whiff_percent", 0)
        whiff = float(whiff) if pd.notna(whiff) else 0
        lg_whiff = league_avgs.get(pt, {}).get("whiff_percent", whiff)
        w_color = "#2ec4b6" if whiff > lg_whiff else "#d62828" if whiff < lg_whiff * 0.8 else TEXT_COLOR
        ax.text(41, y, f"{whiff:.1f}%", fontsize=11, color=w_color,
                fontweight="bold", ha="center", va="center", family="monospace")

        # K%
        kp = prow.get("k_percent", 0)
        kp = float(kp) if pd.notna(kp) else 0
        lg_kp = league_avgs.get(pt, {}).get("k_percent", kp)
        k_color = "#2ec4b6" if kp > lg_kp else "#d62828" if kp < lg_kp * 0.8 else TEXT_COLOR
        ax.text(52, y, f"{kp:.1f}%", fontsize=11, color=k_color,
                fontweight="bold", ha="center", va="center", family="monospace")

        # wOBA
        woba = prow.get("woba", 0)
        woba = float(woba) if pd.notna(woba) else 0
        lg_woba = league_avgs.get(pt, {}).get("woba", woba)
        woba_color = "#2ec4b6" if woba < lg_woba else "#d62828" if woba > lg_woba * 1.1 else TEXT_COLOR
        ax.text(63, y, f".{int(woba*1000):03d}" if woba > 0 else "—", fontsize=11,
                color=woba_color, fontweight="bold", ha="center", va="center", family="monospace")

        # Hard Hit%
        hh = prow.get("hard_hit_percent", 0)
        hh = float(hh) if pd.notna(hh) else 0
        lg_hh = league_avgs.get(pt, {}).get("hard_hit_percent", hh)
        hh_color = "#2ec4b6" if hh < lg_hh else "#d62828" if hh > lg_hh * 1.1 else TEXT_COLOR
        ax.text(74, y, f"{hh:.1f}%" if hh > 0 else "—", fontsize=11,
                color=hh_color, fontweight="bold", ha="center", va="center", family="monospace")

        # RV/100
        rv = prow.get("run_value_per_100", 0)
        rv = float(rv) if pd.notna(rv) else 0
        rv_color = "#2ec4b6" if rv < 0 else "#d62828" if rv > 0.5 else "#ffbe0b"
        ax.text(88, y, f"{rv:+.1f}", fontsize=12, color=rv_color,
                fontweight="bold", ha="center", va="center", family="monospace")

        # League avg underneath in small text
        lg_rv = league_avgs.get(pt, {}).get("run_value_per_100", 0)
        ax.text(88, y - (18 / max(n_pitches, 1)), f"lg: {lg_rv:+.1f}", fontsize=7,
                color=MUTED_COLOR, ha="center", va="center")

    # Color legend note
    ax.text(50, 3, "Green = better than league avg  |  Red = worse than league avg",
            fontsize=8, color=MUTED_COLOR, ha="center", va="center", fontstyle="italic")

    # Footer
    ax.text(2, 0.5, "By: @BachTalk1", fontsize=8, color=MUTED_COLOR, va="bottom")
    ax.text(98, 0.5, "Data: Baseball Savant", fontsize=8, color=MUTED_COLOR,
            ha="right", va="bottom")

    _add_watermark(fig, size_pct=0.25, alpha=0.04)

    safe_name = name.replace(" ", "_").lower()
    path = SCREENSHOTS_DIR / f"pitcher_arsenal_{safe_name}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved pitcher card: %s", path)
    return path


def fetch_pitcher_clips(reds_df: pd.DataFrame) -> dict[int, Path]:
    """Fetch Film Room strikeout clips for each Reds starter."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.video_clips import get_pitcher_clip

    clips = {}
    for _, row in reds_df.iterrows():
        pid = int(row["player_id"])
        name = row["player_name"]
        log.info("Fetching video clip for %s (pid=%d)...", name, pid)
        clip_path = get_pitcher_clip(pid, name)
        if clip_path:
            clips[pid] = clip_path
            log.info("  Got clip: %s", clip_path)
        else:
            log.info("  No clip found for %s", name)
    return clips


def post_thread(card_path: Path, reds_df: pd.DataFrame, clips: dict[int, Path],
                pitcher_cards: dict[int, Path], dry_run: bool = False):
    """Post thread: main tweet with card, then pitcher card + video replies."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    tweet_text = (
        "Reds starters need to develop more pitches to be effective. "
        "Full arsenal breakdown vs the league \U0001f9f5\u2b07\ufe0f\n\n"
        "@TJStats #Reds #MLB #Statcast"
    )

    if dry_run:
        print(f"[DRY RUN] Main tweet (len={len(tweet_text)}):")
        print(tweet_text.encode("ascii", "replace").decode())
        print(f"[DRY RUN] Image: {card_path}")
        for _, row in reds_df.iterrows():
            pid = int(row["player_id"])
            pcard = pitcher_cards.get(pid)
            clip = clips.get(pid)
            print(f"[DRY RUN] Reply: {row['player_name']} card={pcard is not None} video={clip is not None}")
        return

    from src.poster import post_with_image, post_reply, post_video_reply

    # Main tweet with card image
    tweet_id = post_with_image(tweet_text, card_path,
                               alt_text="Reds Starting Pitching Arsenal Report 2026")
    print(f"Posted main tweet: https://x.com/BachTalk1/status/{tweet_id}")

    # Per-pitcher replies: card image then video
    reply_to_id = tweet_id
    for _, row in reds_df.iterrows():
        pid = int(row["player_id"])
        name = row["player_name"]
        era = row["era"]
        fip = row["fip"]
        n_types = int(row["num_pitch_types"])

        # Post pitcher card
        pcard = pitcher_cards.get(pid)
        if pcard:
            reply_text = f"{name} \u2014 {era:.2f} ERA / {fip:.2f} FIP | {n_types} pitch types"
            try:
                rid = post_reply(reply_text, in_reply_to=reply_to_id,
                                 image_path=pcard,
                                 alt_text=f"{name} pitch type breakdown")
                print(f"  Card: {name} \u2014 https://x.com/BachTalk1/status/{rid}")
                reply_to_id = rid
            except Exception:
                log.warning("Failed to post card for %s", name, exc_info=True)

        # Post video clip
        clip = clips.get(pid)
        if clip:
            try:
                rid = post_video_reply(reply_to_id, clip, text=f"\U0001f3ac {name} highlight")
                print(f"  Video: {name} \u2014 https://x.com/BachTalk1/status/{rid}")
                reply_to_id = rid
            except Exception:
                log.warning("Failed to post video for %s", name, exc_info=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't post, just preview")
    parser.add_argument("--no-video", action="store_true", help="Skip video clips")
    args = parser.parse_args()

    pitcher_df = fetch_pitcher_stats()
    arsenal_df = fetch_arsenal_stats()
    velo_df = fetch_pitch_velocities(arsenal_df=arsenal_df)

    log.info("Pitcher columns: %s", list(pitcher_df.columns))
    log.info("Arsenal columns: %s", list(arsenal_df.columns))
    log.info("Velo pitch counts: %d pitchers", len(velo_df))

    # Generate card
    card_path = plot_reds_arsenal_post(pitcher_df, arsenal_df, velo_df=velo_df)
    if not card_path:
        print("Card generation failed!")
        return

    # Get Reds starters for clips
    pitch_counts = velo_df[["player_id", "num_pitch_types"]].copy()
    merged = pitcher_df.merge(pitch_counts, on="player_id", how="inner")
    reds = merged[merged["team"] == "CIN"].copy().sort_values("era")

    # Generate per-pitcher cards
    clean_arsenal = arsenal_df[~arsenal_df["pitch_type"].isin(NOISE_PITCHES)].copy()
    league_ids = set(pitcher_df[pitcher_df["team"] != "CIN"]["player_id"])
    league_arsenal = clean_arsenal[clean_arsenal["player_id"].isin(league_ids)]

    pitcher_cards = {}
    for _, row in reds.iterrows():
        pid = int(row["player_id"])
        pcard = plot_pitcher_pitch_card(row, arsenal_df, league_arsenal)
        if pcard:
            pitcher_cards[pid] = pcard
    print(f"\nGenerated {len(pitcher_cards)}/{len(reds)} pitcher cards")

    # Fetch video clips
    clips = {}
    if not args.no_video:
        clips = fetch_pitcher_clips(reds)
        print(f"Got {len(clips)}/{len(reds)} video clips")

    print(f"\n=== Card: {card_path} ===")

    # Post
    post_thread(card_path, reds, clips, pitcher_cards, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
