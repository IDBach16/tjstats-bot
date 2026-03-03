"""Chart generation module — locally-rendered Statcast visuals for tweets.

All public functions return ``Path | None``: the path to a saved PNG inside
``SCREENSHOTS_DIR``, or *None* on any failure.  Charts use the ``Agg``
backend so they work headlessly on GitHub Actions.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Rectangle # noqa: E402
from pybaseball import statcast_pitcher  # noqa: E402

from .config import MLB_SEASON, SCREENSHOTS_DIR

log = logging.getLogger(__name__)

# ── Theme ────────────────────────────────────────────────────────────────
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
FIG_W, FIG_H = 12, 6.75  # 1200×675 at 100 dpi — Twitter-optimal 16:9

# Pitch-type colour palette (matches R Shiny app)
PITCH_COLORS: dict[str, str] = {
    "FF": "#d62828",   # Four-Seam — red
    "SI": "#f77f00",   # Sinker — orange
    "FC": "#8338ec",   # Cutter — purple
    "SL": "#3a86ff",   # Slider — blue
    "SV": "#00b4d8",   # Sweeper — cyan
    "ST": "#00b4d8",   # Sweeper alt code
    "CU": "#2ec4b6",   # Curveball — teal
    "KC": "#06d6a0",   # Knuckle Curve — green
    "CH": "#ffbe0b",   # Changeup — yellow
    "FS": "#fb5607",   # Splitter — bright orange
    "KN": "#9d4edd",   # Knuckleball — violet
}
DEFAULT_PITCH_COLOR = "#888888"

# Pitch-type display names
PITCH_NAMES: dict[str, str] = {
    "FF": "4-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "SV": "Sweeper",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "K-Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "KN": "Knuckle",
}

# ── Module-level Statcast cache ──────────────────────────────────────────
_statcast_cache: dict[int, "pd.DataFrame | None"] = {}


def _apply_dark_theme(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply dark theme colours to a figure + axes."""
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


# ── Data fetcher ─────────────────────────────────────────────────────────

def fetch_statcast_pitches(pitcher_id: int, lookback: int = 60) -> "pd.DataFrame | None":
    """Fetch recent Statcast pitch-level data with caching.

    Uses a 60-day lookback from yesterday.  Falls back to Sep–Oct of
    ``MLB_SEASON`` during the offseason.
    """
    if pitcher_id in _statcast_cache:
        return _statcast_cache[pitcher_id]

    import pandas as pd  # local import keeps top-level lightweight

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback)

    df: pd.DataFrame | None = None
    try:
        log.info("Fetching Statcast pitches for %s (%s → %s)", pitcher_id, start, end)
        raw = statcast_pitcher(
            start_dt=start.strftime("%Y-%m-%d"),
            end_dt=end.strftime("%Y-%m-%d"),
            player_id=pitcher_id,
        )
        if raw is not None and not raw.empty:
            df = raw
    except Exception:
        log.warning("Statcast fetch failed for pitcher %s", pitcher_id, exc_info=True)

    # Offseason fallback: try MLB_SEASON, then MLB_SEASON-1
    for season in (MLB_SEASON, MLB_SEASON - 1):
        if df is not None:
            break
        try:
            fs = date(season, 6, 1)
            fe = date(season, 10, 31)
            log.info("Season fallback for %s (%s → %s)", pitcher_id, fs, fe)
            raw = statcast_pitcher(
                start_dt=fs.strftime("%Y-%m-%d"),
                end_dt=fe.strftime("%Y-%m-%d"),
                player_id=pitcher_id,
            )
            if raw is not None and not raw.empty:
                df = raw
        except Exception:
            log.warning("Season fallback failed for pitcher %s (season %s)", pitcher_id, season, exc_info=True)

    _statcast_cache[pitcher_id] = df
    return df


# ── Chart 1: Pitch Movement (HB × IVB scatter) ─────────────────────────

def plot_pitch_movement(pitcher_id: int, name: str) -> Path | None:
    """Scatter plot of horizontal break vs induced vertical break by pitch type."""
    try:
        df = fetch_statcast_pitches(pitcher_id)
        if df is None:
            return None

        needed = {"pfx_x", "pfx_z", "pitch_type"}
        if not needed.issubset(df.columns):
            return None

        df = df.dropna(subset=["pfx_x", "pfx_z", "pitch_type"])
        if df.empty:
            return None

        # Convert feet → inches
        df = df.copy()
        df["hb"] = df["pfx_x"] * 12
        df["ivb"] = df["pfx_z"] * 12

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=100)
        _apply_dark_theme(ax, fig)

        pitch_types = sorted(df["pitch_type"].unique())
        for pt in pitch_types:
            subset = df[df["pitch_type"] == pt]
            color = PITCH_COLORS.get(pt, DEFAULT_PITCH_COLOR)
            label = PITCH_NAMES.get(pt, pt)
            ax.scatter(
                subset["hb"], subset["ivb"],
                c=color, label=label, alpha=0.6, s=30, edgecolors="none",
            )

        ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
        ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
        ax.set_xlabel("Horizontal Break (in)", fontsize=11)
        ax.set_ylabel("Induced Vertical Break (in)", fontsize=11)
        ax.set_title(f"{name} — Pitch Movement", fontsize=14, fontweight="bold")
        ax.legend(
            loc="upper right", fontsize=9, framealpha=0.3,
            facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        )
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        out = SCREENSHOTS_DIR / f"movement_{pitcher_id}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved pitch movement chart: %s", out)
        return out

    except Exception:
        log.warning("plot_pitch_movement failed for %s", name, exc_info=True)
        return None


# ── Chart 2: Pitch Locations (plate_x × plate_z scatter) ────────────────

def plot_pitch_locations(pitcher_id: int, name: str, *, anonymize: bool = False) -> Path | None:
    """Scatter plot of pitch locations with strike zone overlay.

    When *anonymize* is True the title says "Mystery Pitcher" and the legend
    is hidden (used for the Guess the Pitcher generator).
    """
    try:
        df = fetch_statcast_pitches(pitcher_id)
        if df is None:
            return None

        needed = {"plate_x", "plate_z", "pitch_type"}
        if not needed.issubset(df.columns):
            return None

        df = df.dropna(subset=["plate_x", "plate_z", "pitch_type"])
        if df.empty:
            return None

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=100)
        _apply_dark_theme(ax, fig)

        # Strike zone rectangle (MLB rule-book zone, approximate)
        sz = Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2,
                        edgecolor=TEXT_COLOR, facecolor="none", linestyle="--")
        ax.add_patch(sz)

        pitch_types = sorted(df["pitch_type"].unique())
        for pt in pitch_types:
            subset = df[df["pitch_type"] == pt]
            color = PITCH_COLORS.get(pt, DEFAULT_PITCH_COLOR)
            label = PITCH_NAMES.get(pt, pt)
            ax.scatter(
                subset["plate_x"], subset["plate_z"],
                c=color, label=label, alpha=0.5, s=25, edgecolors="none",
            )

        title = "Mystery Pitcher — Pitch Locations" if anonymize else f"{name} — Pitch Locations"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Horizontal (ft from center)", fontsize=11)
        ax.set_ylabel("Height (ft)", fontsize=11)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        if not anonymize:
            ax.legend(
                loc="upper right", fontsize=9, framealpha=0.3,
                facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
            )

        suffix = "anon" if anonymize else str(pitcher_id)
        out = SCREENSHOTS_DIR / f"locations_{suffix}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved pitch locations chart: %s", out)
        return out

    except Exception:
        log.warning("plot_pitch_locations failed for %s", name, exc_info=True)
        return None


# ── Chart 3: Pitch Heatmap (faceted density by pitch type) ──────────────

def plot_pitch_heatmap(pitcher_id: int, name: str) -> Path | None:
    """Faceted 2-D density heatmaps of pitch location by pitch type."""
    try:
        df = fetch_statcast_pitches(pitcher_id)
        if df is None:
            return None

        needed = {"plate_x", "plate_z", "pitch_type"}
        if not needed.issubset(df.columns):
            return None

        df = df.dropna(subset=["plate_x", "plate_z", "pitch_type"])
        if df.empty:
            return None

        pitch_types = sorted(df["pitch_type"].unique())
        n = len(pitch_types)
        if n == 0:
            return None

        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(FIG_W, FIG_H), dpi=100,
                                  squeeze=False)
        fig.set_facecolor(BG_COLOR)
        fig.suptitle(f"{name} — Pitch Heatmaps", fontsize=14,
                      fontweight="bold", color=TEXT_COLOR, y=0.98)

        for idx, pt in enumerate(pitch_types):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.set_facecolor(BG_COLOR)
            subset = df[df["pitch_type"] == pt]

            ax.hexbin(
                subset["plate_x"], subset["plate_z"],
                gridsize=15, cmap="YlOrRd", mincnt=1, extent=(-2.5, 2.5, 0, 5),
            )

            # Strike zone
            sz = Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=1.5,
                            edgecolor=TEXT_COLOR, facecolor="none", linestyle="--")
            ax.add_patch(sz)

            label = PITCH_NAMES.get(pt, pt)
            color = PITCH_COLORS.get(pt, DEFAULT_PITCH_COLOR)
            ax.set_title(label, fontsize=10, fontweight="bold", color=color)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(0, 5)
            ax.set_aspect("equal")
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)

        # Hide unused subplots
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = SCREENSHOTS_DIR / f"heatmap_{pitcher_id}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved pitch heatmap chart: %s", out)
        return out

    except Exception:
        log.warning("plot_pitch_heatmap failed for %s", name, exc_info=True)
        return None


# ── Chart 4: Percentile Rankings (horizontal bar, Savant-style) ─────────

# Stats to show on the percentile chart, with display labels and whether
# lower-is-better (ascending=True means percentile is inverted).
_PCTILE_STATS = [
    ("era", "ERA", True),
    ("fip", "FIP", True),
    ("strike_out_percentage", "K%", False),
    ("walk_percentage", "BB%", True),
    ("whiff_rate", "Whiff%", False),
    ("chase_percentage", "Chase%", False),
    ("stuff_plus", "Stuff+", False),
    ("barrel_percentage", "Barrel%", True),
]


def plot_percentile_rankings(name: str, season_df: "pd.DataFrame") -> Path | None:
    """Horizontal bar chart showing a pitcher's percentile rank among peers.

    *season_df* should be the full Pitch Profiler season DataFrame so
    percentiles can be computed in-place (no extra API call).
    """
    try:
        import pandas as pd

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in season_df.columns:
                name_col = c
                break
        if not name_col:
            return None

        matches = season_df[season_df[name_col] == name]
        if matches.empty:
            return None
        player = matches.iloc[0]

        labels: list[str] = []
        percentiles: list[float] = []
        colors: list[str] = []

        for col, label, ascending in _PCTILE_STATS:
            if col not in season_df.columns or col not in player.index:
                continue
            vals = season_df[col].dropna()
            if vals.empty:
                continue
            pctile = (vals < player[col]).sum() / len(vals) * 100
            if ascending:
                pctile = 100 - pctile
            labels.append(label)
            percentiles.append(pctile)
            # Color: red (<30), yellow (30-70), blue (>70)
            if pctile >= 70:
                colors.append("#3a86ff")
            elif pctile >= 30:
                colors.append("#ffbe0b")
            else:
                colors.append("#d62828")

        if not labels:
            return None

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=100)
        _apply_dark_theme(ax, fig)

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, percentiles, color=colors, height=0.6, edgecolor="none")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentile", fontsize=11)
        ax.set_title(f"{name} — Percentile Rankings", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        # Add percentile values at the end of each bar
        for bar, pctile in zip(bars, percentiles):
            ax.text(
                bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{pctile:.0f}", va="center", ha="left",
                color=TEXT_COLOR, fontsize=10, fontweight="bold",
            )

        out = SCREENSHOTS_DIR / f"percentile_{name.replace(' ', '_').lower()}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved percentile rankings chart: %s", out)
        return out

    except Exception:
        log.warning("plot_percentile_rankings failed for %s", name, exc_info=True)
        return None
