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

# MLB team colours (primary brand colour per team abbreviation)
TEAM_COLORS: dict[str, str] = {
    "ARI": "#A71930", "ATL": "#CE1141", "BAL": "#DF4601", "BOS": "#BD3039",
    "CHC": "#0E3386", "CWS": "#27251F", "CIN": "#C6011F", "CLE": "#00385D",
    "COL": "#333366", "DET": "#0C2340", "HOU": "#002D62", "KC":  "#004687",
    "LAA": "#BA0021", "LAD": "#005A9C", "MIA": "#00A3E0", "MIL": "#FFC52F",
    "MIN": "#002B5C", "NYM": "#002D72", "NYY": "#003087", "OAK": "#003831",
    "PHI": "#E81828", "PIT": "#FDB827", "SD":  "#2F241D", "SF":  "#FD5A1E",
    "SEA": "#0C2C56", "STL": "#C41E3A", "TB":  "#092C5C", "TEX": "#003278",
    "TOR": "#134A8E", "WSH": "#AB0003",
}

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


# ── Chart 5: Movement Profile (HB × IVB arsenal scatter) ─────────────

def plot_movement_profile(name: str, pitches_df: "pd.DataFrame") -> Path | None:
    """Arsenal movement profile — HB vs IVB scatter from Pitch Profiler data.

    Each pitch type is one marker, sized by usage, colored by pitch type,
    and labeled with pitch name + velocity.
    """
    try:
        import pandas as pd

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in pitches_df.columns:
                name_col = c
                break
        if not name_col:
            return None

        prows = pitches_df[pitches_df[name_col] == name].copy()
        if prows.empty or "pitch_type" not in prows.columns:
            return None

        needed = {"hb", "ivb"}
        if not needed.issubset(prows.columns):
            return None

        # Coerce numeric
        for nc in ("hb", "ivb", "velocity", "percentage_thrown"):
            if nc in prows.columns:
                prows[nc] = pd.to_numeric(prows[nc], errors="coerce")

        # Aggregate by pitch type
        agg = {"hb": "mean", "ivb": "mean"}
        if "velocity" in prows.columns:
            agg["velocity"] = "mean"
        if "percentage_thrown" in prows.columns:
            agg["percentage_thrown"] = "sum"

        grouped = prows.groupby("pitch_type", as_index=False).agg(agg)
        grouped = grouped.dropna(subset=["hb", "ivb"])
        if grouped.empty:
            return None

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=100)
        _apply_dark_theme(ax, fig)

        for _, row in grouped.iterrows():
            pt = str(row["pitch_type"])
            hb = float(row["hb"])
            ivb = float(row["ivb"])
            color = PITCH_COLORS.get(pt, DEFAULT_PITCH_COLOR)
            label = PITCH_NAMES.get(pt, pt)

            # Marker size based on usage (min 80, max 400)
            usage = float(row.get("percentage_thrown", 0.1) or 0.1)
            size = max(80, min(400, usage * 800))

            ax.scatter(hb, ivb, c=color, s=size, alpha=0.85,
                       edgecolors="white", linewidths=0.8, zorder=3)

            # Label: pitch name + velocity
            velo_str = ""
            if "velocity" in row.index and pd.notna(row["velocity"]):
                velo_str = f" ({float(row['velocity']):.1f})"
            ax.annotate(
                f"{label}{velo_str}", (hb, ivb),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, fontweight="bold", color=color,
            )

        ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
        ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
        ax.set_xlabel("Horizontal Break (in)", fontsize=11)
        ax.set_ylabel("Induced Vertical Break (in)", fontsize=11)
        ax.set_title(f"{name} — Movement Profile", fontsize=14, fontweight="bold")
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        # Footer
        fig.text(0.5, 0.01, "@TJStatsBot  •  Pitch Profiler Data",
                 fontsize=9, color="#666666", ha="center")

        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"movement_profile_{safe}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved movement profile chart: %s", out)
        return out

    except Exception:
        log.warning("plot_movement_profile failed for %s", name, exc_info=True)
        return None


# ── Chart 6: Pitcher Card (premium PLV-style card) ───────────────────

# Stats for the left panel — (column, display_label, lower_is_better)
_CARD_STATS = [
    ("era", "ERA", True),
    ("fip", "FIP", True),
    ("strike_out_percentage", "K%", False),
    ("walk_percentage", "BB%", True),
    ("whiff_rate", "Whiff%", False),
    ("chase_percentage", "Chase%", False),
    ("stuff_plus", "Stuff+", False),
    ("pitching_plus", "Pitching+", False),
]


def plot_pitcher_card(
    name: str,
    season_df: "pd.DataFrame",
    pitches_df: "pd.DataFrame",
    team: str | None = None,
) -> Path | None:
    """Render a premium pitcher card (1200×675, dark theme).

    Left panel: season overview stats with color-coded percentile bars.
    Right panel: per-pitch arsenal breakdown (velo + whiff bars).

    *season_df* is the full Pitch Profiler season DataFrame (for percentile
    computation).  *pitches_df* is the season pitch-type DataFrame.
    """
    try:
        import pandas as pd

        # ── Locate pitcher row ────────────────────────────────────────
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

        # ── Team colour accent ────────────────────────────────────────
        if not team:
            for tc in ("team", "team_abbreviation", "team_abbrev"):
                if tc in player.index and player[tc]:
                    team = str(player[tc]).upper()
                    break
        accent = TEAM_COLORS.get(team or "", "#3a86ff")

        # ── Compute percentiles for left panel ────────────────────────
        stat_labels: list[str] = []
        stat_values: list[str] = []
        pctiles: list[float] = []
        bar_colors: list[str] = []

        for col, label, ascending in _CARD_STATS:
            if col not in season_df.columns or col not in player.index:
                continue
            vals = season_df[col].dropna()
            if vals.empty:
                continue
            raw = player[col]
            try:
                raw_f = float(raw)
            except (TypeError, ValueError):
                continue

            pctile = (vals < raw_f).sum() / len(vals) * 100
            if ascending:
                pctile = 100 - pctile

            stat_labels.append(label)
            # Display value: percentages ×100
            if col in ("strike_out_percentage", "walk_percentage",
                        "whiff_rate", "chase_percentage"):
                stat_values.append(f"{raw_f * 100:.1f}%")
            elif col in ("era", "fip"):
                stat_values.append(f"{raw_f:.2f}")
            else:
                stat_values.append(f"{raw_f:.0f}")

            pctiles.append(pctile)
            if pctile >= 70:
                bar_colors.append("#3a86ff")   # elite
            elif pctile >= 30:
                bar_colors.append("#ffbe0b")   # average
            else:
                bar_colors.append("#d62828")   # poor

        if not stat_labels:
            return None

        # ── Arsenal data (right panel) ────────────────────────────────
        arsenal_rows: list[dict] = []
        if pitches_df is not None and not pitches_df.empty:
            pitch_name_col = None
            for c in ("pitcher_name", "player_name", "name"):
                if c in pitches_df.columns:
                    pitch_name_col = c
                    break
            if pitch_name_col and "pitch_type" in pitches_df.columns:
                prows = pitches_df[pitches_df[pitch_name_col] == name].copy()
                if not prows.empty:
                    # Coerce numeric columns so groupby aggregation works
                    for nc in ("velocity", "whiff_rate", "percentage_thrown"):
                        if nc in prows.columns:
                            prows[nc] = pd.to_numeric(prows[nc], errors="coerce")

                    # Aggregate per pitch type (mean velo/whiff, sum usage)
                    agg_map = {}
                    if "velocity" in prows.columns:
                        agg_map["velocity"] = "mean"
                    if "whiff_rate" in prows.columns:
                        agg_map["whiff_rate"] = "mean"
                    if "percentage_thrown" in prows.columns:
                        agg_map["percentage_thrown"] = "sum"

                    grouped = prows.groupby("pitch_type", as_index=False).agg(
                        agg_map if agg_map else {"pitch_type": "first"},
                    )

                    for _, row in grouped.iterrows():
                        pcode = str(row["pitch_type"])
                        velo = row.get("velocity", None)
                        whiff = row.get("whiff_rate", None)
                        usage = row.get("percentage_thrown", 0)
                        try:
                            velo_f = float(velo) if pd.notna(velo) else None
                        except (TypeError, ValueError):
                            velo_f = None
                        try:
                            whiff_f = float(whiff) * 100 if pd.notna(whiff) else None
                        except (TypeError, ValueError):
                            whiff_f = None
                        try:
                            usage_f = float(usage) if pd.notna(usage) else 0
                        except (TypeError, ValueError):
                            usage_f = 0
                        color = PITCH_COLORS.get(pcode, DEFAULT_PITCH_COLOR)
                        display = PITCH_NAMES.get(pcode, pcode)
                        arsenal_rows.append({
                            "name": display, "code": pcode, "velo": velo_f,
                            "whiff": whiff_f, "usage": usage_f, "color": color,
                        })
                    arsenal_rows.sort(key=lambda r: r["usage"], reverse=True)

        # ── Build figure ──────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
        fig.set_facecolor(BG_COLOR)

        # ── Accent bar at top ─────────────────────────────────────────
        accent_ax = fig.add_axes([0, 0.95, 1, 0.05])
        accent_ax.set_xlim(0, 1)
        accent_ax.set_ylim(0, 1)
        accent_ax.add_patch(Rectangle((0, 0), 1, 1, color=accent))
        accent_ax.axis("off")

        # ── Header ────────────────────────────────────────────────────
        team_label = f"{team}  •  " if team else ""
        fig.text(
            0.04, 0.89, name,
            fontsize=22, fontweight="bold", color=TEXT_COLOR,
            ha="left", va="top",
        )
        fig.text(
            0.04, 0.83, f"{team_label}{MLB_SEASON} Season",
            fontsize=12, color="#aaaaaa",
            ha="left", va="top",
        )

        # ── Divider line ──────────────────────────────────────────────
        div_ax = fig.add_axes([0.03, 0.79, 0.94, 0.003])
        div_ax.set_xlim(0, 1)
        div_ax.set_ylim(0, 1)
        div_ax.add_patch(Rectangle((0, 0), 1, 1, color=GRID_COLOR))
        div_ax.axis("off")

        # ── Left panel: Season Overview ───────────────────────────────
        left_ax = fig.add_axes([0.04, 0.12, 0.48, 0.62])
        left_ax.set_facecolor(BG_COLOR)

        n_stats = len(stat_labels)
        y_pos = np.arange(n_stats)
        bars = left_ax.barh(y_pos, pctiles, color=bar_colors, height=0.55,
                            edgecolor="none")
        left_ax.set_yticks(y_pos)
        left_ax.set_yticklabels(stat_labels, fontsize=11, color=TEXT_COLOR)
        left_ax.set_xlim(0, 105)
        left_ax.invert_yaxis()
        left_ax.set_title("SEASON OVERVIEW", fontsize=12, fontweight="bold",
                          color=TEXT_COLOR, loc="left", pad=8)

        # Hide frame
        for spine in left_ax.spines.values():
            spine.set_visible(False)
        left_ax.tick_params(left=False, bottom=False, labelbottom=False,
                            colors=TEXT_COLOR)
        left_ax.set_facecolor(BG_COLOR)
        left_ax.grid(False)

        # Stat value + percentile labels on each bar
        for bar, val, pctile in zip(bars, stat_values, pctiles):
            # Value label inside bar
            left_ax.text(
                2, bar.get_y() + bar.get_height() / 2,
                val, va="center", ha="left",
                color="white", fontsize=10, fontweight="bold",
            )
            # Percentile at end
            left_ax.text(
                bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{pctile:.0f}th", va="center", ha="left",
                color=TEXT_COLOR, fontsize=9,
            )

        # ── Vertical divider ─────────────────────────────────────────
        vdiv = fig.add_axes([0.56, 0.12, 0.002, 0.65])
        vdiv.set_xlim(0, 1)
        vdiv.set_ylim(0, 1)
        vdiv.add_patch(Rectangle((0, 0), 1, 1, color=GRID_COLOR))
        vdiv.axis("off")

        # ── Right panel: Arsenal ──────────────────────────────────────
        if arsenal_rows:
            right_ax = fig.add_axes([0.60, 0.12, 0.36, 0.62])
            right_ax.set_facecolor(BG_COLOR)
            right_ax.set_title("ARSENAL", fontsize=12, fontweight="bold",
                               color=TEXT_COLOR, loc="left", pad=8)

            max_whiff = max((r["whiff"] for r in arsenal_rows if r["whiff"]),
                            default=50)

            n_pitches = len(arsenal_rows)
            # Space pitches evenly in the panel
            row_height = min(0.9 / max(n_pitches, 1), 0.14)

            for i, pitch in enumerate(arsenal_rows):
                y = 1.0 - (i + 1) * row_height * 1.1
                # Pitch name (colored)
                velo_str = f"  {pitch['velo']:.1f} mph" if pitch["velo"] else ""
                right_ax.text(
                    0.02, y, f"{pitch['name']}{velo_str}",
                    transform=right_ax.transAxes,
                    fontsize=10, fontweight="bold", color=pitch["color"],
                    va="center", ha="left",
                )
                # Whiff bar
                if pitch["whiff"] is not None:
                    bar_width = (pitch["whiff"] / max_whiff) * 0.45
                    right_ax.add_patch(Rectangle(
                        (0.55, y - 0.02), bar_width, 0.04,
                        transform=right_ax.transAxes,
                        color=pitch["color"], alpha=0.8,
                    ))
                    right_ax.text(
                        0.55 + bar_width + 0.02, y,
                        f"{pitch['whiff']:.0f}% whiff",
                        transform=right_ax.transAxes,
                        fontsize=8, color=TEXT_COLOR, va="center",
                    )

            right_ax.axis("off")

        # ── Footer ────────────────────────────────────────────────────
        fig.text(
            0.5, 0.03,
            "@TJStatsBot  •  Pitch Profiler Data",
            fontsize=9, color="#666666",
            ha="center", va="center",
        )

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"pitcher_card_{safe}.png"
        fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        log.info("Saved pitcher card: %s", out)
        return out

    except Exception:
        log.warning("plot_pitcher_card failed for %s", name, exc_info=True)
        return None
