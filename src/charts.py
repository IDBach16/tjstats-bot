"""Chart generation module — locally-rendered Statcast visuals for tweets.

All public functions return ``Path | None``: the path to a saved PNG inside
``SCREENSHOTS_DIR``, or *None* on any failure.  Charts use the ``Agg``
backend so they work headlessly on GitHub Actions.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from io import BytesIO

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.patheffects as patheffects        # noqa: E402
import numpy as np                                  # noqa: E402
import requests as _requests                        # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from pybaseball import statcast_pitcher             # noqa: E402

from .config import MLB_SEASON, SCREENSHOTS_DIR

log = logging.getLogger(__name__)

# ── Theme ────────────────────────────────────────────────────────────────
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
FIG_W, FIG_H = 12, 6.75  # 1200×675 at 100 dpi — Twitter-optimal 16:9

# ── Premium Card Theme ────────────────────────────────────────────────
CARD_BG = "#0d1117"
CARD_SURFACE = "#161b22"
CARD_BORDER = "#30363d"
CARD_TEXT = "#f0f6fc"
CARD_TEXT_MUTED = "#8b949e"

# Noise pitch types to filter
_NOISE_PITCHES = {"PO", "IN", "EP", "AB", "AS", "UN", "XX", "NP", "SC"}

# Headshot cache
_headshot_cache: dict[int, "np.ndarray | None"] = {}

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


# ── Pitcher Card Helpers ─────────────────────────────────────────────

def _fetch_headshot(player_id: int, accent: str = "#3a86ff") -> "np.ndarray | None":
    """Fetch MLB headshot, crop to circle with accent ring, return as RGBA array."""
    if player_id in _headshot_cache:
        return _headshot_cache[player_id]

    url = (
        f"https://img.mlb.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/"
        f"w_213,q_auto:best/v1/people/{player_id}/headshot/67/current"
    )
    try:
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        from PIL import Image, ImageDraw

        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        # Make square
        size = min(img.size)
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((200, 200), Image.LANCZOS)

        # Circular mask
        mask = Image.new("L", (200, 200), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((4, 4, 196, 196), fill=255)
        img.putalpha(mask)

        # Draw accent ring
        ring = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring)
        # Parse accent hex to RGB
        ac = accent.lstrip("#")
        ar, ag, ab = int(ac[:2], 16), int(ac[2:4], 16), int(ac[4:6], 16)
        ring_draw.ellipse((1, 1, 199, 199), outline=(ar, ag, ab, 255), width=4)
        img = Image.alpha_composite(img, ring)

        arr = np.array(img)
        _headshot_cache[player_id] = arr
        return arr
    except Exception:
        log.debug("Headshot fetch failed for player %s", player_id)
        _headshot_cache[player_id] = None
        return None


def _pctile_color(pctile: float) -> str:
    """Map a 0-100 percentile to a Savant-style color (red → orange → yellow → blue)."""
    if pctile >= 80:
        return "#3a86ff"
    elif pctile >= 60:
        return "#60a5fa"
    elif pctile >= 40:
        return "#ffbe0b"
    elif pctile >= 20:
        return "#e85d04"
    else:
        return "#d62828"


def _draw_gradient_rect(fig, rect, color_top, color_bottom, alpha=1.0):
    """Draw a vertical gradient rectangle on the figure."""
    ax = fig.add_axes(rect)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("grad", [color_bottom, color_top])
    ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower",
              extent=[0, 1, 0, 1], alpha=alpha)
    ax.axis("off")
    return ax


def _rounded_bar(ax, x, y, width, height, color, alpha=0.85):
    """Draw a rounded-corner bar on axes using FancyBboxPatch."""
    if width <= 0:
        return
    rounding = min(height * 0.4, width * 0.3, 0.015)
    patch = FancyBboxPatch(
        (x, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=color, edgecolor="none", alpha=alpha,
    )
    ax.add_patch(patch)


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
    player_id: int | None = None,
) -> Path | None:
    """Render a premium pitcher card (1200×675, dark theme).

    Left panel: season overview stats with Savant-style percentile bars.
    Right panel: per-pitch arsenal breakdown with usage + whiff bars.
    Header: player headshot, name, team, hero stats.
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

        # ── Compute percentiles ───────────────────────────────────────
        stat_labels: list[str] = []
        stat_values: list[str] = []
        stat_raw: list[float] = []
        pctiles: list[float] = []

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
            stat_raw.append(raw_f)
            if col in ("strike_out_percentage", "walk_percentage",
                        "whiff_rate", "chase_percentage"):
                stat_values.append(f"{raw_f * 100:.1f}%")
            elif col in ("era", "fip"):
                stat_values.append(f"{raw_f:.2f}")
            else:
                stat_values.append(f"{raw_f:.0f}")
            pctiles.append(pctile)

        if not stat_labels:
            return None

        # ── Arsenal data ──────────────────────────────────────────────
        arsenal_rows: list[dict] = []
        if pitches_df is not None and not pitches_df.empty:
            pitch_name_col = None
            for c in ("pitcher_name", "player_name", "name"):
                if c in pitches_df.columns:
                    pitch_name_col = c
                    break
            if pitch_name_col and "pitch_type" in pitches_df.columns:
                prows = pitches_df[pitches_df[pitch_name_col] == name].copy()
                # Filter noise pitch types
                prows = prows[~prows["pitch_type"].isin(_NOISE_PITCHES)]
                if not prows.empty:
                    for nc in ("velocity", "whiff_rate", "percentage_thrown"):
                        if nc in prows.columns:
                            prows[nc] = pd.to_numeric(prows[nc], errors="coerce")

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

                    # Normalize usage to fractions summing to 1.0
                    if "percentage_thrown" in grouped.columns:
                        total = grouped["percentage_thrown"].sum()
                        if total > 0:
                            grouped["percentage_thrown"] = (
                                grouped["percentage_thrown"] / total
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
        fig.set_facecolor(CARD_BG)

        # ── Subtle background noise texture ───────────────────────────
        noise = np.random.default_rng(42).uniform(0.04, 0.07, (68, 120))
        bg_ax = fig.add_axes([0, 0, 1, 1])
        bg_ax.imshow(noise, aspect="auto", cmap="gray", alpha=0.03,
                     extent=[0, 1, 0, 1])
        bg_ax.axis("off")

        # ── Header gradient (team colour fade) ────────────────────────
        _draw_gradient_rect(fig, [0, 0.78, 1, 0.22], accent, CARD_BG, alpha=0.25)

        # ── Accent stripe at very top ─────────────────────────────────
        stripe = fig.add_axes([0, 0.97, 1, 0.03])
        stripe.set_xlim(0, 1)
        stripe.set_ylim(0, 1)
        stripe.add_patch(Rectangle((0, 0), 1, 1, color=accent))
        stripe.axis("off")

        # ── Headshot ──────────────────────────────────────────────────
        name_x = 0.04  # default if no headshot
        has_headshot = False
        if player_id:
            hs_arr = _fetch_headshot(player_id, accent)
            if hs_arr is not None:
                has_headshot = True
                name_x = 0.18
                hs_ax = fig.add_axes([0.02, 0.75, 0.13, 0.22])
                hs_ax.imshow(hs_arr)
                hs_ax.axis("off")

        # ── Player name (with shadow) ─────────────────────────────────
        shadow = [patheffects.withStroke(linewidth=4, foreground=CARD_BG)]
        fig.text(
            name_x, 0.95, name,
            fontsize=28, fontweight="bold", color=CARD_TEXT,
            ha="left", va="top", path_effects=shadow,
        )

        # ── Team + season subtitle ────────────────────────────────────
        team_label = f"{team}  |  " if team else ""
        fig.text(
            name_x, 0.88, f"{team_label}{MLB_SEASON} Season",
            fontsize=13, color=CARD_TEXT_MUTED,
            ha="left", va="top",
        )

        # ── Hero stat boxes (4 key stats in pill boxes) ───────────────
        hero_stats = []
        hero_map = {"ERA": 0, "K%": 1, "Stuff+": 2, "Pitching+": 3,
                     "FIP": 4, "Whiff%": 5}
        for lbl, val, pct in zip(stat_labels, stat_values, pctiles):
            if lbl in hero_map:
                hero_stats.append((lbl, val, pct, hero_map[lbl]))
        hero_stats.sort(key=lambda x: x[3])
        hero_stats = hero_stats[:4]

        box_y = 0.79
        box_w = 0.095
        box_h = 0.07
        box_gap = 0.008
        for i, (lbl, val, pct, _) in enumerate(hero_stats):
            bx = name_x + i * (box_w + box_gap)
            box_ax = fig.add_axes([bx, box_y, box_w, box_h])
            box_ax.set_xlim(0, 1)
            box_ax.set_ylim(0, 1)
            box_ax.add_patch(FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0,rounding_size=0.15",
                facecolor=CARD_SURFACE, edgecolor=CARD_BORDER, linewidth=1,
            ))
            # Accent top edge
            box_ax.plot([0.15, 0.85], [0.98, 0.98], color=_pctile_color(pct),
                        linewidth=3, solid_capstyle="round")
            # Value
            box_ax.text(0.5, 0.62, val, fontsize=14, fontweight="bold",
                        color=CARD_TEXT, ha="center", va="center")
            # Label
            box_ax.text(0.5, 0.22, lbl, fontsize=8,
                        color=CARD_TEXT_MUTED, ha="center", va="center")
            box_ax.axis("off")

        # ── Accent divider line ───────────────────────────────────────
        div_ax = fig.add_axes([0.03, 0.74, 0.94, 0.004])
        div_ax.set_xlim(0, 1)
        div_ax.set_ylim(0, 1)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        cmap_div = LinearSegmentedColormap.from_list("div", ["#00000000", accent, "#00000000"])
        div_ax.imshow(grad, aspect="auto", cmap=cmap_div, extent=[0, 1, 0, 1])
        div_ax.axis("off")

        # ── Section header: PERCENTILE RANKINGS ───────────────────────
        fig.text(0.05, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.065, 0.71, "PERCENTILE RANKINGS", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Left panel: percentile bars ───────────────────────────────
        left_ax = fig.add_axes([0.04, 0.07, 0.48, 0.61])
        left_ax.set_xlim(0, 1.12)
        left_ax.set_ylim(-0.5, len(stat_labels) - 0.5)
        left_ax.invert_yaxis()
        left_ax.set_facecolor("none")
        for spine in left_ax.spines.values():
            spine.set_visible(False)
        left_ax.tick_params(left=False, bottom=False, labelbottom=False,
                            labelleft=False)

        bar_track_w = 0.62
        bar_x0 = 0.18
        bar_h = 0.028  # bar height in axes fraction

        for i, (lbl, val, pct) in enumerate(zip(stat_labels, stat_values, pctiles)):
            y = i
            color = _pctile_color(pct)

            # Stat label
            left_ax.text(0.0, y, lbl, fontsize=11, fontweight="bold",
                         color=CARD_TEXT, va="center", ha="left")

            # Track background (rounded)
            _rounded_bar(left_ax, bar_x0, y, bar_track_w, bar_h,
                         CARD_BORDER, alpha=0.5)

            # Filled bar (rounded)
            fill_w = max((pct / 100) * bar_track_w, 0.008)
            _rounded_bar(left_ax, bar_x0, y, fill_w, bar_h, color, alpha=0.9)

            # Value label inside bar (or just right of it if narrow)
            val_x = bar_x0 + fill_w - 0.01 if pct > 25 else bar_x0 + fill_w + 0.01
            val_ha = "right" if pct > 25 else "left"
            val_color = "white" if pct > 25 else CARD_TEXT
            left_ax.text(val_x, y, val, fontsize=9, fontweight="bold",
                         color=val_color, va="center", ha=val_ha)

            # Percentile at far right
            left_ax.text(bar_x0 + bar_track_w + 0.03, y,
                         f"{pct:.0f}th", fontsize=10, fontweight="bold",
                         color=color, va="center", ha="left")

        # ── Vertical divider ──────────────────────────────────────────
        vdiv = fig.add_axes([0.545, 0.08, 0.003, 0.63])
        vdiv.set_xlim(0, 1)
        vdiv.set_ylim(0, 1)
        vgrad = np.linspace(0, 1, 256).reshape(-1, 1)
        cmap_v = LinearSegmentedColormap.from_list("vd", ["#00000000", CARD_BORDER, "#00000000"])
        vdiv.imshow(vgrad, aspect="auto", cmap=cmap_v, extent=[0, 1, 0, 1])
        vdiv.axis("off")

        # ── Section header: ARSENAL ───────────────────────────────────
        fig.text(0.57, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.585, 0.71, "ARSENAL", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Right panel: arsenal ──────────────────────────────────────
        if arsenal_rows:
            right_ax = fig.add_axes([0.56, 0.07, 0.42, 0.61])
            right_ax.set_xlim(0, 1)
            n_pitches = len(arsenal_rows)
            right_ax.set_ylim(-0.5, n_pitches - 0.5)
            right_ax.invert_yaxis()
            right_ax.set_facecolor("none")
            for spine in right_ax.spines.values():
                spine.set_visible(False)
            right_ax.tick_params(left=False, bottom=False,
                                 labelbottom=False, labelleft=False)

            max_usage = max((r["usage"] for r in arsenal_rows), default=1) or 1

            for i, pitch in enumerate(arsenal_rows):
                y = i
                c = pitch["color"]

                # Color dot
                right_ax.plot(0.03, y, "o", color=c, markersize=10,
                              markeredgecolor="white", markeredgewidth=0.8)

                # Pitch name
                right_ax.text(0.08, y - 0.12, pitch["name"],
                              fontsize=11, fontweight="bold", color=c,
                              va="center", ha="left")

                # Velocity
                if pitch["velo"]:
                    right_ax.text(0.08, y + 0.18,
                                  f"{pitch['velo']:.1f} mph",
                                  fontsize=9, color=CARD_TEXT_MUTED,
                                  va="center", ha="left")

                # Usage bar
                usage_x0 = 0.38
                usage_max_w = 0.35
                usage_w = (pitch["usage"] / max_usage) * usage_max_w
                _rounded_bar(right_ax, usage_x0, y - 0.08, usage_max_w,
                             0.022, CARD_BORDER, alpha=0.4)
                _rounded_bar(right_ax, usage_x0, y - 0.08, max(usage_w, 0.008),
                             0.022, c, alpha=0.75)

                # Usage % label (usage is always 0-1 fraction after normalization)
                usage_pct = pitch["usage"] * 100
                right_ax.text(usage_x0 + usage_max_w + 0.02, y - 0.08,
                              f"{usage_pct:.0f}%", fontsize=9,
                              fontweight="bold", color=CARD_TEXT,
                              va="center", ha="left")

                # Whiff bar (smaller, below usage)
                if pitch["whiff"] is not None:
                    whiff_w = (pitch["whiff"] / 50) * usage_max_w  # 50% = full bar
                    whiff_w = min(whiff_w, usage_max_w)
                    _rounded_bar(right_ax, usage_x0, y + 0.14,
                                 usage_max_w, 0.016, CARD_BORDER, alpha=0.3)
                    _rounded_bar(right_ax, usage_x0, y + 0.14,
                                 max(whiff_w, 0.005), 0.016, c, alpha=0.45)
                    right_ax.text(usage_x0 + usage_max_w + 0.02, y + 0.14,
                                  f"{pitch['whiff']:.0f}% whiff",
                                  fontsize=7, color=CARD_TEXT_MUTED,
                                  va="center", ha="left")

        # ── Footer ────────────────────────────────────────────────────
        foot_ax = fig.add_axes([0.03, 0.0, 0.94, 0.003])
        foot_ax.set_xlim(0, 1)
        foot_ax.set_ylim(0, 1)
        foot_ax.add_patch(Rectangle((0, 0), 1, 1, color=CARD_BORDER))
        foot_ax.axis("off")

        fig.text(0.04, 0.025, "@TJStatsBot", fontsize=10, color=accent,
                 ha="left", va="center", fontweight="bold")
        fig.text(0.5, 0.025, "Pitch Profiler Data", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="center", va="center")
        fig.text(0.96, 0.025, f"{MLB_SEASON}", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="right", va="center")

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"pitcher_card_{safe}.png"
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=100,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        log.info("Saved pitcher card: %s", out)
        return out

    except Exception:
        log.warning("plot_pitcher_card failed for %s", name, exc_info=True)
        return None


# ── TJStats Pitch Colours (from notebook) ────────────────────────────

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

# Color maps for table cell coloring (blue = good, orange = bad)
_CMAP_GOOD = LinearSegmentedColormap.from_list("tj", ['#FFB000', '#FFFFFF', '#648FFF'])
_CMAP_BAD = LinearSegmentedColormap.from_list("tj_r", ['#648FFF', '#FFFFFF', '#FFB000'])


# ── Chart 7: Pitching Summary Dashboard (TJStats-style) ─────────────

# Columns for the pitch stats table and their display info
_PITCH_TABLE_COLS = [
    # (pp_column, header, format, higher_is_better)
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

# Season overview stats (from season_df)
_SUMMARY_STATS = [
    ("innings_pitched", "$\\bf{IP}$", ".1f"),
    ("batters_faced", "$\\bf{PA}$", ".0f"),
    ("whip", "$\\bf{WHIP}$", ".2f"),
    ("era", "$\\bf{ERA}$", ".2f"),
    ("fip", "$\\bf{FIP}$", ".2f"),
    ("strike_out_percentage", "$\\bf{K\\%}$", ".1%"),
    ("walk_percentage", "$\\bf{BB\\%}$", ".1%"),
    ("whiff_rate", "$\\bf{Whiff\\%}$", ".1%"),
    ("stuff_plus", "$\\bf{Stf+}$", ".0f"),
    ("pitching_plus", "$\\bf{Pit+}$", ".0f"),
]


def _get_table_cell_color(value: float, league_mean: float,
                          cmap, spread: float = 0.3) -> str:
    """Color-code a cell value relative to the league mean."""
    import matplotlib.colors as mcolors
    lo = league_mean * (1 - spread)
    hi = league_mean * (1 + spread)
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    return mcolors.to_hex(cmap(norm(value)))


def plot_pitching_summary(
    name: str,
    season_df: "pd.DataFrame",
    pitches_df: "pd.DataFrame",
    all_pitches_df: "pd.DataFrame | None" = None,
    team: str | None = None,
    player_id: int | None = None,
) -> Path | None:
    """Render a TJStats-style full pitching summary dashboard.

    Uses Pitch Profiler API data instead of raw Statcast.
    Layout: header (headshot + bio + logo), season stats table,
    movement chart + percentile bars, pitch stats table, footer.
    """
    try:
        import pandas as pd
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors

        # ── Locate pitcher in season data ─────────────────────────────
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

        # ── Team info ─────────────────────────────────────────────────
        if not team:
            for tc in ("team", "team_abbreviation", "team_abbrev"):
                if tc in player.index and player[tc]:
                    team = str(player[tc]).upper()
                    break
        accent = TEAM_COLORS.get(team or "", "#3a86ff")

        # ── Pitcher hand ──────────────────────────────────────────────
        p_throws = "R"
        for hc in ("p_throws", "throws", "hand"):
            if hc in player.index and player[hc]:
                p_throws = str(player[hc])[0].upper()
                break

        # ── Pitch-type data ───────────────────────────────────────────
        pitch_name_col = None
        if pitches_df is not None and not pitches_df.empty:
            for c in ("pitcher_name", "player_name", "name"):
                if c in pitches_df.columns:
                    pitch_name_col = c
                    break

        pitch_rows = pd.DataFrame()
        if pitch_name_col and "pitch_type" in pitches_df.columns:
            prows = pitches_df[pitches_df[pitch_name_col] == name].copy()
            prows = prows[~prows["pitch_type"].isin(_NOISE_PITCHES)]

            # Try to get pitcher hand from pitch data
            if "p_throws" in prows.columns and not prows.empty:
                pt_val = prows["p_throws"].dropna().iloc[0] if not prows["p_throws"].dropna().empty else None
                if pt_val:
                    p_throws = str(pt_val)[0].upper()

            if not prows.empty:
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

        # ── Figure + GridSpec ─────────────────────────────────────────
        fig = plt.figure(figsize=(20, 20), dpi=150)
        fig.set_facecolor("white")

        gs = gridspec.GridSpec(
            6, 8,
            height_ratios=[2, 18, 8, 30, 30, 5],
            width_ratios=[1, 18, 18, 18, 18, 18, 18, 1],
            hspace=0.3, wspace=0.3,
        )

        # Border axes (hidden)
        for pos in [gs[0, 1:7], gs[-1, 1:7], gs[:, 0], gs[:, -1]]:
            bax = fig.add_subplot(pos)
            bax.axis("off")

        # ── Row 1: Header — headshot / bio / logo ─────────────────────
        ax_headshot = fig.add_subplot(gs[1, 1:3])
        ax_bio = fig.add_subplot(gs[1, 3:5])
        ax_logo = fig.add_subplot(gs[1, 5:7])

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
                resp = _requests.get(hs_url, timeout=10)
                resp.raise_for_status()
                from PIL import Image
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
                    fontsize=48, fontweight="bold")
        ax_bio.text(0.5, 0.60, hand_str, va="top", ha="center",
                    fontsize=26, color="#555555")
        ax_bio.text(0.5, 0.35, "Season Pitching Summary", va="top",
                    ha="center", fontsize=34, fontweight="bold")
        ax_bio.text(0.5, 0.10, f"{MLB_SEASON} MLB Season", va="top",
                    ha="center", fontsize=26, fontstyle="italic",
                    color="#666666")

        # Team logo
        ax_logo.axis("off")
        if team:
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
            slug = _LOGO_MAP.get(team)
            if slug:
                try:
                    logo_url = (
                        f"https://a.espncdn.com/combiner/i?img="
                        f"/i/teamlogos/mlb/500/scoreboard/{slug}.png"
                        f"&h=500&w=500"
                    )
                    resp = _requests.get(logo_url, timeout=10)
                    resp.raise_for_status()
                    from PIL import Image
                    img = Image.open(BytesIO(resp.content))
                    ax_logo.set_xlim(0, 1.3)
                    ax_logo.set_ylim(0, 1)
                    ax_logo.imshow(img, extent=[0.3, 1.3, 0, 1],
                                   origin="upper")
                except Exception:
                    log.debug("Logo failed for %s", team)

        # ── Row 2: Season Stats Table ─────────────────────────────────
        ax_season = fig.add_subplot(gs[2, 1:7])
        ax_season.axis("off")

        season_headers = []
        season_values = []
        for col, header, fmt in _SUMMARY_STATS:
            if col in player.index:
                try:
                    val = float(player[col])
                    season_values.append(format(val, fmt))
                    season_headers.append(header)
                except (TypeError, ValueError):
                    pass

        if season_values:
            tbl = ax_season.table(
                cellText=[season_values],
                colLabels=season_headers,
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

        # ── Row 3: Movement chart + Percentile bars ───────────────────
        ax_break = fig.add_subplot(gs[3, 1:4])
        ax_pctile = fig.add_subplot(gs[3, 4:7])

        # Movement scatter
        if not pitch_rows.empty and "hb" in pitch_rows.columns and "ivb" in pitch_rows.columns:
            for _, row in pitch_rows.iterrows():
                pt = str(row["pitch_type"])
                hb_val = float(row.get("hb", 0) or 0)
                ivb_val = float(row.get("ivb", 0) or 0)
                color = _TJ_COLOUR.get(pt, "#888888")
                pname = _TJ_NAME.get(pt, pt)
                usage = float(row.get("percentage_thrown", 0.1) or 0.1)
                size = max(100, min(600, usage * 1200))

                # Flip HB for RHP (like the notebook)
                if p_throws == "R":
                    hb_plot = -hb_val
                else:
                    hb_plot = hb_val

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

        # Percentile rankings
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
        for col, label, ascending in pctile_stats:
            if col not in season_df.columns or col not in player.index:
                continue
            vals = season_df[col].dropna()
            if vals.empty:
                continue
            try:
                raw_f = float(player[col])
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

        # ── Row 4: Color-coded pitch stats table ─────────────────────
        ax_table = fig.add_subplot(gs[4, 1:7])
        ax_table.axis("off")

        if not pitch_rows.empty:
            # Build table data
            tbl_headers = ["$\\bf{Pitch\\ Name}$", "$\\bf{Count\\%}$"]
            tbl_headers += [h for _, h, _, _ in _PITCH_TABLE_COLS
                            if _ in pitch_rows.columns or True]

            # Use all_pitches_df for league comparison if available
            league_df = pitches_df if all_pitches_df is None else all_pitches_df

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
                    # Always add exactly one cell per column
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
                    if higher_good is not None and pitch_name_col:
                        league_vals = (
                            league_df[league_df["pitch_type"] == pt][pp_col]
                            if pp_col in league_df.columns
                            else pd.Series()
                        )
                        league_vals = pd.to_numeric(
                            league_vals, errors="coerce"
                        ).dropna()
                        if not league_vals.empty:
                            lmean = league_vals.mean()
                            cmap = (_CMAP_GOOD if higher_good
                                    else _CMAP_BAD)
                            row_colors.append(
                                _get_table_cell_color(
                                    val, lmean, cmap))
                        else:
                            row_colors.append("#ffffff")
                    else:
                        row_colors.append("#ffffff")

                cell_text.append(row_data)
                cell_colors.append(row_colors)

            # Determine which headers we actually have
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
                # White text on dark colors
                r, g, b = mcolors.to_rgb(row_label_colors[i])
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                cell.set_text_props(
                    color="white" if luma < 0.5 else "black",
                    fontweight="bold",
                )

                # Set edge colors for all cells
                for j in range(n_cols):
                    tbl.get_celld()[(i + 1, j)].set_edgecolor("#cccccc")

        # ── Footer ────────────────────────────────────────────────────
        ax_footer = fig.add_subplot(gs[-1, 1:7])
        ax_footer.axis("off")
        ax_footer.text(0, 1, "By: @BachTalk2", ha="left", va="top",
                       fontsize=22, fontweight="bold")
        ax_footer.text(0.5, 1,
                       "Colour Coding Compares to League Average By Pitch",
                       ha="center", va="top", fontsize=14,
                       color="#666666")
        ax_footer.text(1, 1, "Data: Pitch Profiler\nImages: MLB, ESPN",
                       ha="right", va="top", fontsize=22)

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"pitching_summary_{safe}.png"
        fig.savefig(out, facecolor="white", dpi=150,
                    bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        log.info("Saved pitching summary: %s", out)
        return out

    except Exception:
        log.warning("plot_pitching_summary failed for %s", name,
                    exc_info=True)
        return None
