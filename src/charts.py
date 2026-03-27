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
from PIL import Image as _PILImage                                 # noqa: E402
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


# ── Clean White Theme ────────────────────────────────────────────────
WHITE_BG = "#ffffff"
WHITE_TEXT = "#222222"
WHITE_MUTED = "#888888"
WHITE_GRID = "#d0d0d0"
FOOTER_LEFT = "By: @BachTalk1"
FOOTER_RIGHT = "Data: Pitch Profiler"

# ── Watermark ───────────────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_WATERMARK_PATH = _ASSETS_DIR / "BachTalk.png"
_watermark_cache: "np.ndarray | None" = None


_watermark_cache_dark = None  # dark logo pixels on transparent bg (for light cards)

def _draw_watermark(fig, alpha=0.15, scale=0.45, dark_bg=True):
    """Overlay BachTalk logo as a watermark in the center of the figure."""
    global _watermark_cache, _watermark_cache_dark

    if dark_bg:
        # White logo for dark backgrounds
        if _watermark_cache is None:
            if not _WATERMARK_PATH.exists():
                return
            try:
                img = _PILImage.open(_WATERMARK_PATH).convert("RGBA")
                arr = np.array(img, dtype=np.float32)
                is_white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
                arr[is_white, 3] = 0
                not_transparent = arr[:, :, 3] > 0
                arr[not_transparent, 0] = 255
                arr[not_transparent, 1] = 255
                arr[not_transparent, 2] = 255
                _watermark_cache = arr.astype(np.uint8)
            except Exception:
                return
        cache = _watermark_cache
    else:
        # Dark logo for light/white backgrounds
        if _watermark_cache_dark is None:
            if not _WATERMARK_PATH.exists():
                return
            try:
                img = _PILImage.open(_WATERMARK_PATH).convert("RGBA")
                arr = np.array(img, dtype=np.float32)
                is_white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
                arr[is_white, 3] = 0
                # Keep original dark colors for light backgrounds
                _watermark_cache_dark = arr.astype(np.uint8)
            except Exception:
                return
        cache = _watermark_cache_dark

    if cache is None:
        return
    ax_wm = fig.add_axes([0.5 - scale / 2, 0.5 - scale / 2, scale, scale],
                         zorder=10)
    ax_wm.imshow(cache, alpha=alpha)
    ax_wm.set_facecolor("none")
    ax_wm.axis("off")


def _apply_white_theme(ax: plt.Axes) -> None:
    """Apply clean white theme to an axes."""
    ax.set_facecolor(WHITE_BG)
    ax.tick_params(colors=WHITE_TEXT, which="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(WHITE_GRID)
    ax.xaxis.label.set_color(WHITE_TEXT)
    ax.yaxis.label.set_color(WHITE_TEXT)
    ax.grid(True, color=WHITE_GRID, linewidth=0.5, alpha=0.4)


def _draw_confidence_ellipse(ax, x, y, color, n_std=1.5):
    """Draw a covariance confidence ellipse on ax."""
    from matplotlib.patches import Ellipse as _Ellipse
    import matplotlib.transforms as _transforms

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n = min(len(x), len(y))
    if n < 5:
        return
    x, y = x[:n], y[:n]

    try:
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = _Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
            facecolor=color, alpha=0.08, edgecolor=color,
            linewidth=1.5, linestyle="--",
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = (_transforms.Affine2D()
                  .rotate_deg(45)
                  .scale(scale_x, scale_y)
                  .translate(x.mean(), y.mean()))
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
    except (ValueError, np.linalg.LinAlgError):
        pass


def _draw_header(fig, name: str, player_id=None, subtitle="",
                 accent="#3a86ff"):
    """Draw a header strip at the top of a white-themed figure."""
    # Player name
    fig.text(0.5, 0.95, name, fontsize=22, fontweight="bold",
             color=WHITE_TEXT, ha="center", va="top",
             fontfamily="sans-serif")
    if subtitle:
        fig.text(0.5, 0.91, subtitle, fontsize=12, color=WHITE_MUTED,
                 ha="center", va="top", fontfamily="sans-serif")
    # Headshot (if available)
    if player_id:
        try:
            headshot = _fetch_headshot(player_id, accent=accent)
            if headshot is not None:
                ax_hs = fig.add_axes([0.02, 0.86, 0.10, 0.13])
                ax_hs.imshow(headshot)
                ax_hs.axis("off")
        except Exception:
            pass


def _draw_footer(fig):
    """Draw a footer strip at the bottom of a white-themed figure."""
    fig.text(0.08, 0.02, FOOTER_LEFT, fontsize=11, fontweight="bold",
             color=WHITE_TEXT, ha="left", va="bottom",
             fontfamily="sans-serif")
    fig.text(0.92, 0.02, FOOTER_RIGHT, fontsize=11, color=WHITE_MUTED,
             ha="right", va="bottom", fontfamily="sans-serif")


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

        fig = plt.figure(figsize=(12, 8), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.08, 0.10, 0.84, 0.72])
        _apply_white_theme(ax)

        pitch_types = sorted(df["pitch_type"].unique())
        for pt in pitch_types:
            if pt in _NOISE_PITCHES:
                continue
            subset = df[df["pitch_type"] == pt]
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)
            ax.scatter(
                subset["hb"], subset["ivb"],
                c=color, label=label, alpha=0.5, s=30, edgecolors="none",
                zorder=3,
            )
            _draw_confidence_ellipse(
                ax, subset["hb"].values, subset["ivb"].values, color
            )

        ax.axhline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Horizontal Break (in)", fontsize=12)
        ax.set_ylabel("Induced Vertical Break (in)", fontsize=12)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.legend(
            loc="upper right", fontsize=9, framealpha=0.9,
            facecolor="white", edgecolor="#cccccc",
        )

        _draw_header(fig, name, player_id=pitcher_id,
                     subtitle="Pitch Movement (Statcast)")
        _draw_footer(fig)

        out = SCREENSHOTS_DIR / f"movement_{pitcher_id}.png"
        _draw_watermark(fig)
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

        fig = plt.figure(figsize=(9, 10), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.10, 0.08, 0.80, 0.72])
        _apply_white_theme(ax)

        # Strike zone — solid outer box + 3×3 inner grid
        sz_left, sz_bot, sz_w, sz_h = -0.83, 1.5, 1.66, 2.0
        sz = Rectangle((sz_left, sz_bot), sz_w, sz_h, linewidth=2.5,
                        edgecolor="#333333", facecolor="none")
        ax.add_patch(sz)
        for i in range(1, 3):
            ax.plot([sz_left + i * sz_w / 3, sz_left + i * sz_w / 3],
                    [sz_bot, sz_bot + sz_h], color="#cccccc", linewidth=0.8)
            ax.plot([sz_left, sz_left + sz_w],
                    [sz_bot + i * sz_h / 3, sz_bot + i * sz_h / 3],
                    color="#cccccc", linewidth=0.8)

        # Home plate pentagon
        hp_w = 0.83
        hp_pts = np.array([
            [-hp_w, 0.2], [hp_w, 0.2], [hp_w, 0.35],
            [0, 0.5], [-hp_w, 0.35], [-hp_w, 0.2]
        ])
        ax.plot(hp_pts[:, 0], hp_pts[:, 1], color="#555555", linewidth=1.5)

        pitch_types = sorted(df["pitch_type"].unique())
        for pt in pitch_types:
            if pt in _NOISE_PITCHES:
                continue
            subset = df[df["pitch_type"] == pt]
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)
            ax.scatter(
                subset["plate_x"], subset["plate_z"],
                c=color, label=label, alpha=0.45, s=28, edgecolors="none",
                zorder=3,
            )
            if not anonymize:
                _draw_confidence_ellipse(
                    ax, subset["plate_x"].values, subset["plate_z"].values,
                    color, n_std=1.5,
                )

        ax.set_xlabel("Horizontal (ft from center)", fontsize=12)
        ax.set_ylabel("Height (ft)", fontsize=12)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")

        if not anonymize:
            ax.legend(
                loc="upper right", fontsize=9, framealpha=0.9,
                facecolor="white", edgecolor="#cccccc",
            )

        display_name = "Mystery Pitcher" if anonymize else name
        pid = None if anonymize else pitcher_id
        _draw_header(fig, display_name, player_id=pid,
                     subtitle="Pitch Locations (Statcast)")
        _draw_footer(fig)

        suffix = "anon" if anonymize else str(pitcher_id)
        out = SCREENSHOTS_DIR / f"locations_{suffix}.png"
        _draw_watermark(fig)
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

        pitch_types = [pt for pt in sorted(df["pitch_type"].unique())
                       if pt not in _NOISE_PITCHES]
        n = len(pitch_types)
        if n == 0:
            return None

        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig_h = 3.5 * rows + 2.5
        fig, axes = plt.subplots(rows, cols, figsize=(12, fig_h), dpi=150,
                                  squeeze=False)
        fig.set_facecolor(WHITE_BG)

        for idx, pt in enumerate(pitch_types):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.set_facecolor(WHITE_BG)
            subset = df[df["pitch_type"] == pt]
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)

            # Per-pitch colormap: white → pitch color
            from matplotlib.colors import LinearSegmentedColormap as _LSC
            cmap = _LSC.from_list("", ["#ffffff", color])
            ax.hexbin(
                subset["plate_x"], subset["plate_z"],
                gridsize=15, cmap=cmap, mincnt=1,
                extent=(-2.5, 2.5, 0, 5), alpha=0.85,
            )

            # Strike zone
            sz = Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2,
                            edgecolor="#333333", facecolor="none")
            ax.add_patch(sz)

            ax.set_title(label, fontsize=12, fontweight="bold", color=color)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(0, 5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7, colors="#555555")
            for spine in ax.spines.values():
                spine.set_color(WHITE_GRID)

            # Pitch count
            ax.text(0.95, 0.95, f"n={len(subset)}", transform=ax.transAxes,
                    fontsize=8, color=WHITE_MUTED, ha="right", va="top")

            # Hide inner tick labels
            if c > 0:
                ax.set_yticklabels([])
            if r < rows - 1:
                ax.set_xticklabels([])

        # Hide unused subplots
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.subplots_adjust(top=0.85, bottom=0.08, hspace=0.3, wspace=0.15)
        _draw_header(fig, name, player_id=pitcher_id,
                     subtitle="Pitch Location Heatmaps")
        _draw_footer(fig)

        out = SCREENSHOTS_DIR / f"heatmap_{pitcher_id}.png"
        _draw_watermark(fig)
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


def plot_percentile_rankings(name: str, season_df: "pd.DataFrame",
                             player_id: int | None = None) -> Path | None:
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

        if player_id is None:
            player_id = player.get("pitcher_id") or player.get("player_id")
            if player_id is not None:
                player_id = int(player_id)

        labels: list[str] = []
        percentiles: list[float] = []
        raw_values: list[str] = []

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
            # Format raw value
            v = float(player[col])
            if col in ("strike_out_percentage", "walk_percentage",
                        "whiff_rate", "chase_percentage", "barrel_percentage"):
                raw_values.append(f"{v * 100:.1f}%")
            elif col in ("era", "fip"):
                raw_values.append(f"{v:.2f}")
            else:
                raw_values.append(f"{v:.0f}")

        if not labels:
            return None

        n_bars = len(labels)
        fig_h = max(5.5, 1.0 * n_bars + 3.5)
        fig = plt.figure(figsize=(10, fig_h), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.18, 0.10, 0.72, 0.68])
        ax.set_facecolor(WHITE_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.set_xlim(0, 100)

        y_pos = np.arange(n_bars)
        bar_h = 0.55

        for i, (label, pctile, raw) in enumerate(
                zip(labels, percentiles, raw_values)):
            color = _pctile_color(pctile)

            # Gray track
            ax.barh(i, 100, height=bar_h, color="#e8e8e8",
                    edgecolor="none", zorder=1)
            # Filled bar
            ax.barh(i, pctile, height=bar_h, color=color,
                    edgecolor="none", zorder=2)

            # Stat label (left)
            ax.text(-2, i, label, ha="right", va="center",
                    fontsize=12, fontweight="bold", color=WHITE_TEXT)

            # Raw value inside bar
            val_x = max(pctile / 2, 5)
            val_color = "white" if pctile > 25 else WHITE_TEXT
            ax.text(val_x, i, raw, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=val_color, zorder=3)

            # Percentile rank (right)
            ax.text(102, i, f"{pctile:.0f}th", ha="left", va="center",
                    fontsize=11, fontweight="bold", color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * n_bars)
        ax.invert_yaxis()

        _draw_header(fig, name, player_id=player_id,
                     subtitle="Percentile Rankings (vs. Qualified Pitchers)")
        _draw_footer(fig)

        out = SCREENSHOTS_DIR / f"percentile_{name.replace(' ', '_').lower()}.png"
        _draw_watermark(fig)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved percentile rankings chart: %s", out)
        return out

    except Exception:
        log.warning("plot_percentile_rankings failed for %s", name, exc_info=True)
        return None


# ── Chart 5: Movement Profile (HB × IVB arsenal scatter) ─────────────

def plot_movement_profile(name: str, pitches_df: "pd.DataFrame",
                          player_id: int | None = None) -> Path | None:
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

        # Detect handedness for arm/glove side labels
        is_lhp = False
        if "p_throws" in prows.columns:
            hand = str(prows["p_throws"].iloc[0]).upper()
            is_lhp = hand == "L"

        # Get player_id from data if not provided
        if player_id is None:
            for id_col in ("pitcher_id", "player_id"):
                if id_col in prows.columns:
                    try:
                        player_id = int(prows[id_col].iloc[0])
                    except (TypeError, ValueError):
                        pass
                    break

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

        fig = plt.figure(figsize=(12, 9), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.10, 0.10, 0.80, 0.70])
        _apply_white_theme(ax)

        for _, row in grouped.iterrows():
            pt = str(row["pitch_type"])
            if pt in _NOISE_PITCHES:
                continue
            hb = float(row["hb"])
            ivb = float(row["ivb"])
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)

            # Marker size based on usage (min 150, max 600)
            usage = float(row.get("percentage_thrown", 0.1) or 0.1)
            size = max(150, min(600, usage * 1200))

            ax.scatter(hb, ivb, c=color, s=size, alpha=0.85,
                       edgecolors="black", linewidths=0.8, zorder=3)

            # Label with background box
            velo_str = ""
            if "velocity" in row.index and pd.notna(row["velocity"]):
                velo_str = f" ({float(row['velocity']):.1f})"
            ax.annotate(
                f"{label}{velo_str}", (hb, ivb),
                textcoords="offset points", xytext=(10, 10),
                fontsize=10, fontweight="bold", color=color,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5),
                zorder=4,
            )

        ax.axhline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizontal Break (in)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Induced Vertical Break (in)", fontsize=12, fontweight="bold")
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)

        # Arm side / Glove side labels
        arm_side = "← Arm Side" if not is_lhp else "Arm Side →"
        glove_side = "Glove Side →" if not is_lhp else "← Glove Side"
        ax.text(0.02, 0.02, glove_side if is_lhp else arm_side,
                transform=ax.transAxes, fontsize=9, color=WHITE_MUTED,
                ha="left", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.text(0.98, 0.02, arm_side if is_lhp else glove_side,
                transform=ax.transAxes, fontsize=9, color=WHITE_MUTED,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        _draw_header(fig, name, player_id=player_id,
                     subtitle="Movement Profile (Pitch Profiler)")
        _draw_footer(fig)

        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"movement_profile_{safe}.png"
        _draw_watermark(fig)
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
    level: str = "MLB",
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
            name_x, 0.88, f"{team_label}{MLB_SEASON} {level} Season",
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

        fig.text(0.04, 0.025, "@BachTalk1", fontsize=10, color=accent,
                 ha="left", va="center", fontweight="bold")
        fig.text(0.5, 0.025, "Pitch Profiler Data", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="center", va="center")
        fig.text(0.96, 0.025, f"{MLB_SEASON}", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="right", va="center")

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"pitcher_card_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=100,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        log.info("Saved pitcher card: %s", out)
        return out

    except Exception:
        log.warning("plot_pitcher_card failed for %s", name, exc_info=True)
        return None


# ── Chart 6b: MiLB Pitcher Card (prospect-focused design) ────────────

# MiLB left-panel stats: (column, display_label, lower_is_better)
_MILB_CARD_STATS = [
    ("strike_out_percentage", "K%", False),
    ("walk_percentage", "BB%", True),
    ("k_minus_bb", "K-BB%", False),
    ("whiff_rate", "Whiff%", False),
    ("chase_percentage", "Chase%", False),
    ("called_strikes_plus_whiffs_percentage", "CSW%", False),
    ("zone_percentage", "Zone%", False),
    ("first_pitch_strike_percentage", "FPK%", False),
    ("hard_hit_percentage", "Hard Hit%", True),
    ("ground_ball_percentage", "GB%", False),
]


def plot_milb_pitcher_card(
    name: str,
    season_df: "pd.DataFrame",
    pitches_df: "pd.DataFrame",
    team: str | None = None,
    player_id: int | None = None,
    level: str = "AAA",
) -> Path | None:
    """Render a MiLB prospect pitcher card (1200×675, dark theme).

    Left panel: prospect-relevant percentile bars (K%, BB%, K-BB%, Whiff%,
    Chase%, CSW%, Zone%, FPK%, Hard Hit%, GB%).
    Right panel: arsenal breakdown with velo, spin, movement, usage, whiff bars.
    Header: headshot, name, team, level badge, hero stats.
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

        from .milb_statcast import LEVEL_NAMES
        level_display = LEVEL_NAMES.get(level, level)

        # ── Compute percentiles ───────────────────────────────────────
        stat_labels: list[str] = []
        stat_values: list[str] = []
        stat_raw: list[float] = []
        pctiles: list[float] = []

        for col, label, ascending in _MILB_CARD_STATS:
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
                        "whiff_rate", "chase_percentage", "k_minus_bb",
                        "called_strikes_plus_whiffs_percentage",
                        "zone_percentage", "first_pitch_strike_percentage",
                        "hard_hit_percentage", "ground_ball_percentage",
                        "barrel_percentage"):
                stat_values.append(f"{raw_f * 100:.1f}%")
            else:
                stat_values.append(f"{raw_f:.2f}")
            pctiles.append(pctile)

        if not stat_labels:
            return None

        # ── Arsenal data (expanded) ──────────────────────────────────
        arsenal_rows: list[dict] = []
        if pitches_df is not None and not pitches_df.empty:
            pitch_name_col = None
            for c in ("pitcher_name", "player_name", "name"):
                if c in pitches_df.columns:
                    pitch_name_col = c
                    break
            if pitch_name_col and "pitch_type" in pitches_df.columns:
                prows = pitches_df[pitches_df[pitch_name_col] == name].copy()
                prows = prows[~prows["pitch_type"].isin(_NOISE_PITCHES)]
                if not prows.empty:
                    num_cols = ["velocity", "whiff_rate", "percentage_thrown",
                                "spin_rate", "ivb", "hb", "chase_percentage"]
                    for nc in num_cols:
                        if nc in prows.columns:
                            prows[nc] = pd.to_numeric(prows[nc], errors="coerce")

                    agg_map = {}
                    for c in ["velocity", "whiff_rate", "percentage_thrown",
                              "spin_rate", "ivb", "hb", "chase_percentage"]:
                        if c in prows.columns:
                            agg_map[c] = "mean" if c != "percentage_thrown" else "sum"

                    grouped = prows.groupby("pitch_type", as_index=False).agg(
                        agg_map if agg_map else {"pitch_type": "first"},
                    )

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
                        spin = row.get("spin_rate", None)
                        ivb = row.get("ivb", None)
                        hb = row.get("hb", None)
                        chase = row.get("chase_percentage", None)
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
                        try:
                            spin_f = float(spin) if pd.notna(spin) else None
                        except (TypeError, ValueError):
                            spin_f = None
                        try:
                            ivb_f = float(ivb) if pd.notna(ivb) else None
                        except (TypeError, ValueError):
                            ivb_f = None
                        try:
                            hb_f = float(hb) if pd.notna(hb) else None
                        except (TypeError, ValueError):
                            hb_f = None
                        try:
                            chase_f = float(chase) * 100 if pd.notna(chase) else None
                        except (TypeError, ValueError):
                            chase_f = None

                        color = PITCH_COLORS.get(pcode, DEFAULT_PITCH_COLOR)
                        display = PITCH_NAMES.get(pcode, pcode)
                        arsenal_rows.append({
                            "name": display, "code": pcode,
                            "velo": velo_f, "spin": spin_f,
                            "ivb": ivb_f, "hb": hb_f,
                            "whiff": whiff_f, "chase": chase_f,
                            "usage": usage_f, "color": color,
                        })
                    arsenal_rows.sort(key=lambda r: r["usage"], reverse=True)

        # ── Build figure ──────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
        fig.set_facecolor(CARD_BG)

        # Background noise texture
        noise = np.random.default_rng(42).uniform(0.04, 0.07, (68, 120))
        bg_ax = fig.add_axes([0, 0, 1, 1])
        bg_ax.imshow(noise, aspect="auto", cmap="gray", alpha=0.03,
                     extent=[0, 1, 0, 1])
        bg_ax.axis("off")

        # Header gradient
        _draw_gradient_rect(fig, [0, 0.78, 1, 0.22], accent, CARD_BG, alpha=0.25)

        # Accent stripe at top
        stripe = fig.add_axes([0, 0.97, 1, 0.03])
        stripe.set_xlim(0, 1)
        stripe.set_ylim(0, 1)
        stripe.add_patch(Rectangle((0, 0), 1, 1, color=accent))
        stripe.axis("off")

        # Headshot
        name_x = 0.04
        if player_id:
            hs_arr = _fetch_headshot(player_id, accent)
            if hs_arr is not None:
                name_x = 0.18
                hs_ax = fig.add_axes([0.02, 0.75, 0.13, 0.22])
                hs_ax.imshow(hs_arr)
                hs_ax.axis("off")

        # Player name
        shadow = [patheffects.withStroke(linewidth=4, foreground=CARD_BG)]
        fig.text(
            name_x, 0.95, name,
            fontsize=28, fontweight="bold", color=CARD_TEXT,
            ha="left", va="top", path_effects=shadow,
        )

        # Team + level subtitle
        team_label = f"{team}  |  " if team else ""
        fig.text(
            name_x, 0.88, f"{team_label}{MLB_SEASON} {level_display}",
            fontsize=13, color=CARD_TEXT_MUTED,
            ha="left", va="top",
        )

        # Level badge (colored pill)
        badge_x = 0.88
        badge_ax = fig.add_axes([badge_x, 0.90, 0.10, 0.06])
        badge_ax.set_xlim(0, 1)
        badge_ax.set_ylim(0, 1)
        badge_ax.add_patch(FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0,rounding_size=0.3",
            facecolor=accent, edgecolor="none", alpha=0.9,
        ))
        badge_ax.text(0.5, 0.5, level, fontsize=12, fontweight="bold",
                      color="white", ha="center", va="center")
        badge_ax.axis("off")

        # ── Hero stat boxes (K%, Whiff%, CSW%, K-BB%) ───────────────
        hero_map = {"K%": 0, "Whiff%": 1, "CSW%": 2, "K-BB%": 3}
        hero_stats = []
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
            box_ax.plot([0.15, 0.85], [0.98, 0.98], color=_pctile_color(pct),
                        linewidth=3, solid_capstyle="round")
            box_ax.text(0.5, 0.62, val, fontsize=14, fontweight="bold",
                        color=CARD_TEXT, ha="center", va="center")
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

        # ── Section header: PROSPECT PROFILE ────────────────────────
        fig.text(0.05, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.065, 0.71, "PROSPECT PROFILE", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Left panel: percentile bars ───────────────────────────────
        left_ax = fig.add_axes([0.04, 0.07, 0.43, 0.61])
        left_ax.set_xlim(0, 1.12)
        left_ax.set_ylim(-0.5, len(stat_labels) - 0.5)
        left_ax.invert_yaxis()
        left_ax.set_facecolor("none")
        for spine in left_ax.spines.values():
            spine.set_visible(False)
        left_ax.tick_params(left=False, bottom=False, labelbottom=False,
                            labelleft=False)

        bar_track_w = 0.58
        bar_x0 = 0.20
        bar_h = 0.026

        for i, (lbl, val, pct) in enumerate(zip(stat_labels, stat_values, pctiles)):
            y = i
            color = _pctile_color(pct)

            left_ax.text(0.0, y, lbl, fontsize=10, fontweight="bold",
                         color=CARD_TEXT, va="center", ha="left")

            _rounded_bar(left_ax, bar_x0, y, bar_track_w, bar_h,
                         CARD_BORDER, alpha=0.5)
            fill_w = max((pct / 100) * bar_track_w, 0.008)
            _rounded_bar(left_ax, bar_x0, y, fill_w, bar_h, color, alpha=0.9)

            val_x = bar_x0 + fill_w - 0.01 if pct > 25 else bar_x0 + fill_w + 0.01
            val_ha = "right" if pct > 25 else "left"
            val_color = "white" if pct > 25 else CARD_TEXT
            left_ax.text(val_x, y, val, fontsize=9, fontweight="bold",
                         color=val_color, va="center", ha=val_ha)

            left_ax.text(bar_x0 + bar_track_w + 0.03, y,
                         f"{pct:.0f}th", fontsize=9, fontweight="bold",
                         color=color, va="center", ha="left")

        # ── Vertical divider ──────────────────────────────────────────
        vdiv = fig.add_axes([0.505, 0.08, 0.003, 0.63])
        vdiv.set_xlim(0, 1)
        vdiv.set_ylim(0, 1)
        vgrad = np.linspace(0, 1, 256).reshape(-1, 1)
        cmap_v = LinearSegmentedColormap.from_list("vd", ["#00000000", CARD_BORDER, "#00000000"])
        vdiv.imshow(vgrad, aspect="auto", cmap=cmap_v, extent=[0, 1, 0, 1])
        vdiv.axis("off")

        # ── Section header: ARSENAL ───────────────────────────────────
        fig.text(0.53, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.545, 0.71, "ARSENAL", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Right panel: expanded arsenal ─────────────────────────────
        if arsenal_rows:
            right_ax = fig.add_axes([0.52, 0.07, 0.46, 0.61])
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

                # Color dot (centered vertically)
                right_ax.plot(0.02, y - 0.08, "o", color=c, markersize=9,
                              markeredgecolor="white", markeredgewidth=0.8)

                # Pitch name + usage % on same line
                usage_pct = pitch["usage"] * 100
                right_ax.text(0.06, y - 0.28, pitch["name"],
                              fontsize=10, fontweight="bold", color=c,
                              va="center", ha="left")
                right_ax.text(0.30, y - 0.28,
                              f"{usage_pct:.0f}%",
                              fontsize=9, fontweight="bold",
                              color=CARD_TEXT, va="center", ha="left")

                # Stat line below name: Velo | Spin | IVB/HB
                stat_parts = []
                if pitch["velo"]:
                    stat_parts.append(f"{pitch['velo']:.1f}")
                if pitch["spin"]:
                    stat_parts.append(f"{pitch['spin']:.0f} rpm")
                if pitch["ivb"] is not None and pitch["hb"] is not None:
                    stat_parts.append(f"{pitch['ivb']:+.1f}\" iVB  {pitch['hb']:+.1f}\" HB")
                stat_text = "  |  ".join(stat_parts) if stat_parts else ""
                right_ax.text(0.06, y - 0.08, stat_text,
                              fontsize=7, color=CARD_TEXT_MUTED,
                              va="center", ha="left")

                # Whiff + Chase as compact bars side by side below stats
                bar_x0 = 0.06
                bar_max_w = 0.38

                # Whiff bar
                if pitch["whiff"] is not None:
                    whiff_w = (pitch["whiff"] / 50) * bar_max_w
                    whiff_w = min(whiff_w, bar_max_w)
                    _rounded_bar(right_ax, bar_x0, y + 0.12,
                                 bar_max_w, 0.016, CARD_BORDER, alpha=0.35)
                    _rounded_bar(right_ax, bar_x0, y + 0.12,
                                 max(whiff_w, 0.004), 0.016, c, alpha=0.65)
                    right_ax.text(bar_x0 + bar_max_w + 0.015, y + 0.12,
                                  f"{pitch['whiff']:.0f}% whiff",
                                  fontsize=7, color=CARD_TEXT_MUTED,
                                  va="center", ha="left")

                # Chase bar
                if pitch["chase"] is not None:
                    chase_w = (pitch["chase"] / 50) * bar_max_w
                    chase_w = min(chase_w, bar_max_w)
                    _rounded_bar(right_ax, bar_x0, y + 0.30,
                                 bar_max_w, 0.016, CARD_BORDER, alpha=0.25)
                    _rounded_bar(right_ax, bar_x0, y + 0.30,
                                 max(chase_w, 0.004), 0.016, c, alpha=0.40)
                    right_ax.text(bar_x0 + bar_max_w + 0.015, y + 0.30,
                                  f"{pitch['chase']:.0f}% chase",
                                  fontsize=7, color=CARD_TEXT_MUTED,
                                  va="center", ha="left")

        # ── Footer ────────────────────────────────────────────────────
        foot_ax = fig.add_axes([0.03, 0.0, 0.94, 0.003])
        foot_ax.set_xlim(0, 1)
        foot_ax.set_ylim(0, 1)
        foot_ax.add_patch(Rectangle((0, 0), 1, 1, color=CARD_BORDER))
        foot_ax.axis("off")

        fig.text(0.04, 0.025, "@BachTalk1", fontsize=10, color=accent,
                 ha="left", va="center", fontweight="bold")
        fig.text(0.5, 0.025, "Data: Baseball Savant", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="center", va="center")
        fig.text(0.96, 0.025, f"{MLB_SEASON} {level_display}", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="right", va="center")

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"milb_pitcher_card_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=100,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        log.info("Saved MiLB pitcher card: %s", out)
        return out

    except Exception:
        log.warning("plot_milb_pitcher_card failed for %s", name, exc_info=True)
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

# MiLB pitch table — replaces RV/100, wOBA, Stuff+ with Savant-derived stats
_MILB_PITCH_TABLE_COLS = [
    ("velocity", "$\\bf{Velo}$", ".1f", True),
    ("ivb", "$\\bf{iVB}$", ".1f", None),
    ("hb", "$\\bf{HB}$", ".1f", None),
    ("spin_rate", "$\\bf{Spin}$", ".0f", None),
    ("release_extension", "$\\bf{Ext.}$", ".1f", True),
    ("csw", "$\\bf{CSW\\%}$", ".1%", True),
    ("zone_rate", "$\\bf{Zone\\%}$", ".1%", None),
    ("swing_rate", "$\\bf{Swing\\%}$", ".1%", None),
    ("avg_exit_velo", "$\\bf{EV}$", ".1f", False),
    ("hard_hit_rate", "$\\bf{HH\\%}$", ".1%", False),
    ("xba", "$\\bf{xBA}$", ".3f", False),
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

# MiLB season overview — replaces Stuff+/Pitching+ with prospect-relevant stats
_MILB_SUMMARY_STATS = [
    ("total_pitches", "$\\bf{Pitches}$", ".0f"),
    ("batters_faced", "$\\bf{PA}$", ".0f"),
    ("strike_out_percentage", "$\\bf{K\\%}$", ".1%"),
    ("walk_percentage", "$\\bf{BB\\%}$", ".1%"),
    ("k_minus_bb", "$\\bf{K-BB\\%}$", ".1%"),
    ("whiff_rate", "$\\bf{Whiff\\%}$", ".1%"),
    ("chase_percentage", "$\\bf{Chase\\%}$", ".1%"),
    ("called_strikes_plus_whiffs_percentage", "$\\bf{CSW\\%}$", ".1%"),
    ("zone_percentage", "$\\bf{Zone\\%}$", ".1%"),
    ("hard_hit_percentage", "$\\bf{HH\\%}$", ".1%"),
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
    level: str = "MLB",
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
                    # MiLB-specific columns
                    "csw", "zone_rate", "swing_rate",
                    "avg_exit_velo", "hard_hit_rate", "xba",
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
        ax_bio.text(0.5, 0.10, f"{MLB_SEASON} {level} Season", va="top",
                    ha="center", fontsize=26, fontstyle="italic",
                    color="#666666")

        # Team logo (MLB only — MiLB uses parent org logos which can be misleading)
        ax_logo.axis("off")
        if team and level == "MLB":
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
        summary_stats = _MILB_SUMMARY_STATS if level != "MLB" else _SUMMARY_STATS
        for col, header, fmt in summary_stats:
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
            pitch_table_cols = _MILB_PITCH_TABLE_COLS if level != "MLB" else _PITCH_TABLE_COLS
            tbl_headers = ["$\\bf{Pitch\\ Name}$", "$\\bf{Count\\%}$"]
            tbl_headers += [h for _, h, _, _ in pitch_table_cols
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

                for pp_col, _, fmt, higher_good in pitch_table_cols:
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
            for pp_col, header, _, _ in pitch_table_cols:
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
        ax_footer.text(0, 1, "By: @BachTalk1", ha="left", va="top",
                       fontsize=22, fontweight="bold")
        ax_footer.text(0.5, 1,
                       "Colour Coding Compares to League Average By Pitch",
                       ha="center", va="top", fontsize=14,
                       color="#666666")
        data_src = "Data: Baseball Savant" if level != "MLB" else "Data: Pitch Profiler"
        ax_footer.text(1, 1, f"{data_src}\nImages: MLB, ESPN",
                       ha="right", va="top", fontsize=22)

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"pitching_summary_{safe}.png"
        _draw_watermark(fig, alpha=0.08, dark_bg=False)
        fig.savefig(out, facecolor="white", dpi=150,
                    bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        log.info("Saved pitching summary: %s", out)
        return out

    except Exception:
        log.warning("plot_pitching_summary failed for %s", name,
                    exc_info=True)
        return None


# ── Chart 8: Release Point Plot (catcher perspective) ─────────────────

def plot_release_points(pitcher_id: int, name: str) -> Path | None:
    """Release point scatter from catcher's perspective using Statcast data."""
    try:
        df = fetch_statcast_pitches(pitcher_id)
        if df is None:
            return None

        needed = {"release_pos_x", "release_pos_z", "pitch_type", "p_throws"}
        if not needed.issubset(df.columns):
            return None

        df = df.dropna(subset=["release_pos_x", "release_pos_z", "pitch_type"])
        if df.empty:
            return None

        df = df.copy()
        pitcher_hand = df["p_throws"].iloc[0] if "p_throws" in df.columns else "R"

        fig = plt.figure(figsize=(10, 10), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.10, 0.08, 0.80, 0.74])
        _apply_white_theme(ax)

        # Mound/rubber representation
        mound = Circle((0, 10 / 12), radius=1.5, edgecolor="#a63b17",
                        facecolor="#a63b17", alpha=0.15, zorder=1)
        ax.add_patch(mound)
        rubber = Rectangle((-0.5, 9 / 12), 1.0, 1 / 6, edgecolor="#555555",
                           facecolor="white", linewidth=1.5, zorder=2)
        ax.add_patch(rubber)

        pitch_types = sorted(df["pitch_type"].unique())
        for pt in pitch_types:
            if pt in _NOISE_PITCHES:
                continue
            subset = df[df["pitch_type"] == pt]
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)

            x_vals = subset["release_pos_x"]
            z_vals = subset["release_pos_z"]

            ax.scatter(
                x_vals, z_vals,
                c=color, label=label, alpha=0.5, s=35,
                edgecolors="black", linewidths=0.3, zorder=3,
            )
            _draw_confidence_ellipse(
                ax, x_vals.values, z_vals.values, color, n_std=1.5
            )

        ax.axhline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="#b0b0b0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Horizontal Release (ft)", fontsize=12)
        ax.set_ylabel("Vertical Release (ft)", fontsize=12)
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 8)

        # Arm/glove side labels (catcher's perspective: positive x = catcher's right)
        # RHP releases from catcher's left (negative x), LHP from catcher's right (positive x)
        if pitcher_hand == "R":
            ax.text(-3.8, 0.15, "\u2190 Arm Side (RHP)", fontstyle="italic",
                    fontsize=10, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))
            ax.text(3.8, 0.15, "Glove Side \u2192", fontstyle="italic",
                    fontsize=10, ha="right", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))
        else:
            ax.text(-3.8, 0.15, "\u2190 Glove Side", fontstyle="italic",
                    fontsize=10, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))
            ax.text(3.8, 0.15, "Arm Side (LHP) \u2192", fontstyle="italic",
                    fontsize=10, ha="right", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))

        ax.legend(
            loc="upper right", fontsize=9, framealpha=0.9,
            facecolor="white", edgecolor="#cccccc",
        )

        _draw_header(fig, name, player_id=pitcher_id,
                     subtitle="Release Points — Catcher Perspective (Statcast)")
        _draw_footer(fig)

        out = SCREENSHOTS_DIR / f"release_{pitcher_id}.png"
        _draw_watermark(fig)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved release point chart: %s", out)
        return out

    except Exception:
        log.warning("plot_release_points failed for %s", name, exc_info=True)
        return None


# ── Chart 9: Velocity Distribution (violin plot by pitch type) ────────

def plot_velocity_distribution(pitcher_id: int, name: str) -> Path | None:
    """Violin/box plot showing velocity distributions per pitch type."""
    try:
        df = fetch_statcast_pitches(pitcher_id)
        if df is None:
            return None

        needed = {"release_speed", "pitch_type"}
        if not needed.issubset(df.columns):
            return None

        df = df.dropna(subset=["release_speed", "pitch_type"])
        df = df[~df["pitch_type"].isin(_NOISE_PITCHES)]
        if df.empty:
            return None

        # Order pitch types by median velocity (fastest first)
        medians = df.groupby("pitch_type")["release_speed"].median().sort_values(ascending=False)
        ordered = list(medians.index)
        if not ordered:
            return None

        fig = plt.figure(figsize=(12, 8), dpi=150)
        fig.set_facecolor(WHITE_BG)

        ax = fig.add_axes([0.08, 0.10, 0.84, 0.72])
        _apply_white_theme(ax)

        positions = list(range(len(ordered)))
        colors = [_TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR) for pt in ordered]

        for i, pt in enumerate(ordered):
            subset = df[df["pitch_type"] == pt]["release_speed"].values
            if len(subset) < 3:
                continue
            color = _TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR)
            label = _TJ_NAME.get(pt, pt)

            # Violin
            parts = ax.violinplot(subset, positions=[i], showmedians=False,
                                  showextrema=False, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.3)

            # Box overlay
            q1, med, q3 = np.percentile(subset, [25, 50, 75])
            ax.vlines(i, q1, q3, color=color, linewidth=4, zorder=4)
            ax.scatter([i], [med], color="white", s=30, zorder=5,
                       edgecolors=color, linewidths=1.5)

            # Median label
            ax.text(i, med + 0.3, f"{med:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)

            # Pitch count
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] else min(subset) - 2,
                    f"n={len(subset)}", ha="center", va="top",
                    fontsize=7, color=WHITE_MUTED)

        ax.set_xticks(positions)
        ax.set_xticklabels([_TJ_NAME.get(pt, pt) for pt in ordered],
                           fontsize=10, fontweight="bold")
        # Color each x-tick label
        for ticklabel, pt in zip(ax.get_xticklabels(), ordered):
            ticklabel.set_color(_TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR))

        ax.set_ylabel("Velocity (mph)", fontsize=12)
        ax.set_xlabel("")

        _draw_header(fig, name, player_id=pitcher_id,
                     subtitle="Velocity Distribution by Pitch Type (Statcast)")
        _draw_footer(fig)

        out = SCREENSHOTS_DIR / f"velo_dist_{pitcher_id}.png"
        _draw_watermark(fig)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved velocity distribution chart: %s", out)
        return out

    except Exception:
        log.warning("plot_velocity_distribution failed for %s", name,
                    exc_info=True)
        return None


# ── Chart 10: Arsenal Usage Breakdown (horizontal bars) ───────────────

def plot_arsenal_usage(name: str, pitches_df: "pd.DataFrame",
                       player_id: int | None = None) -> Path | None:
    """Horizontal bar chart of pitch type usage with velo + whiff overlays."""
    try:
        import pandas as pd

        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in pitches_df.columns:
                name_col = c
                break
        if not name_col:
            return None

        pdf = pitches_df[pitches_df[name_col] == name].copy()
        if pdf.empty:
            return None

        # Get pitch type column
        pt_col = "pitch_type" if "pitch_type" in pdf.columns else "pitch_name"
        if pt_col not in pdf.columns:
            return None

        # Clean noise
        pdf = pdf[~pdf[pt_col].isin(_NOISE_PITCHES)]
        if pdf.empty:
            return None

        # Get usage percentage
        usage_col = None
        for c in ("percentage_thrown", "pitch_percent", "usage"):
            if c in pdf.columns:
                usage_col = c
                break
        if not usage_col:
            return None

        pdf = pdf.sort_values(usage_col, ascending=True)

        fig = plt.figure(figsize=(12, 8), dpi=150)
        fig.set_facecolor(WHITE_BG)
        ax = fig.add_axes([0.18, 0.10, 0.74, 0.72])
        _apply_white_theme(ax)

        pitch_types = pdf[pt_col].tolist()
        usages = pdf[usage_col].tolist()

        # Convert to percentages if in decimal form
        if all(u <= 1 for u in usages if pd.notna(u)):
            usages = [u * 100 for u in usages]

        colors = [_TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR) for pt in pitch_types]
        labels = [_TJ_NAME.get(pt, pt) for pt in pitch_types]

        bars = ax.barh(range(len(pitch_types)), usages, color=colors, height=0.6,
                       edgecolor="white", linewidth=0.5, alpha=0.85)

        ax.set_yticks(range(len(pitch_types)))
        ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
        for ticklabel, pt in zip(ax.get_yticklabels(), pitch_types):
            ticklabel.set_color(_TJ_COLOUR.get(pt, DEFAULT_PITCH_COLOR))

        # Add usage percentage + velocity + whiff on each bar
        for i, (pt, usage) in enumerate(zip(pitch_types, usages)):
            row = pdf[pdf[pt_col] == pt].iloc[0]

            # Usage label
            ax.text(usage + 0.5, i, f"{usage:.1f}%", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=WHITE_TEXT)

            # Velocity
            velo = None
            for vc in ("velocity", "avg_speed", "release_speed"):
                if vc in row.index:
                    try:
                        velo = float(row[vc])
                    except (TypeError, ValueError):
                        pass
                    break
            # Whiff
            whiff = None
            for wc in ("whiff_rate", "whiff_percent"):
                if wc in row.index:
                    try:
                        w = float(row[wc])
                        whiff = w * 100 if w <= 1 else w
                    except (TypeError, ValueError):
                        pass
                    break

            detail_parts = []
            if velo is not None:
                detail_parts.append(f"{velo:.1f} mph")
            if whiff is not None:
                detail_parts.append(f"{whiff:.0f}% whiff")
            if detail_parts:
                detail = " | ".join(detail_parts)
                ax.text(max(usages) * 0.98, i, detail, va="center",
                        ha="right", fontsize=8, color=WHITE_MUTED,
                        fontstyle="italic")

        ax.set_xlabel("Usage %", fontsize=12)
        ax.set_xlim(0, max(usages) * 1.3)

        _draw_header(fig, name, player_id=player_id,
                     subtitle="Arsenal Usage Breakdown")
        _draw_footer(fig)

        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"arsenal_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved arsenal usage chart: %s", out)
        return out

    except Exception:
        log.warning("plot_arsenal_usage failed for %s", name, exc_info=True)
        return None


# ── Traditional MiLB Stats (non-AAA levels) ──────────────────────────

# Stats for the traditional pitcher card percentile bars
_TRAD_CARD_STATS = [
    # (api_column, display_label, lower_is_better)
    ("era", "ERA", True),
    ("whip", "WHIP", True),
    ("strikeoutsPer9Inn", "K/9", False),
    ("walksPer9Inn", "BB/9", True),
    ("k_minus_bb", "K-BB%", False),
    ("homeRunsPer9", "HR/9", True),
    ("strikeoutWalkRatio", "K/BB", False),
    ("avg", "BAVG", True),
    ("groundOutsToAirouts", "GO/AO", False),
    ("hitsPer9Inn", "H/9", True),
]

# Stats for the traditional summary top table
_TRAD_SUMMARY_STATS = [
    ("inningsPitched", "$\\bf{IP}$", ".1f"),
    ("era", "$\\bf{ERA}$", ".2f"),
    ("whip", "$\\bf{WHIP}$", ".2f"),
    ("strikeoutsPer9Inn", "$\\bf{K/9}$", ".2f"),
    ("walksPer9Inn", "$\\bf{BB/9}$", ".2f"),
    ("k_pct", "$\\bf{K\\%}$", ".1%"),
    ("bb_pct", "$\\bf{BB\\%}$", ".1%"),
    ("homeRunsPer9", "$\\bf{HR/9}$", ".2f"),
    ("avg", "$\\bf{BAVG}$", ".3f"),
    ("groundOutsToAirouts", "$\\bf{GO/AO}$", ".2f"),
]

# Radar chart axes
_RADAR_STATS = [
    # (column, label, higher_is_better)
    ("strikeoutsPer9Inn", "K/9", True),
    ("walksPer9Inn", "BB/9", False),
    ("era", "ERA", False),
    ("whip", "WHIP", False),
    ("homeRunsPer9", "HR/9", False),
    ("groundOutsToAirouts", "GO/AO", True),
]


def plot_traditional_pitcher_card(
    name: str,
    season_df: "pd.DataFrame",
    player_id: int | None = None,
    team: str | None = None,
    level: str = "AA",
    game_log: list[dict] | None = None,
    league_avgs: dict | None = None,
) -> Path | None:
    """Render a traditional-stats MiLB pitcher card (1200x675, dark theme).

    Left panel: percentile bars using traditional stats.
    Right panel: game log trend chart (top) + radar chart (bottom).
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
        if matches.empty and player_id is not None:
            matches = season_df[season_df["player_id"] == player_id]
        if matches.empty:
            return None
        player = matches.iloc[0]

        # ── Team colour accent ────────────────────────────────────────
        if not team:
            for tc in ("team", "team_abbreviation"):
                if tc in player.index and player[tc]:
                    team = str(player[tc]).upper()
                    break
        accent = TEAM_COLORS.get(team or "", "#3a86ff")

        from .milb_statcast import LEVEL_NAMES
        level_display = LEVEL_NAMES.get(level, level)

        # ── Compute percentiles ───────────────────────────────────────
        stat_labels: list[str] = []
        stat_values: list[str] = []
        stat_raw: list[float] = []
        pctiles: list[float] = []

        for col, label, lower_better in _TRAD_CARD_STATS:
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
            if pd.isna(raw_f):
                continue

            pctile = (vals < raw_f).sum() / len(vals) * 100
            if lower_better:
                pctile = 100 - pctile

            stat_labels.append(label)
            stat_raw.append(raw_f)
            if col in ("k_pct", "bb_pct", "k_minus_bb", "strikePercentage"):
                stat_values.append(f"{raw_f * 100:.1f}%")
            elif col in ("era", "whip", "fip"):
                stat_values.append(f"{raw_f:.2f}")
            elif col == "avg":
                stat_values.append(f"{raw_f:.3f}")
            else:
                stat_values.append(f"{raw_f:.2f}")
            pctiles.append(pctile)

        if not stat_labels:
            return None

        # ── Build figure ──────────────────────────────────────────────
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
        fig.set_facecolor(CARD_BG)

        # Background noise texture
        noise = np.random.default_rng(42).uniform(0.04, 0.07, (68, 120))
        bg_ax = fig.add_axes([0, 0, 1, 1])
        bg_ax.imshow(noise, aspect="auto", cmap="gray", alpha=0.03,
                     extent=[0, 1, 0, 1])
        bg_ax.axis("off")

        # Header gradient
        _draw_gradient_rect(fig, [0, 0.78, 1, 0.22], accent, CARD_BG, alpha=0.25)

        # Accent stripe at top
        stripe = fig.add_axes([0, 0.97, 1, 0.03])
        stripe.set_xlim(0, 1)
        stripe.set_ylim(0, 1)
        stripe.add_patch(Rectangle((0, 0), 1, 1, color=accent))
        stripe.axis("off")

        # Headshot
        name_x = 0.04
        if player_id:
            hs_arr = _fetch_headshot(player_id, accent)
            if hs_arr is not None:
                name_x = 0.18
                hs_ax = fig.add_axes([0.02, 0.75, 0.13, 0.22])
                hs_ax.imshow(hs_arr)
                hs_ax.axis("off")

        # Player name
        shadow = [patheffects.withStroke(linewidth=4, foreground=CARD_BG)]
        fig.text(
            name_x, 0.95, name,
            fontsize=28, fontweight="bold", color=CARD_TEXT,
            ha="left", va="top", path_effects=shadow,
        )

        # Team + level subtitle
        team_label = f"{team}  |  " if team else ""
        fig.text(
            name_x, 0.88, f"{team_label}{MLB_SEASON} {level_display}",
            fontsize=13, color=CARD_TEXT_MUTED,
            ha="left", va="top",
        )

        # Level badge
        badge_ax = fig.add_axes([0.88, 0.90, 0.10, 0.06])
        badge_ax.set_xlim(0, 1)
        badge_ax.set_ylim(0, 1)
        badge_ax.add_patch(FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0,rounding_size=0.3",
            facecolor=accent, edgecolor="none", alpha=0.9,
        ))
        badge_ax.text(0.5, 0.5, level, fontsize=12, fontweight="bold",
                      color="white", ha="center", va="center")
        badge_ax.axis("off")

        # ── Hero stat boxes (ERA, K/9, WHIP, K/BB) ───────────────────
        hero_map = {"ERA": 0, "K/9": 1, "WHIP": 2, "K/BB": 3}
        hero_stats = []
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
            box_ax.plot([0.15, 0.85], [0.98, 0.98], color=_pctile_color(pct),
                        linewidth=3, solid_capstyle="round")
            box_ax.text(0.5, 0.62, val, fontsize=14, fontweight="bold",
                        color=CARD_TEXT, ha="center", va="center")
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

        # ── Section header: TRADITIONAL PROFILE ──────────────────────
        fig.text(0.05, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.065, 0.71, "TRADITIONAL PROFILE", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Left panel: percentile bars ───────────────────────────────
        left_ax = fig.add_axes([0.04, 0.07, 0.43, 0.61])
        left_ax.set_xlim(0, 1.12)
        left_ax.set_ylim(-0.5, len(stat_labels) - 0.5)
        left_ax.invert_yaxis()
        left_ax.set_facecolor("none")
        for spine in left_ax.spines.values():
            spine.set_visible(False)
        left_ax.tick_params(left=False, bottom=False, labelbottom=False,
                            labelleft=False)

        bar_track_w = 0.58
        bar_x0 = 0.20
        bar_h = 0.026

        for i, (lbl, val, pct) in enumerate(zip(stat_labels, stat_values, pctiles)):
            y = i
            color = _pctile_color(pct)

            left_ax.text(0.0, y, lbl, fontsize=10, fontweight="bold",
                         color=CARD_TEXT, va="center", ha="left")

            _rounded_bar(left_ax, bar_x0, y, bar_track_w, bar_h,
                         CARD_BORDER, alpha=0.5)
            fill_w = max((pct / 100) * bar_track_w, 0.008)
            _rounded_bar(left_ax, bar_x0, y, fill_w, bar_h, color, alpha=0.9)

            val_x = bar_x0 + fill_w - 0.01 if pct > 25 else bar_x0 + fill_w + 0.01
            val_ha = "right" if pct > 25 else "left"
            val_color = "white" if pct > 25 else CARD_TEXT
            left_ax.text(val_x, y, val, fontsize=9, fontweight="bold",
                         color=val_color, va="center", ha=val_ha)

            left_ax.text(bar_x0 + bar_track_w + 0.03, y,
                         f"{pct:.0f}th", fontsize=9, fontweight="bold",
                         color=color, va="center", ha="left")

        # ── Vertical divider ──────────────────────────────────────────
        vdiv = fig.add_axes([0.505, 0.08, 0.003, 0.63])
        vdiv.set_xlim(0, 1)
        vdiv.set_ylim(0, 1)
        vgrad = np.linspace(0, 1, 256).reshape(-1, 1)
        cmap_v = LinearSegmentedColormap.from_list("vd", ["#00000000", CARD_BORDER, "#00000000"])
        vdiv.imshow(vgrad, aspect="auto", cmap=cmap_v, extent=[0, 1, 0, 1])
        vdiv.axis("off")

        # ── Right panel header ────────────────────────────────────────
        fig.text(0.53, 0.71, "\u2502", fontsize=14, color=accent,
                 ha="left", va="top", fontweight="bold")
        fig.text(0.545, 0.71, "SEASON TREND", fontsize=11,
                 color=CARD_TEXT_MUTED, ha="left", va="top",
                 fontweight="bold", style="italic")

        # ── Right panel: game log trend (top) + radar (bottom) ────────
        has_gamelog = game_log and len(game_log) >= 2
        has_radar = league_avgs and len(league_avgs) >= 3

        if has_gamelog and has_radar:
            # Split right panel: trend top, radar bottom
            trend_ax = fig.add_axes([0.54, 0.40, 0.42, 0.28])
            radar_ax = fig.add_axes([0.60, 0.05, 0.32, 0.32], polar=True)
        elif has_gamelog:
            trend_ax = fig.add_axes([0.54, 0.10, 0.42, 0.55])
            radar_ax = None
        elif has_radar:
            trend_ax = None
            radar_ax = fig.add_axes([0.58, 0.10, 0.38, 0.50], polar=True)
        else:
            trend_ax = None
            radar_ax = None

        # ── Game log trend chart ──────────────────────────────────────
        if trend_ax and has_gamelog:
            gl = game_log
            dates = list(range(len(gl)))
            eras = [g.get("era", np.nan) for g in gl]
            ks = []
            for g in gl:
                ip = g.get("inningsPitched", 0) or 0
                k = g.get("strikeOuts", 0) or 0
                ks.append((k * 9 / ip) if ip > 0 else np.nan)

            trend_ax.set_facecolor(CARD_SURFACE)
            for spine in trend_ax.spines.values():
                spine.set_color(CARD_BORDER)
            trend_ax.tick_params(colors=CARD_TEXT_MUTED, labelsize=7)
            trend_ax.grid(True, color=CARD_BORDER, alpha=0.3)

            # ERA line
            valid_eras = [(d, e) for d, e in zip(dates, eras) if not np.isnan(e)]
            if valid_eras:
                d_e, v_e = zip(*valid_eras)
                trend_ax.plot(d_e, v_e, color="#ff6b6b", linewidth=2,
                              marker="o", markersize=4, label="ERA", zorder=3)

            trend_ax.set_ylabel("ERA", color="#ff6b6b", fontsize=8)

            # K/9 on secondary y-axis
            ax2 = trend_ax.twinx()
            ax2.set_facecolor("none")
            ax2.tick_params(colors=CARD_TEXT_MUTED, labelsize=7)
            ax2.spines["right"].set_color(CARD_BORDER)
            ax2.spines["left"].set_color(CARD_BORDER)
            ax2.spines["top"].set_color(CARD_BORDER)
            ax2.spines["bottom"].set_color(CARD_BORDER)

            valid_ks = [(d, k) for d, k in zip(dates, ks) if not np.isnan(k)]
            if valid_ks:
                d_k, v_k = zip(*valid_ks)
                ax2.plot(d_k, v_k, color="#4ecdc4", linewidth=2,
                         marker="s", markersize=4, label="K/9", zorder=3)

            ax2.set_ylabel("K/9", color="#4ecdc4", fontsize=8)

            trend_ax.set_xlabel("Game #", color=CARD_TEXT_MUTED, fontsize=8)
            trend_ax.set_title("ERA & K/9 by Game", color=CARD_TEXT,
                               fontsize=10, fontweight="bold", pad=6)

            # Legend
            lines1, labels1 = trend_ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            trend_ax.legend(lines1 + lines2, labels1 + labels2,
                            loc="upper right", fontsize=7,
                            facecolor=CARD_SURFACE, edgecolor=CARD_BORDER,
                            labelcolor=CARD_TEXT_MUTED)

        # ── Radar chart ───────────────────────────────────────────────
        if radar_ax and has_radar:
            radar_stats = []
            for col, label, higher_good in _RADAR_STATS:
                if col in player.index and col in league_avgs:
                    try:
                        val = float(player[col])
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(val):
                        continue
                    avg = league_avgs[col]
                    # Normalize: ratio of player to league avg
                    # For "lower is better" stats, invert
                    if higher_good:
                        score = val / avg if avg > 0 else 1.0
                    else:
                        score = avg / val if val > 0 else 1.0
                    # Clamp to 0.2 - 2.0
                    score = max(0.2, min(2.0, score))
                    radar_stats.append((label, score))

            if len(radar_stats) >= 3:
                labels_r = [s[0] for s in radar_stats]
                scores = [s[1] for s in radar_stats]

                angles = np.linspace(0, 2 * np.pi, len(labels_r),
                                     endpoint=False).tolist()
                scores_plot = scores + scores[:1]
                angles_plot = angles + angles[:1]
                avg_line = [1.0] * (len(labels_r) + 1)

                radar_ax.set_facecolor(CARD_SURFACE)
                radar_ax.plot(angles_plot, avg_line, color=CARD_TEXT_MUTED,
                              linewidth=1, linestyle="--", alpha=0.5,
                              label="Lg Avg")
                radar_ax.fill(angles_plot, avg_line, color=CARD_TEXT_MUTED,
                              alpha=0.05)
                radar_ax.plot(angles_plot, scores_plot, color=accent,
                              linewidth=2, label=name.split()[-1])
                radar_ax.fill(angles_plot, scores_plot, color=accent,
                              alpha=0.15)

                radar_ax.set_xticks(angles)
                radar_ax.set_xticklabels(labels_r, fontsize=8, color=CARD_TEXT,
                                         fontweight="bold")
                radar_ax.set_yticklabels([])
                radar_ax.set_ylim(0, 2.0)
                radar_ax.spines["polar"].set_color(CARD_BORDER)
                radar_ax.grid(color=CARD_BORDER, alpha=0.3)
                radar_ax.set_title("vs League Avg", color=CARD_TEXT,
                                   fontsize=10, fontweight="bold", pad=12)
                radar_ax.legend(loc="lower right", fontsize=7,
                                facecolor=CARD_SURFACE, edgecolor=CARD_BORDER,
                                labelcolor=CARD_TEXT_MUTED,
                                bbox_to_anchor=(1.3, -0.1))

        # ── Fallback: no data for right panel ─────────────────────────
        if not has_gamelog and not has_radar:
            no_ax = fig.add_axes([0.52, 0.07, 0.46, 0.61])
            no_ax.set_facecolor("none")
            no_ax.axis("off")
            # Show key stats as large text
            stat_lines = []
            for col, lbl in [("era", "ERA"), ("whip", "WHIP"),
                              ("inningsPitched", "IP"),
                              ("strikeOuts", "K"), ("baseOnBalls", "BB")]:
                if col in player.index:
                    try:
                        v = float(player[col])
                        if col == "era":
                            stat_lines.append(f"{lbl}: {v:.2f}")
                        elif col == "whip":
                            stat_lines.append(f"{lbl}: {v:.2f}")
                        else:
                            stat_lines.append(f"{lbl}: {v:.0f}")
                    except (TypeError, ValueError):
                        pass
            for i, line in enumerate(stat_lines):
                no_ax.text(0.5, 0.85 - i * 0.18, line,
                           fontsize=18, fontweight="bold",
                           color=CARD_TEXT, ha="center", va="center")

        # ── Footer ────────────────────────────────────────────────────
        foot_ax = fig.add_axes([0.03, 0.0, 0.94, 0.003])
        foot_ax.set_xlim(0, 1)
        foot_ax.set_ylim(0, 1)
        foot_ax.add_patch(Rectangle((0, 0), 1, 1, color=CARD_BORDER))
        foot_ax.axis("off")

        fig.text(0.04, 0.025, "@BachTalk1", fontsize=10, color=accent,
                 ha="left", va="center", fontweight="bold")
        fig.text(0.5, 0.025, "Data: MLB Stats API", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="center", va="center")
        fig.text(0.96, 0.025, f"{MLB_SEASON} {level_display}", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="right", va="center")

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"trad_pitcher_card_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=100,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        log.info("Saved traditional pitcher card: %s", out)
        return out

    except Exception:
        log.warning("plot_traditional_pitcher_card failed for %s", name,
                    exc_info=True)
        return None


def plot_traditional_pitching_summary(
    name: str,
    season_df: "pd.DataFrame",
    player_id: int | None = None,
    team: str | None = None,
    level: str = "AA",
    game_log: list[dict] | None = None,
    monthly_splits: list[dict] | None = None,
    league_avgs: dict | None = None,
) -> Path | None:
    """Render a traditional-stats pitching summary dashboard (white theme).

    Row 1: Header (headshot + bio)
    Row 2: Season stats table
    Row 3: Game log trend chart + radar chart (replaces movement plot)
    Row 4: Monthly splits table (replaces pitch-type table)
    Row 5: Footer
    """
    try:
        import pandas as pd
        import matplotlib.gridspec as gridspec

        # ── Locate pitcher ────────────────────────────────────────────
        name_col = None
        for c in ("pitcher_name", "player_name", "name"):
            if c in season_df.columns:
                name_col = c
                break
        if not name_col:
            return None

        matches = season_df[season_df[name_col] == name]
        if matches.empty and player_id is not None:
            matches = season_df[season_df["player_id"] == player_id]
        if matches.empty:
            return None
        player = matches.iloc[0]

        # ── Team info ─────────────────────────────────────────────────
        if not team:
            for tc in ("team", "team_abbreviation"):
                if tc in player.index and player[tc]:
                    team = str(player[tc]).upper()
                    break
        accent = TEAM_COLORS.get(team or "", "#3a86ff")

        from .milb_statcast import LEVEL_NAMES
        level_display = LEVEL_NAMES.get(level, level)

        # Pitcher hand
        p_throws = "R"
        for hc in ("p_throws", "throws"):
            if hc in player.index and player[hc]:
                p_throws = str(player[hc])[0].upper()
                break

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

        # ── Row 1: Header ─────────────────────────────────────────────
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
        ax_bio.text(0.5, 0.10, f"{MLB_SEASON} {level_display} Season", va="top",
                    ha="center", fontsize=26, fontstyle="italic",
                    color="#666666")

        ax_logo.axis("off")

        # ── Row 2: Season Stats Table ─────────────────────────────────
        ax_season = fig.add_subplot(gs[2, 1:7])
        ax_season.axis("off")

        season_headers = []
        season_values = []
        for col, header, fmt in _TRAD_SUMMARY_STATS:
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

        # ── Row 3: Game Log Trend + Radar Chart ───────────────────────
        ax_trend = fig.add_subplot(gs[3, 1:4])
        ax_radar_host = fig.add_subplot(gs[3, 4:7])

        # Game log trend
        has_gamelog = game_log and len(game_log) >= 2
        if has_gamelog:
            gl = game_log
            game_nums = list(range(1, len(gl) + 1))
            eras = [g.get("era", np.nan) for g in gl]
            ks_per_9 = []
            for g in gl:
                ip = g.get("inningsPitched", 0) or 0
                k = g.get("strikeOuts", 0) or 0
                ks_per_9.append((k * 9 / ip) if ip > 0 else np.nan)

            # ERA line
            valid_eras = [(d, e) for d, e in zip(game_nums, eras)
                          if not np.isnan(e)]
            if valid_eras:
                d_e, v_e = zip(*valid_eras)
                ax_trend.plot(d_e, v_e, color="#d62828", linewidth=2.5,
                              marker="o", markersize=5, label="ERA",
                              zorder=3)
            ax_trend.set_ylabel("ERA", fontsize=14, color="#d62828")

            # K/9 on twin axis
            ax_k9 = ax_trend.twinx()
            valid_ks = [(d, k) for d, k in zip(game_nums, ks_per_9)
                        if not np.isnan(k)]
            if valid_ks:
                d_k, v_k = zip(*valid_ks)
                ax_k9.plot(d_k, v_k, color="#3a86ff", linewidth=2.5,
                           marker="s", markersize=5, label="K/9",
                           zorder=3)
            ax_k9.set_ylabel("K/9", fontsize=14, color="#3a86ff")
            ax_k9.tick_params(labelsize=10)

            ax_trend.set_xlabel("Game #", fontsize=14)
            ax_trend.set_title("ERA & K/9 by Game", fontsize=20,
                               fontweight="bold")
            ax_trend.grid(True, alpha=0.3)
            ax_trend.tick_params(labelsize=10)

            # Legend
            lines1, labels1 = ax_trend.get_legend_handles_labels()
            lines2, labels2 = ax_k9.get_legend_handles_labels()
            ax_trend.legend(lines1 + lines2, labels1 + labels2,
                            loc="upper right", fontsize=12)
        else:
            ax_trend.axis("off")
            ax_trend.text(0.5, 0.5, "Not enough games for trend",
                          ha="center", va="center", fontsize=16)

        # Radar chart
        has_radar = league_avgs and len(league_avgs) >= 3
        ax_radar_host.axis("off")  # hide the grid subplot

        if has_radar:
            radar_stats = []
            for col, label, higher_good in _RADAR_STATS:
                if col in player.index and col in league_avgs:
                    try:
                        val = float(player[col])
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(val):
                        continue
                    avg = league_avgs[col]
                    if higher_good:
                        score = val / avg if avg > 0 else 1.0
                    else:
                        score = avg / val if val > 0 else 1.0
                    score = max(0.2, min(2.0, score))
                    radar_stats.append((label, score))

            if len(radar_stats) >= 3:
                # Create polar axes in the same region
                pos = ax_radar_host.get_position()
                radar_ax = fig.add_axes(
                    [pos.x0 + 0.02, pos.y0, pos.width - 0.04, pos.height],
                    polar=True,
                )

                labels_r = [s[0] for s in radar_stats]
                scores = [s[1] for s in radar_stats]
                angles = np.linspace(0, 2 * np.pi, len(labels_r),
                                     endpoint=False).tolist()
                scores_plot = scores + scores[:1]
                angles_plot = angles + angles[:1]
                avg_line = [1.0] * (len(labels_r) + 1)

                radar_ax.plot(angles_plot, avg_line, color="#888888",
                              linewidth=1.5, linestyle="--", alpha=0.6,
                              label="Lg Avg")
                radar_ax.fill(angles_plot, avg_line, color="#888888",
                              alpha=0.05)
                radar_ax.plot(angles_plot, scores_plot, color=accent,
                              linewidth=2.5, label=name.split()[-1])
                radar_ax.fill(angles_plot, scores_plot, color=accent,
                              alpha=0.15)

                radar_ax.set_xticks(angles)
                radar_ax.set_xticklabels(labels_r, fontsize=14,
                                         fontweight="bold")
                radar_ax.set_yticklabels([])
                radar_ax.set_ylim(0, 2.0)
                radar_ax.grid(True, alpha=0.3)
                radar_ax.set_title("vs League Average", fontsize=20,
                                   fontweight="bold", pad=15)
                radar_ax.legend(loc="lower right", fontsize=12,
                                bbox_to_anchor=(1.3, -0.1))

        # ── Row 4: Monthly Splits Table ───────────────────────────────
        ax_monthly = fig.add_subplot(gs[4, 1:7])
        ax_monthly.axis("off")

        if monthly_splits and len(monthly_splits) >= 1:
            month_headers = [
                "$\\bf{Month}$", "$\\bf{IP}$", "$\\bf{ERA}$",
                "$\\bf{WHIP}$", "$\\bf{K/9}$", "$\\bf{BB/9}$",
                "$\\bf{BAVG}$", "$\\bf{GO/AO}$",
            ]
            month_cols = [
                ("inningsPitched", ".1f"),
                ("era", ".2f"),
                ("whip", ".2f"),
                ("strikeoutsPer9Inn", ".2f"),
                ("walksPer9Inn", ".2f"),
                ("avg", ".3f"),
                ("groundOutsToAirouts", ".2f"),
            ]

            cell_text = []
            for m in monthly_splits:
                row_data = [m.get("month", "?")]
                for col, fmt in month_cols:
                    val = m.get(col, np.nan)
                    try:
                        val_f = float(val)
                        if pd.isna(val_f):
                            row_data.append("\u2014")
                        else:
                            row_data.append(format(val_f, fmt))
                    except (TypeError, ValueError):
                        row_data.append("\u2014")
                cell_text.append(row_data)

            if cell_text:
                tbl = ax_monthly.table(
                    cellText=cell_text,
                    colLabels=month_headers,
                    cellLoc="center",
                    bbox=[0, -0.05, 1, 1],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(16)
                tbl.scale(1, 2.2)
                for key, cell in tbl.get_celld().items():
                    cell.set_edgecolor("#cccccc")
                    if key[0] == 0:
                        cell.set_facecolor("#f0f0f0")
                        cell.set_text_props(fontweight="bold")
        else:
            ax_monthly.text(0.5, 0.5, "Monthly splits not available",
                            ha="center", va="center", fontsize=18,
                            color="#888888")

        # ── Footer ────────────────────────────────────────────────────
        ax_footer = fig.add_subplot(gs[-1, 1:7])
        ax_footer.axis("off")
        ax_footer.text(0, 1, "By: @BachTalk1", ha="left", va="top",
                       fontsize=22, fontweight="bold")
        ax_footer.text(0.5, 1,
                       "Traditional Stats \u2014 No Statcast at this level",
                       ha="center", va="top", fontsize=14, color="#666666")
        ax_footer.text(1, 1, "Data: MLB Stats API\nImages: MLB",
                       ha="right", va="top", fontsize=22)

        # ── Save ──────────────────────────────────────────────────────
        safe = name.replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"trad_pitching_summary_{safe}.png"
        _draw_watermark(fig, alpha=0.08, dark_bg=False)
        fig.savefig(out, facecolor="white", dpi=150,
                    bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        log.info("Saved traditional pitching summary: %s", out)
        return out

    except Exception:
        log.warning("plot_traditional_pitching_summary failed for %s", name,
                    exc_info=True)
        return None


# ── Biomechanics Educational Charts ──────────────────────────────────

# Accent colours for biomechanics charts
_BIO_PRIMARY = "#3a86ff"
_BIO_SECONDARY = "#ff6b6b"
_BIO_HIGHLIGHT = "#ffbe0b"


def plot_biomechanics(
    topic: dict,
    df: "pd.DataFrame",
    stats: dict,
) -> Path | None:
    """Render a biomechanics educational chart (1200x675, dark theme).

    Supports scatter plots (x vs y with trend line) and
    distribution charts (histogram with percentile markers).
    """
    try:
        import pandas as pd
        from scipy import stats as sp_stats

        chart_type = topic.get("chart_type", "scatter")
        x_col = topic["x_col"]
        y_col = topic.get("y_col")

        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
        fig.set_facecolor(CARD_BG)

        # Background noise
        noise = np.random.default_rng(42).uniform(0.04, 0.07, (68, 120))
        bg_ax = fig.add_axes([0, 0, 1, 1])
        bg_ax.imshow(noise, aspect="auto", cmap="gray", alpha=0.03,
                     extent=[0, 1, 0, 1])
        bg_ax.axis("off")

        # Accent stripe
        stripe = fig.add_axes([0, 0.97, 1, 0.03])
        stripe.set_xlim(0, 1)
        stripe.set_ylim(0, 1)
        stripe.add_patch(Rectangle((0, 0), 1, 1, color=_BIO_PRIMARY))
        stripe.axis("off")

        # Title
        shadow = [patheffects.withStroke(linewidth=4, foreground=CARD_BG)]
        fig.text(0.5, 0.94, topic["title"],
                 fontsize=22, fontweight="bold", color=CARD_TEXT,
                 ha="center", va="top", path_effects=shadow)

        # Subtitle
        n_pitchers = stats.get("n_pitchers", "?")
        n_pitches = stats.get("n_pitches", "?")
        fig.text(0.5, 0.89,
                 f"Driveline Open Biomechanics  |  {n_pitches} pitches, "
                 f"{n_pitchers} pitchers (mostly college)",
                 fontsize=10, color=CARD_TEXT_MUTED,
                 ha="center", va="top")

        # Main chart area
        ax = fig.add_axes([0.10, 0.13, 0.82, 0.70])
        ax.set_facecolor(CARD_SURFACE)
        for spine in ax.spines.values():
            spine.set_color(CARD_BORDER)
        ax.tick_params(colors=CARD_TEXT_MUTED, labelsize=9)
        ax.grid(True, color=CARD_BORDER, alpha=0.3, linewidth=0.5)

        if chart_type == "scatter" and y_col:
            valid = df[[x_col, y_col]].dropna()
            x = valid[x_col].values
            y = valid[y_col].values

            ax.scatter(x, y, c=_BIO_PRIMARY, alpha=0.5, s=40,
                       edgecolors="white", linewidths=0.3, zorder=3)

            # Trend line
            if len(x) > 10:
                slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=_BIO_SECONDARY, linewidth=2.5,
                        linestyle="--", alpha=0.8, zorder=4)

                # R-squared annotation
                r_sq = r_val ** 2
                corr = stats.get("correlation", r_val)
                ax.text(0.97, 0.95,
                        f"r = {corr:.2f}  |  R\u00b2 = {r_sq:.2f}",
                        transform=ax.transAxes, fontsize=11,
                        fontweight="bold", color=_BIO_HIGHLIGHT,
                        ha="right", va="top",
                        bbox=dict(facecolor=CARD_BG, edgecolor=CARD_BORDER,
                                  boxstyle="round,pad=0.4", alpha=0.9))

            ax.set_xlabel(topic["x_label"], fontsize=12, color=CARD_TEXT,
                          fontweight="bold")
            ax.set_ylabel(topic["y_label"], fontsize=12, color=CARD_TEXT,
                          fontweight="bold")

        elif chart_type == "distribution":
            vals = df[x_col].dropna().values

            ax.hist(vals, bins=25, color=_BIO_PRIMARY, alpha=0.7,
                    edgecolor=CARD_BORDER, linewidth=0.5, zorder=3)

            # Percentile lines
            p10 = stats.get("x_p10", np.percentile(vals, 10))
            p50 = stats.get("x_median", np.median(vals))
            p90 = stats.get("x_p90", np.percentile(vals, 90))

            for pval, plabel, color in [
                (p10, "10th %ile", CARD_TEXT_MUTED),
                (p50, "Median", _BIO_HIGHLIGHT),
                (p90, "90th %ile", _BIO_SECONDARY),
            ]:
                ax.axvline(pval, color=color, linewidth=2, linestyle="--",
                           alpha=0.8, zorder=4)
                ax.text(pval, ax.get_ylim()[1] * 0.92,
                        f" {plabel}\n {pval:.1f}",
                        fontsize=9, fontweight="bold", color=color,
                        va="top", ha="left")

            ax.set_xlabel(topic["x_label"], fontsize=12, color=CARD_TEXT,
                          fontweight="bold")
            ax.set_ylabel("Count", fontsize=12, color=CARD_TEXT,
                          fontweight="bold")

        # Footer
        foot_ax = fig.add_axes([0.03, 0.0, 0.94, 0.003])
        foot_ax.set_xlim(0, 1)
        foot_ax.set_ylim(0, 1)
        foot_ax.add_patch(Rectangle((0, 0), 1, 1, color=CARD_BORDER))
        foot_ax.axis("off")

        fig.text(0.04, 0.025, "@BachTalk1", fontsize=10,
                 color=_BIO_PRIMARY, ha="left", va="center",
                 fontweight="bold")
        fig.text(0.5, 0.025, "Data: Driveline Open Biomechanics Project",
                 fontsize=9, color=CARD_TEXT_MUTED, ha="center", va="center")
        fig.text(0.96, 0.025, "Biomechanics 101", fontsize=9,
                 color=CARD_TEXT_MUTED, ha="right", va="center")

        # Save
        safe = topic["id"].replace(" ", "_").lower()
        out = SCREENSHOTS_DIR / f"biomech_{safe}.png"
        _draw_watermark(fig)
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=100,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        log.info("Saved biomechanics chart: %s", out)
        return out

    except Exception:
        log.warning("plot_biomechanics failed for topic %s",
                    topic.get("id", "?"), exc_info=True)
        return None


# ── Reds Game Summary Card ──────────────────────────────────────────────

# TJStats pitch colour palette (richer palette for game summary cards)
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

import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.colors as mcolors     # noqa: E402
import pandas as pd                     # noqa: E402


def _get_table_cell_color(value: float, league_mean: float,
                          cmap, spread: float = 0.3) -> str:
    """Return hex colour for a table cell based on value vs league mean."""
    lo = league_mean * (1 - spread)
    hi = league_mean * (1 + spread)
    if lo >= hi:
        return "#ffffff"
    try:
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        return mcolors.to_hex(cmap(norm(value)))
    except (ValueError, ZeroDivisionError):
        return "#ffffff"


def _pctile_color(p: float) -> str:
    """Return a colour for a percentile value."""
    if p >= 90:
        return "#c0392b"   # elite red
    if p >= 70:
        return "#e67e22"   # above avg orange
    if p >= 40:
        return "#f1c40f"   # average yellow
    if p >= 20:
        return "#3498db"   # below avg blue
    return "#2c3e50"       # poor dark


def plot_reds_matchup_header(
    opponent_abbrev: str,
    game_date: str,
    starter_name: str,
    num_pitchers: int,
    score_line: str = "",
    is_home: bool = True,
) -> Path | None:
    """Generate MLB-style split-screen matchup header with team colors + logos."""
    try:
        from io import BytesIO
        from PIL import Image

        _LOGO_SLUGS = {
            "ARI": "ari", "ATL": "atl", "BAL": "bal", "BOS": "bos",
            "CHC": "chc", "CWS": "chw", "CIN": "cin", "CLE": "cle",
            "COL": "col", "DET": "det", "HOU": "hou", "KC": "kc",
            "LAA": "laa", "LAD": "lad", "MIA": "mia", "MIL": "mil",
            "MIN": "min", "NYM": "nym", "NYY": "nyy", "OAK": "oak",
            "PHI": "phi", "PIT": "pit", "SD": "sd", "SF": "sf",
            "SEA": "sea", "STL": "stl", "TB": "tb", "TEX": "tex",
            "TOR": "tor", "WSH": "wsh",
        }
        _TEAM_BG_COLORS = {
            "ARI": "#A71930", "ATL": "#13274F", "BAL": "#DF4601", "BOS": "#0C2340",
            "CHC": "#0E3386", "CWS": "#27251F", "CIN": "#C6011F", "CLE": "#00385D",
            "COL": "#333366", "DET": "#0C2340", "HOU": "#002D62", "KC": "#004687",
            "LAA": "#862633", "LAD": "#005A9C", "MIA": "#00A3E0", "MIL": "#12284B",
            "MIN": "#002B5C", "NYM": "#002D72", "NYY": "#0C2340", "OAK": "#003831",
            "PHI": "#E81828", "PIT": "#27251F", "SD": "#2F241D", "SF": "#FD5A1E",
            "SEA": "#0C2C56", "STL": "#C41E3A", "TB": "#092C5C", "TEX": "#003278",
            "TOR": "#134A8E", "WSH": "#AB0003",
        }
        _TEAM_NAMES = {
            "ARI": "D-backs", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
            "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds", "CLE": "Guardians",
            "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
            "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
            "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "OAK": "Athletics",
            "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SF": "Giants",
            "SEA": "Mariners", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
            "TOR": "Blue Jays", "WSH": "Nationals",
        }

        def _get_logo(abbrev):
            slug = _LOGO_SLUGS.get(abbrev, abbrev.lower())
            url = (f"https://a.espncdn.com/combiner/i?img="
                   f"/i/teamlogos/mlb/500/scoreboard/{slug}.png&h=500&w=500")
            resp = _requests.get(url, timeout=10, allow_redirects=True)
            return Image.open(BytesIO(resp.content)).convert("RGBA")

        # Away team on left, home team on right (MLB style)
        if is_home:
            left_abbrev, right_abbrev = opponent_abbrev, "CIN"
        else:
            left_abbrev, right_abbrev = "CIN", opponent_abbrev

        left_logo = _get_logo(left_abbrev)
        right_logo = _get_logo(right_abbrev)
        left_bg = _TEAM_BG_COLORS.get(left_abbrev, "#333333")
        right_bg = _TEAM_BG_COLORS.get(right_abbrev, "#C6011F")
        left_name = _TEAM_NAMES.get(left_abbrev, left_abbrev)
        right_name = _TEAM_NAMES.get(right_abbrev, right_abbrev)

        fig = plt.figure(figsize=(12, 6.3), dpi=150)
        fig.set_facecolor("#000000")

        # Left half background
        ax_left_bg = fig.add_axes([0, 0, 0.5, 1])
        ax_left_bg.set_xlim(0, 1); ax_left_bg.set_ylim(0, 1)
        ax_left_bg.add_patch(Rectangle((0, 0), 1, 1, color=left_bg))
        ax_left_bg.axis("off")

        # Right half background
        ax_right_bg = fig.add_axes([0.5, 0, 0.5, 1])
        ax_right_bg.set_xlim(0, 1); ax_right_bg.set_ylim(0, 1)
        ax_right_bg.add_patch(Rectangle((0, 0), 1, 1, color=right_bg))
        ax_right_bg.axis("off")

        # Logos
        ax_ll = fig.add_axes([0.07, 0.25, 0.32, 0.55])
        ax_ll.imshow(left_logo); ax_ll.axis("off")
        ax_rl = fig.add_axes([0.58, 0.25, 0.32, 0.55])
        ax_rl.imshow(right_logo); ax_rl.axis("off")

        # Center divider line
        import matplotlib.lines as mlines
        divider = mlines.Line2D([0.5, 0.5], [0.08, 0.92],
                                transform=fig.transFigure,
                                color="#ffffff", linewidth=2, alpha=0.15)
        fig.add_artist(divider)

        # Team names
        fig.text(0.25, 0.18, left_abbrev, fontsize=12, color="#ffffff",
                 ha="center", va="center", alpha=0.6)
        fig.text(0.25, 0.12, left_name.upper(), fontsize=18,
                 fontweight="bold", color="#ffffff", ha="center", va="center")

        fig.text(0.75, 0.18, right_abbrev, fontsize=12, color="#ffffff",
                 ha="center", va="center", alpha=0.6)
        fig.text(0.75, 0.12, right_name.upper(), fontsize=18,
                 fontweight="bold", color="#ffffff", ha="center", va="center")

        # Score line at bottom
        if score_line:
            fig.text(0.5, 0.04, score_line, fontsize=11, color="#ffffff",
                     ha="center", va="center", alpha=0.7)

        # Game date bottom right
        fig.text(0.96, 0.04, game_date, fontsize=9, color="#ffffff",
                 ha="right", va="center", alpha=0.5)

        # Starter info bottom left
        fig.text(0.04, 0.04, f"Starter: {starter_name}",
                 fontsize=9, color="#ffffff", ha="left", va="center", alpha=0.5)

        out = SCREENSHOTS_DIR / "reds_matchup_header.png"
        fig.savefig(out, facecolor="#000000", dpi=150,
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        log.info("Saved matchup header: %s", out)
        return out

    except Exception:
        log.warning("plot_reds_matchup_header failed", exc_info=True)
        return None


def plot_reds_game_summary(
    name: str,
    game_stats: pd.Series,
    game_pitches_df: pd.DataFrame,
    all_pitches_df: pd.DataFrame | None = None,
    team: str = "CIN",
    player_id: int | None = None,
    pbp_df: pd.DataFrame | None = None,
    season_df: pd.DataFrame | None = None,
    game_date: str = "",
    opponent: str = "",
    season: int | None = None,
) -> Path | None:
    """Render a game-day pitching summary card for a Reds pitcher.

    Layout (8-row GridSpec):
      Row 0: top border
      Row 1: header — headshot, bio text, team logo
      Row 2: game stats table (IP, H, R, ER, BB, K, NP, etc.)
      Row 3: movement chart (left) + percentile bars (right)
      Row 4: pitch locations scatter plot
      Row 5: colour-coded pitch-type stats table
      Row 6: footer
      Row 7: bottom border

    Returns path to saved PNG, or None on failure.
    """
    if season is None:
        season = MLB_SEASON

    try:
        accent = TEAM_COLORS.get(team, "#3a86ff")

        # Pitcher hand
        p_throws = "R"
        for hc in ("p_throws", "throws", "hand"):
            if hc in game_stats.index and game_stats[hc]:
                p_throws = str(game_stats[hc])[0].upper()
                break

        # ── Pitch-type aggregation ───────────────────────────────────
        pitch_rows = pd.DataFrame()
        if game_pitches_df is not None and not game_pitches_df.empty:
            prows = game_pitches_df.copy()
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
        # 7 rows: border, header, stats, movement+locations+pctile, pitch table, footer, border
        fig = plt.figure(figsize=(20, 22), dpi=150)
        fig.set_facecolor("white")

        gs = gridspec.GridSpec(
            7, 9,
            height_ratios=[2, 18, 8, 35, 30, 5, 2],
            width_ratios=[1, 6, 6, 6, 6, 6, 6, 6, 1],
            hspace=0.3, wspace=0.4,
        )

        # Border axes (hidden)
        for pos in [gs[0, 1:8], gs[6, 1:8], gs[:, 0], gs[:, -1]]:
            bax = fig.add_subplot(pos)
            bax.axis("off")

        # ── Row 1: Header — headshot / bio / logo ────────────────────
        ax_headshot = fig.add_subplot(gs[1, 1:3])
        ax_bio = fig.add_subplot(gs[1, 3:7])
        ax_logo = fig.add_subplot(gs[1, 7:8])

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
                img = _PILImage.open(BytesIO(resp.content))
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
        subtitle = "Game Summary"
        if game_date and opponent:
            subtitle = f"Game Summary \u2014 {game_date} vs {opponent}"
        ax_bio.text(0.5, 0.38, subtitle,
                    va="top", ha="center", fontsize=26, fontweight="bold")
        ax_bio.text(0.5, 0.12, f"{season} MLB Season", va="top",
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
                resp = _requests.get(logo_url, timeout=10)
                resp.raise_for_status()
                img = _PILImage.open(BytesIO(resp.content))
                ax_logo.set_xlim(0, 1.3)
                ax_logo.set_ylim(0, 1)
                ax_logo.imshow(img, extent=[0.3, 1.3, 0, 1], origin="upper")
            except Exception:
                log.debug("Logo failed for %s", team)

        # ── Row 2: Game Stats Table ──────────────────────────────────
        ax_season = fig.add_subplot(gs[2, 1:8])
        ax_season.axis("off")

        game_headers = []
        game_values = []
        for col, header, fmt in _GAME_STATS:
            if col in game_stats.index:
                try:
                    val = float(game_stats[col])
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

        # ── Row 3: Movement (LEFT) + Locations (CENTER) + Percentiles (RIGHT)
        ax_break = fig.add_subplot(gs[3, 1:3])

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
        ax_pctile = fig.add_subplot(gs[3, 6:8])

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
            if col not in compare_df.columns or col not in game_stats.index:
                continue
            vals = pd.to_numeric(compare_df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            try:
                raw_f = float(game_stats[col])
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

        # ── Row 3 (center): Pitch location scatter plot ──────────────
        ax_loc = fig.add_subplot(gs[3, 3:6])

        if (pbp_df is not None and not pbp_df.empty
                and "plate_x" in pbp_df.columns and "plate_z" in pbp_df.columns):
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
        ax_table = fig.add_subplot(gs[4, 1:8])
        ax_table.axis("off")

        if not pitch_rows.empty:
            league_df = all_pitches_df if all_pitches_df is not None else game_pitches_df

            cell_text = []
            cell_colors = []
            row_label_colors = []

            for _, row in pitch_rows.iterrows():
                pt = str(row["pitch_type"])
                pname_display = _TJ_NAME.get(pt, pt)
                usage = float(row.get("percentage_thrown", 0) or 0)
                row_data = [pname_display, f"{usage:.1%}"]
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
        ax_footer = fig.add_subplot(gs[5, 1:8])
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
        _draw_watermark(fig, alpha=0.08, dark_bg=False)
        fig.savefig(out, facecolor="white", dpi=150,
                    bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        log.info("Saved game summary card: %s", out)
        return out

    except Exception:
        log.warning("plot_reds_game_summary failed for %s", name,
                    exc_info=True)
        return None
