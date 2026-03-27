"""Content generator: Weekly Top 10 Swing+ leaderboard (Mondays).

Pure mechanics model — trained on bat tracking features only (no delta run exp).
Uses Baseball Savant bat tracking data + FanGraphs xwOBA for validation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle
from PIL import Image as PILImage

from .base import ContentGenerator, PostContent
from ..config import SCREENSHOTS_DIR, MLB_SEASON

log = logging.getLogger(__name__)

# Theme
CARD_BG = "#0d1117"
CARD_SURFACE = "#161b22"
CARD_BORDER = "#30363d"
CARD_TEXT = "#f0f6fc"
CARD_TEXT_MUTED = "#8b949e"
BIO_PRIMARY = "#3a86ff"
GOLD = "#ffbe0b"

WATERMARK_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "BachTalk.png"

FEATURES = [
    "bat_speed", "squared_up_rate", "squared_up_speed_rate",
    "ideal_attack_angle_rate", "swing_length",
    "sweetspot_speed_high", "hit_into_play_rate", "swords",
]


def _compute_swing_plus():
    """Compute Swing+ for all qualified hitters using pure mechanics model."""
    from pybaseball import batting_stats
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import requests

    # Pull bat tracking from Savant
    url = "https://baseballsavant.mlb.com/leaderboard/bat-tracking?attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd=&gameType=&isHardHit=&minSwings=100&minGroupSwings=1&pitchHand=&pitchType=&playerPool=All&season={}&seasonStart=&seasonEnd=&team=&type=batter&csv=true".format(MLB_SEASON)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200 or len(resp.text) < 100:
            log.warning("Bat tracking CSV fetch failed: %d", resp.status_code)
            return None
        import io
        bt = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        log.warning("Failed to fetch bat tracking data", exc_info=True)
        return None

    if bt.empty:
        return None

    # Map column names
    col_map = {
        "avg_sweetspot_speed_mph": "bat_speed",
        "squared_up_per_bat_contact": "squared_up_rate",
        "squared_up_with_speed_per_bat_contact": "squared_up_speed_rate",
        "rate_ideal_attack_angle": "ideal_attack_angle_rate",
        "swing_length": "swing_length",
        "avg_is_sweetspot_speed_high": "sweetspot_speed_high",
        "hit_into_play_per_swing": "hit_into_play_rate",
        "swords": "swords",
    }

    # Try to find correct column names
    for savant_col, feat in list(col_map.items()):
        if savant_col not in bt.columns:
            # Try qualified versions
            qual = savant_col.replace("_mph", "_mph_qualified").replace("swing_length", "swing_length_qualified")
            if qual in bt.columns:
                col_map[savant_col] = feat
                bt[savant_col] = bt[qual]

    # Check we have the needed columns
    missing = [c for c in col_map.keys() if c not in bt.columns]
    if missing:
        log.warning("Missing bat tracking columns: %s. Available: %s", missing, bt.columns.tolist()[:20])
        return None

    bt_clean = bt.rename(columns=col_map)

    # Get name column
    name_col = None
    for c in ["batter_name", "name", "player_name"]:
        if c in bt_clean.columns:
            name_col = c
            break
    if not name_col:
        return None

    bt_clean["name_fg"] = bt_clean[name_col].apply(
        lambda x: " ".join(str(x).split(", ")[::-1]).strip() if ", " in str(x) else str(x).strip()
    )

    # Pull xwOBA for training
    try:
        fg = batting_stats(MLB_SEASON, qual=50)
        fg["name_fg"] = fg["Name"].str.strip()
        merged = bt_clean.merge(fg[["name_fg", "xwOBA"]], on="name_fg", how="inner")
        merged = merged.dropna(subset=["xwOBA"] + FEATURES)
    except Exception:
        log.warning("FanGraphs merge failed", exc_info=True)
        return None

    if len(merged) < 50:
        log.warning("Only %d merged rows, not enough", len(merged))
        return None

    # Train ridge regression
    X = StandardScaler().fit_transform(merged[FEATURES].values)
    y = merged["xwOBA"].values
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=10).fit(X, y)
    pred = ridge.predict(X)
    pm, ps = pred.mean(), pred.std()
    merged["swing_plus"] = np.round(100 + ((pred - pm) / ps) * 15, 1)

    # Get team info
    team_col = None
    for c in ["team_name", "team", "team_abbreviation"]:
        if c in merged.columns:
            team_col = c
            break

    return merged, team_col


def _fetch_headshot(player_id):
    """Download and make circular headshot."""
    from PIL import Image, ImageDraw
    from io import BytesIO
    import requests
    url = f"https://securea.mlb.com/mlb/images/players/head_shot/{player_id}.jpg"
    try:
        resp = requests.get(url, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        size = min(img.size)
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size)).resize((100, 100), Image.LANCZOS)
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 100, 100), fill=255)
        img.putalpha(mask)
        return np.array(img)
    except Exception:
        return None


def _build_top10_image(df, team_col):
    """Build the top 10 Swing+ bar chart with headshots — dark theme."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    top10 = df.nlargest(10, "swing_plus").reset_index(drop=True)
    n = len(top10)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bg_color = "#141414"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bar_color = "#3a86ff"
    max_swing = top10["swing_plus"].max()
    x_max = max_swing + 6

    rank_x = 93.0
    hs_x = 94.5
    bar_left = 96.0

    y_positions = np.arange(n - 1, -1, -1)

    for i in range(n):
        row = top10.iloc[i]
        y = y_positions[i]
        sp_val = row["swing_plus"]
        name = row.get("name_fg", "?")
        team = str(row.get(team_col, "")) if team_col else ""
        xwoba = row.get("xwOBA", 0)
        bat_spd = row.get("bat_speed", 0)

        # Bar
        ax.barh(y, sp_val - bar_left, left=bar_left, height=0.7,
                color=bar_color, edgecolor="none", zorder=2)

        # Rank
        ax.text(rank_x, y, f"#{i+1}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#888888")

        # Headshot
        pid = row.get("player_id", None)
        if pid:
            try:
                hs = _fetch_headshot(int(pid))
                if hs is not None:
                    imagebox = OffsetImage(hs, zoom=0.28)
                    imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (hs_x, y), frameon=False,
                                        box_alignment=(0.5, 0.5), zorder=4)
                    ax.add_artist(ab)
            except Exception:
                pass

        # Name on bar
        ax.text(bar_left + 0.2, y + 0.16, name, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=5)

        # Subtitle: team · bat speed · xwOBA
        subtitle = f"{team}  ·  {bat_spd:.1f} mph  ·  .{str(xwoba)[2:5]} xwOBA"
        ax.text(bar_left + 0.2, y - 0.17, subtitle, ha="left", va="center",
                fontsize=7, color="#cc3333", zorder=5)

        # Swing+ value
        ax.text(sp_val + 0.4, y, f"{sp_val:.1f}", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color="white")

    # 100 = Avg line
    ax.axvline(x=100, color="#666666", linestyle="--", linewidth=0.7, zorder=1, alpha=0.5)
    ax.text(100, n - 0.55, "100 = Avg", ha="center", va="bottom",
            fontsize=7.5, color="#777777")

    ax.set_xlim(rank_x - 1.0, x_max)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Swing+", fontsize=10, color="#999999")
    ax.tick_params(axis="x", colors="#666666", labelsize=8.5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#333333")

    fig.suptitle(f"Top 10 Hitters by Swing+ ({MLB_SEASON})",
                 fontsize=19, fontweight="bold", color="white", y=0.97)
    ax.set_title(f"{MLB_SEASON} Season  ·  Pure Mechanics Model  ·  Min 100 Swings",
                 fontsize=8.5, color="#777777", pad=10)

    fig.text(0.03, 0.01, "Data: Baseball Savant Bat Tracking",
             fontsize=6.5, color="#555555")
    fig.text(0.97, 0.01, "@BachTalk1",
             fontsize=6.5, color="#555555", ha="right")

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
            ax_wm.imshow(arr, alpha=0.12)
            ax_wm.set_facecolor("none")
            ax_wm.axis("off")
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = SCREENSHOTS_DIR / "swing_plus_top10.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close()
    return out


class SwingPlusTop10Generator(ContentGenerator):
    name = "swing_plus_top10"

    async def generate(self) -> PostContent:
        result = _compute_swing_plus()
        if result is None:
            log.warning("Swing+ computation failed")
            return PostContent(text="")

        df, team_col = result
        image_path = _build_top10_image(df, team_col)
        if not image_path:
            return PostContent(text="")

        top = df.nlargest(3, "swing_plus")
        names = [r["name_fg"] for _, r in top.iterrows()]

        text = (
            f"Updated Swing+ Top 10 — {MLB_SEASON}\n\n"
            f"Pure mechanics model trained on bat tracking data. "
            f"{names[0]}, {names[1]}, and {names[2]} lead the way.\n\n"
            f"@TJStats #MLB #Statcast #BatTracking"
        )

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Swing+ Top 10 leaderboard for {MLB_SEASON}",
            tags=["swing_plus_top10"],
        )
