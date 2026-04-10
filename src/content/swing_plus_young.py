"""Content generator: Top 10 Swing+ for young hitters (25 and under).

Same pure mechanics model as swing_plus_top10 but filtered to players
aged 25 or younger. Uses MLB Stats API for birth dates.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage, ImageDraw

from .base import ContentGenerator, PostContent
from ..config import SCREENSHOTS_DIR, MLB_SEASON, DEFAULT_HASHTAGS
from ..video_clips import _download_mp4
from .swing_plus_top10 import _compute_swing_plus, _fetch_headshot, _get_savant_video

log = logging.getLogger(__name__)

WATERMARK_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "BachTalk.png"

MAX_AGE = 25


def _get_player_ages(player_ids: list[int]) -> dict[int, int]:
    """Fetch birth dates from MLB Stats API and return {player_id: age}."""
    ages = {}
    today = date.today()
    # Batch in groups of 100
    for i in range(0, len(player_ids), 100):
        batch = player_ids[i:i + 100]
        ids_str = ",".join(str(pid) for pid in batch)
        try:
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}",
                timeout=15,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid = person["id"]
                bd = person.get("birthDate")
                if bd:
                    born = date.fromisoformat(bd)
                    age = today.year - born.year - (
                        (today.month, today.day) < (born.month, born.day)
                    )
                    ages[pid] = age
        except Exception:
            log.warning("MLB API age lookup failed for batch starting %s", batch[0], exc_info=True)
    return ages


def _build_young_top10_image(df):
    """Build horizontal bar chart — same style as main Swing+ but for young hitters."""
    top10 = df.nlargest(10, "swing_plus").reset_index(drop=True)
    n = len(top10)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bg_color = "#141414"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bar_color = "#00c896"  # green tint to differentiate from main Swing+
    max_swing = top10["swing_plus"].max()
    x_max = max_swing + 6

    rank_x = 93.0; hs_x = 94.5; bar_left = 96.0
    y_positions = np.arange(n - 1, -1, -1)

    for i in range(n):
        row = top10.iloc[i]
        y = y_positions[i]
        sp_val = row["swing_plus"]
        name = row.get("name_fg", "?")
        xwoba = row.get("xwOBA", 0)
        bat_spd = row.get("bat_speed", 0)
        age = row.get("age", "")

        ax.barh(y, sp_val - bar_left, left=bar_left, height=0.7,
                color=bar_color, edgecolor="none", zorder=2)

        ax.text(rank_x, y, f"#{i+1}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#888888")

        pid = row.get("player_id")
        if pid and not np.isnan(pid):
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

        ax.text(bar_left + 0.2, y + 0.16, name, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=5)

        xw_str = f".{str(xwoba)[2:5]}" if xwoba and xwoba > 0 else ""
        subtitle = f"Age {age}  ·  {bat_spd:.1f} mph"
        if xw_str:
            subtitle += f"  ·  {xw_str} xwOBA"
        ax.text(bar_left + 0.2, y - 0.17, subtitle, ha="left", va="center",
                fontsize=7, color="#00e6a0", zorder=5)

        ax.text(sp_val + 0.4, y, f"{sp_val:.1f}", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color="white")

    ax.axvline(x=100, color="#666666", linestyle="--", linewidth=0.7, zorder=1, alpha=0.5)
    ax.text(100, n - 0.55, "100 = Avg", ha="center", va="bottom", fontsize=7.5, color="#777777")

    ax.set_xlim(rank_x - 1.0, x_max)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Swing+", fontsize=10, color="#999999")
    ax.tick_params(axis="x", colors="#666666", labelsize=8.5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#333333")

    fig.suptitle(f"Swing+ Top 10 — 25 & Under ({MLB_SEASON})",
                 fontsize=19, fontweight="bold", color="white", y=0.97)
    ax.set_title(f"{MLB_SEASON} Season  ·  Pure Mechanics Model  ·  Min 50 Swings  ·  Age ≤ 25",
                 fontsize=8.5, color="#777777", pad=10)

    fig.text(0.03, 0.01, "Data: Baseball Savant Bat Tracking", fontsize=6.5, color="#555555")
    fig.text(0.97, 0.01, "@BachTalk1", fontsize=6.5, color="#555555", ha="right")

    if WATERMARK_PATH.exists():
        try:
            img = PILImage.open(WATERMARK_PATH).convert("RGBA")
            arr = np.array(img, dtype=np.float32)
            arr[(arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240), 3] = 0
            nt = arr[:, :, 3] > 0; arr[nt, 0] = 255; arr[nt, 1] = 255; arr[nt, 2] = 255
            ax_wm = fig.add_axes([0.375, 0.3, 0.25, 0.25], zorder=10)
            ax_wm.imshow(arr.astype(np.uint8), alpha=0.12)
            ax_wm.set_facecolor("none"); ax_wm.axis("off")
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = SCREENSHOTS_DIR / "swing_plus_young.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close()
    return out


class SwingPlusYoungGenerator(ContentGenerator):
    name = "swing_plus_young"

    async def generate(self) -> PostContent:
        result = _compute_swing_plus()
        if result is None:
            log.warning("Swing+ computation failed")
            return PostContent(text="")

        df = result

        # Filter to 25 and under
        valid_ids = df["player_id"].dropna().astype(int).tolist()
        if not valid_ids:
            log.warning("No player IDs for age lookup")
            return PostContent(text="")

        ages = _get_player_ages(valid_ids)
        df["age"] = df["player_id"].apply(lambda x: ages.get(int(x)) if pd.notna(x) else None)
        young = df[df["age"].notna() & (df["age"] <= MAX_AGE)].copy()

        if len(young) < 5:
            log.warning("Only %d hitters aged ≤ %d, need at least 5", len(young), MAX_AGE)
            return PostContent(text="")

        image_path = _build_young_top10_image(young)
        if not image_path:
            return PostContent(text="")

        top10 = young.nlargest(10, "swing_plus").reset_index(drop=True)
        names = [r["name_fg"] for _, r in top10.head(3).iterrows()]

        text = (
            f"Swing+ Top 10 — 25 & Under ({MLB_SEASON})\n\n"
            f"The best young swings in baseball by our pure mechanics model. "
            f"{names[0]}, {names[1]}, and {names[2]} lead the way.\n\n"
            f"Thread with video highlights below\n\n"
            f"@TJStats {DEFAULT_HASHTAGS} #BatTracking"
        )

        # Build replies — video for top 5 only
        replies = []
        for i, (_, row) in enumerate(top10.iterrows()):
            hitter_name = row["name_fg"]
            sp = row["swing_plus"]
            bat_spd = row.get("bat_speed", 0)
            sq_up = row.get("squared_up_rate", 0)
            sw_len = row.get("swing_length", 0)
            xwoba = row.get("xwOBA", 0)
            hard_sw = row.get("sweetspot_speed_high", 0)
            age = int(row.get("age", 0))

            line = f"#{i+1} {hitter_name} (Age {age}) — Swing+ {sp:.1f}"

            stats = []
            if bat_spd:
                stats.append(f"Bat Speed {bat_spd:.1f}mph")
            if sq_up:
                sq_pct = sq_up * 100 if sq_up <= 1 else sq_up
                stats.append(f"Squared Up {sq_pct:.0f}%")
            if sw_len:
                stats.append(f"Swing Length {sw_len:.1f}ft")
            if hard_sw:
                hs_pct = hard_sw * 100 if hard_sw <= 1 else hard_sw
                stats.append(f"Hard Swing {hs_pct:.0f}%")
            if xwoba and xwoba > 0:
                stats.append(f"xwOBA {xwoba:.3f}")

            details = " | ".join(stats)
            reply_text = f"{line}\n{details}"

            video_path = None
            if i < 5:
                pid = row.get("player_id")
                if pid and not np.isnan(pid):
                    video_path = _get_savant_video(int(pid), hitter_name)

            replies.append(PostContent(
                text=reply_text,
                video_path=video_path,
                tags=["swing_plus_young", hitter_name],
            ))

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Swing+ Top 10 hitters aged 25 and under for {MLB_SEASON}",
            tags=["swing_plus_young"],
            replies=replies,
        )
