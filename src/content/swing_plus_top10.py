"""Content generator: Weekly Top 10 Swing+ leaderboard (Mondays).

Pure mechanics model — trained on bat tracking features only.
Uses Baseball Savant bat tracking data. Thread format: header image
+ individual replies with Savant video for each hitter.
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage, ImageDraw

from .base import ContentGenerator, PostContent
from ..config import SCREENSHOTS_DIR, MLB_SEASON, DEFAULT_HASHTAGS
from ..video_clips import _download_mp4

log = logging.getLogger(__name__)

WATERMARK_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "BachTalk.png"

# Pure mechanics features (no delta_run_exp)
FEATURES = [
    "bat_speed", "squared_up_rate", "squared_up_speed_rate",
    "swing_length", "sweetspot_speed_high", "hit_into_play_rate", "swords",
]

# Column mapping: Savant CSV name -> model feature name
# Handles both 2025 and 2026 column formats
_COL_MAPS = [
    # 2026 format
    {"avg_bat_speed": "bat_speed", "squared_up_per_bat_contact": "squared_up_rate",
     "blast_per_bat_contact": "squared_up_speed_rate", "swing_length": "swing_length",
     "hard_swing_rate": "sweetspot_speed_high",
     "batted_ball_event_per_swing": "hit_into_play_rate", "swords": "swords"},
    # 2025 format
    {"avg_sweetspot_speed_mph": "bat_speed", "squared_up_per_bat_contact": "squared_up_rate",
     "squared_up_with_speed_per_bat_contact": "squared_up_speed_rate",
     "swing_length": "swing_length", "avg_is_sweetspot_speed_high": "sweetspot_speed_high",
     "hit_into_play_per_swing": "hit_into_play_rate", "swords": "swords"},
]


def _compute_swing_plus():
    """Compute Swing+ for all qualified hitters."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from pybaseball import batting_stats

    # Fetch bat tracking CSV
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking?"
        f"attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd=&gameType=&"
        f"isHardHit=&minSwings=50&minGroupSwings=1&pitchHand=&pitchType=&"
        f"playerPool=All&season={MLB_SEASON}&seasonStart=&seasonEnd=&team=&"
        f"type=batter&csv=true"
    )
    try:
        resp = requests.get(url, timeout=30)
        bt = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        log.warning("Bat tracking fetch failed", exc_info=True)
        return None

    if bt.empty or len(bt) < 20:
        log.warning("Not enough bat tracking data: %d rows", len(bt))
        return None

    # Find the right column mapping
    col_map = None
    for cm in _COL_MAPS:
        missing = [k for k in cm.keys() if k not in bt.columns]
        if not missing:
            col_map = cm
            break
        # Try qualified versions
        adjusted = dict(cm)
        for k in missing:
            qual = k + "_qualified"
            if qual in bt.columns:
                bt[k] = bt[qual]
        missing2 = [k for k in cm.keys() if k not in bt.columns]
        if not missing2:
            col_map = cm
            break

    if col_map is None:
        log.warning("Cannot map bat tracking columns. Available: %s", bt.columns.tolist()[:15])
        return None

    bt_clean = bt.rename(columns=col_map)

    # Name + ID columns
    name_col = next((c for c in ["name", "batter_name", "player_name"] if c in bt_clean.columns), None)
    id_col = next((c for c in ["id", "batter_id", "player_id", "savant_batter_id"] if c in bt_clean.columns), None)

    if not name_col:
        return None

    bt_clean["name_fg"] = bt_clean[name_col].apply(
        lambda x: " ".join(str(x).split(", ")[::-1]).strip() if ", " in str(x) else str(x).strip()
    )

    # Convert features to numeric
    for f in FEATURES:
        if f in bt_clean.columns:
            bt_clean[f] = pd.to_numeric(bt_clean[f], errors="coerce")

    bt_clean = bt_clean.dropna(subset=FEATURES)

    if len(bt_clean) < 20:
        log.warning("Only %d hitters after dropna", len(bt_clean))
        return None

    # Get xwOBA for training
    try:
        fg = batting_stats(MLB_SEASON, qual=20)
        fg["name_fg"] = fg["Name"].str.strip()
        merged = bt_clean.merge(fg[["name_fg", "xwOBA"]], on="name_fg", how="inner")
        merged = merged.dropna(subset=["xwOBA"])
    except Exception:
        log.warning("FanGraphs xwOBA fetch failed — using raw z-scores", exc_info=True)
        merged = bt_clean.copy()
        merged["xwOBA"] = 0  # fallback: no training target

    if len(merged) < 20:
        log.warning("Only %d merged rows", len(merged))
        return None

    # Train model
    X = merged[FEATURES].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if merged["xwOBA"].sum() > 0:
        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=min(10, len(merged))).fit(X_scaled, merged["xwOBA"].values)
        pred = ridge.predict(X_scaled)
    else:
        # Fallback: simple z-score composite with equal weights
        pred = X_scaled.mean(axis=1)

    pm, ps = pred.mean(), pred.std()
    if ps == 0:
        ps = 1
    merged["swing_plus"] = np.round(100 + ((pred - pm) / ps) * 15, 1)

    # Store player_id
    if id_col and id_col in merged.columns:
        merged["player_id"] = pd.to_numeric(merged[id_col], errors="coerce")
    else:
        merged["player_id"] = None

    return merged


def _fetch_headshot(player_id):
    """Download circular headshot."""
    from PIL import Image
    try:
        url = f"https://securea.mlb.com/mlb/images/players/head_shot/{int(player_id)}.jpg"
        resp = requests.get(url, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        size = min(img.size)
        l = (img.width - size) // 2; t = (img.height - size) // 2
        img = img.crop((l, t, l + size, t + size)).resize((100, 100), Image.LANCZOS)
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 100, 100), fill=255)
        img.putalpha(mask)
        return np.array(img)
    except Exception:
        return None


def _build_top10_image(df):
    """Build horizontal bar chart with headshots — dark theme."""
    top10 = df.nlargest(10, "swing_plus").reset_index(drop=True)
    n = len(top10)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bg_color = "#141414"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    bar_color = "#3a86ff"
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
        subtitle = f"{bat_spd:.1f} mph"
        if xw_str:
            subtitle += f"  ·  {xw_str} xwOBA"
        ax.text(bar_left + 0.2, y - 0.17, subtitle, ha="left", va="center",
                fontsize=7, color="#cc3333", zorder=5)

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

    fig.suptitle(f"Top 10 Hitters by Swing+ ({MLB_SEASON})",
                 fontsize=19, fontweight="bold", color="white", y=0.97)
    ax.set_title(f"{MLB_SEASON} Season  ·  Pure Mechanics Model  ·  Min 50 Swings",
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
    out = SCREENSHOTS_DIR / "swing_plus_top10.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close()
    return out


def _get_savant_video(player_id, player_name):
    """Get a Savant video clip for a hitter (home run or hard hit)."""
    from pybaseball import statcast_batter
    from datetime import date, timedelta

    pid = int(player_id)

    # Try progressively wider windows: 14 days, then 30 days
    for lookback in [14, 30]:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=lookback)

        try:
            df = statcast_batter(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), pid)
            if df is None or df.empty:
                continue

            # Only use hits — prefer HR, then XBH, then singles
            hrs = df[df["events"] == "home_run"]
            xbh = df[df["events"].isin(["double", "triple", "home_run"])]
            hits = df[df["events"].isin(["single", "double", "triple", "home_run"])]
            target = hrs if not hrs.empty else xbh if not xbh.empty else hits

            if target.empty:
                continue

            # Try multiple candidate plays
            hit_events = {"single", "double", "triple", "home_run"}
            for _, row in target.head(5).iterrows():
                game_pk = int(row["game_pk"])
                event_type = str(row.get("events", "")).lower()
                pbp = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay",
                    timeout=15,
                ).json()

                for play in pbp.get("allPlays", []):
                    if play.get("matchup", {}).get("batter", {}).get("id") != pid:
                        continue
                    # Only match PBP plays that are actual hits
                    pbp_event = (play.get("result", {}).get("event") or "").lower().replace(" ", "_")
                    if pbp_event not in hit_events:
                        continue
                    for pe in reversed(play.get("playEvents", [])):
                        play_id = pe.get("playId")
                        if not play_id:
                            continue
                        surl = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                        sr = requests.get(surl, timeout=10)
                        mp4s = re.findall(
                            r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', sr.text
                        )
                        if mp4s:
                            safe = player_name.replace(" ", "_").lower()
                            clip_path = (
                                SCREENSHOTS_DIR.parent / "data" / "clips"
                                / f"swing_{safe}_{pid}.mp4"
                            )
                            clip_path.parent.mkdir(parents=True, exist_ok=True)
                            if _download_mp4(mp4s[0], clip_path):
                                log.info("Got %s video: %s from game %s", player_name, pbp_event, game_pk)
                                return clip_path
                        break  # only try last playEvent per at-bat

        except Exception:
            log.warning("Savant video failed for %s (lookback=%d)", player_name, lookback, exc_info=True)

    log.warning("No video found for %s (id=%s)", player_name, player_id)
    return None


class SwingPlusTop10Generator(ContentGenerator):
    name = "swing_plus_top10"

    async def generate(self) -> PostContent:
        result = _compute_swing_plus()
        if result is None:
            log.warning("Swing+ computation failed")
            return PostContent(text="")

        df = result
        image_path = _build_top10_image(df)
        if not image_path:
            return PostContent(text="")

        top10 = df.nlargest(10, "swing_plus").reset_index(drop=True)
        names = [r["name_fg"] for _, r in top10.head(3).iterrows()]

        # Header tweet
        text = (
            f"Updated Swing+ Top 10 — {MLB_SEASON}\n\n"
            f"Pure mechanics model trained on bat tracking data. "
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

            line = f"#{i+1} {hitter_name} — Swing+ {sp:.1f}"

            # Build stat description (~150 chars)
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

            # Video for top 5 only
            video_path = None
            if i < 5:
                pid = row.get("player_id")
                if pid and not np.isnan(pid):
                    video_path = _get_savant_video(int(pid), hitter_name)

            replies.append(PostContent(
                text=reply_text,
                video_path=video_path,
                tags=["swing_plus_top10", hitter_name],
            ))

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"Swing+ Top 10 leaderboard for {MLB_SEASON}",
            tags=["swing_plus_top10"],
            replies=replies,
        )
