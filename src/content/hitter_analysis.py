"""Content generator: Hitter Analysis thread (Tue/Fri).

Picks a hitter with interesting Swing+ profile, fetches their TJStats
percentile card screenshot, pulls Savant video clips, and builds an
analysis thread using Swing+ model data.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import ContentGenerator, PostContent
from ..config import SCREENSHOTS_DIR, CLIPS_DIR, MLB_SEASON, DEFAULT_HASHTAGS
from ..video_clips import _download_mp4
from .swing_plus_top10 import _compute_swing_plus, FEATURES

log = logging.getLogger(__name__)

# Savant swing path leaderboard for attack angle data
_SWING_PATH_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-path-attack-angle"
    "?gameType=Regular&minSwings=1&season={season}&type=batter&csv=true"
)


def _get_swing_path_data(season: int) -> pd.DataFrame | None:
    """Fetch attack angle / swing tilt data from Savant."""
    try:
        url = _SWING_PATH_URL.format(season=season)
        resp = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(resp.text))
        df["_pid"] = pd.to_numeric(df["id"], errors="coerce")
        return df
    except Exception:
        log.warning("Swing path data fetch failed", exc_info=True)
        return None


def _percentile(series: pd.Series, value: float) -> int:
    """Compute percentile rank of value within series."""
    return int((series < value).mean() * 100)


def _pick_hitter(df: pd.DataFrame) -> pd.Series | None:
    """Pick an interesting hitter to analyze.

    Prioritize hitters with a gap between their mechanics and results —
    high swing+ but low xwOBA (underperforming) or low swing+ but high
    xwOBA (overperforming). Also avoid repeating recently analyzed hitters.
    """
    from ..scheduler import was_recently_posted

    if df.empty:
        return None

    # Calculate gap between swing+ prediction and actual xwOBA
    xwoba_mean = df["xwOBA"].mean()
    xwoba_std = df["xwOBA"].std()
    if xwoba_std == 0:
        xwoba_std = 1
    df = df.copy()
    df["xwoba_plus"] = 100 + ((df["xwOBA"] - xwoba_mean) / xwoba_std) * 15
    df["gap"] = abs(df["swing_plus"] - df["xwoba_plus"])

    # Sort by gap (most interesting discrepancies first)
    candidates = df.nlargest(30, "gap")

    for _, row in candidates.iterrows():
        name = row.get("name_fg", "")
        tag = f"hitter_analysis_{name.replace(' ', '_').lower()}"
        if not was_recently_posted(tag, lookback=30):
            return row

    # Fallback: pick highest swing+ not recently posted
    for _, row in df.nlargest(20, "swing_plus").iterrows():
        name = row.get("name_fg", "")
        tag = f"hitter_analysis_{name.replace(' ', '_').lower()}"
        if not was_recently_posted(tag, lookback=30):
            return row

    return df.nlargest(1, "swing_plus").iloc[0]


def _get_hitter_videos(player_id: int, player_name: str, max_clips: int = 3):
    """Get Savant video clips for a hitter — hard out, grounder, hit."""
    from pybaseball import statcast_batter
    from datetime import date, timedelta

    pid = int(player_id)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)

    try:
        df = statcast_batter(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), pid)
    except Exception:
        log.warning("Statcast batter fetch failed for %s", player_name, exc_info=True)
        return {}

    if df is None or df.empty:
        return {}

    events = df[df["events"].notna()].copy()
    clips = {}
    safe_name = player_name.replace(" ", "_").lower()

    for _, row in events.iterrows():
        if len(clips) >= max_clips:
            break

        ev = row["events"]
        la = row.get("launch_angle", np.nan)
        ev_speed = row.get("launch_speed", np.nan)
        game_pk = int(row["game_pk"])

        if ev in ["walk", "strikeout", "hit_by_pitch", "caught_stealing_2b",
                   "pickoff_1b", "sac_fly", "sac_bunt"]:
            continue

        label = None
        if "grounder" not in clips and not np.isnan(la) and la < 0:
            label = "grounder"
        elif "hard_out" not in clips and ev == "field_out" and not np.isnan(ev_speed) and ev_speed > 95:
            label = "hard_out"
        elif "hit" not in clips and ev in ["single", "double", "triple", "home_run"]:
            label = "hit"

        if not label:
            continue

        try:
            pbp = requests.get(
                f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay",
                timeout=15,
            ).json()

            for play in pbp.get("allPlays", []):
                if play.get("matchup", {}).get("batter", {}).get("id") != pid:
                    continue
                for pe in reversed(play.get("playEvents", [])):
                    play_id = pe.get("playId")
                    if not play_id:
                        continue
                    if pe.get("isPitch") and pe.get("details", {}).get("isInPlay"):
                        hit_data = pe.get("hitData", {})
                        pe_la = hit_data.get("launchAngle")
                        if pe_la is not None and not np.isnan(la):
                            if abs(pe_la - la) > 2:
                                continue

                        surl = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
                        sr = requests.get(surl, timeout=10)
                        mp4s = re.findall(
                            r'https?://sporty-clips\.mlb\.com/[^\s"<>]+\.mp4', sr.text
                        )
                        if mp4s:
                            clip_path = CLIPS_DIR / f"analysis_{safe_name}_{label}.mp4"
                            if _download_mp4(mp4s[0], clip_path):
                                clips[label] = (clip_path, ev_speed, la)
                                log.info("Got %s %s video: EV=%.1f LA=%.1f",
                                         player_name, label, ev_speed, la)
                            break
                if label in clips:
                    break
        except Exception:
            log.warning("Video fetch failed for %s %s", player_name, label, exc_info=True)

    return clips


def _build_analysis_tweets(row: pd.Series, swing_path: pd.DataFrame | None,
                           league_df: pd.DataFrame) -> list[str]:
    """Build analysis tweet texts (each < 150 chars) from model data."""
    name = row.get("name_fg", "?")
    sp = row["swing_plus"]

    # Get attack angle data
    pid = row.get("player_id")
    aa = np.nan
    aa_pct = None
    tilt = np.nan
    tilt_pct = None
    if swing_path is not None and pid and not np.isnan(pid):
        sp_row = swing_path[swing_path["_pid"] == int(pid)]
        if not sp_row.empty:
            aa = sp_row.iloc[0].get("attack_angle", np.nan)
            tilt = sp_row.iloc[0].get("swing_tilt", np.nan)
            if not np.isnan(aa):
                aa_pct = _percentile(swing_path["attack_angle"].dropna(), aa)
            if not np.isnan(tilt):
                tilt_pct = _percentile(swing_path["swing_tilt"].dropna(), tilt)

    # Feature percentiles
    bat_spd = row.get("bat_speed", 0)
    sq_up = row.get("squared_up_rate", 0)
    blast = row.get("squared_up_speed_rate", 0)
    sw_len = row.get("swing_length", 0)
    hard_sw = row.get("sweetspot_speed_high", 0)
    hip = row.get("hit_into_play_rate", 0)
    swords = row.get("swords", 0)
    brl = row.get("brl_percent", 0)
    xwoba = row.get("xwOBA", 0)

    bat_spd_pct = _percentile(league_df["bat_speed"], bat_spd)
    sq_up_pct = _percentile(league_df["squared_up_rate"], sq_up * 100 if sq_up <= 1 else sq_up)
    brl_pct = _percentile(league_df["brl_percent"], brl)
    hard_sw_pct = _percentile(league_df["sweetspot_speed_high"], hard_sw * 100 if hard_sw <= 1 else hard_sw)
    hip_pct = _percentile(league_df["hit_into_play_rate"], hip * 100 if hip <= 1 else hip)
    sw_len_pct = _percentile(league_df["swing_length"], sw_len)

    tweets = []

    # Determine profile type
    high_contact = sq_up_pct >= 70 or hip_pct >= 70
    high_power = brl_pct >= 70 or bat_spd_pct >= 70
    low_barrels = brl_pct <= 25
    low_contact = sq_up_pct <= 30 or hip_pct <= 30
    flat_swing = aa_pct is not None and aa_pct <= 25

    if high_contact and low_barrels:
        # Contact-over-power profile (like Hayes)
        sq_pct = sq_up * 100 if sq_up <= 1 else sq_up
        tweets.append(
            f"{sq_pct:.0f}% squared up ({sq_up_pct}th pctl), "
            f"{hip * 100 if hip <= 1 else hip:.0f}% hit into play. "
            f"Elite bat-to-ball. That's not the issue."
        )
        tweets.append(
            f"{brl:.1f}% barrel rate ({brl_pct}th pctl). "
            f"{hard_sw * 100 if hard_sw <= 1 else hard_sw:.1f}% hard swing rate ({hard_sw_pct}th pctl). "
            f"Contact without damage."
        )
        if flat_swing and not np.isnan(aa) and not np.isnan(tilt):
            tweets.append(
                f"{aa:.0f}° attack angle ({aa_pct}th pctl), "
                f"{tilt:.0f}° swing tilt ({tilt_pct}th pctl). "
                f"Bat is too flat — needs more loft to barrel up."
            )
        else:
            tweets.append(
                f"Bat speed {bat_spd:.1f} mph ({bat_spd_pct}th pctl). "
                f"Swing length {sw_len:.1f} ft ({sw_len_pct}th pctl). "
                f"Needs more intent to generate damage."
            )
    elif high_power and low_contact:
        # Power-over-contact profile
        tweets.append(
            f"Bat speed {bat_spd:.1f} mph ({bat_spd_pct}th pctl). "
            f"{brl:.1f}% barrel rate ({brl_pct}th pctl). "
            f"The power is elite."
        )
        sq_pct = sq_up * 100 if sq_up <= 1 else sq_up
        tweets.append(
            f"{sq_pct:.0f}% squared up ({sq_up_pct}th pctl). "
            f"{int(swords)} swords. "
            f"Bat-to-ball needs work — too many whiffs."
        )
        tweets.append(
            f"Swing length {sw_len:.1f} ft ({sw_len_pct}th pctl). "
            f"A shorter path to the ball could improve contact without losing power."
        )
    elif high_power and high_contact:
        # Elite profile
        sq_pct = sq_up * 100 if sq_up <= 1 else sq_up
        tweets.append(
            f"Bat speed {bat_spd:.1f} mph ({bat_spd_pct}th pctl). "
            f"{brl:.1f}% barrels ({brl_pct}th pctl). "
            f"Elite damage output."
        )
        tweets.append(
            f"{sq_pct:.0f}% squared up ({sq_up_pct}th pctl). "
            f"{hip * 100 if hip <= 1 else hip:.0f}% hit into play ({hip_pct}th pctl). "
            f"Contact + power = the full package."
        )
        if not np.isnan(aa) and not np.isnan(tilt):
            tweets.append(
                f"{aa:.0f}° attack angle, {tilt:.0f}° tilt. "
                f"Swing path is optimized for damage. Model swing."
            )
        else:
            tweets.append(
                f"Swing length {sw_len:.1f} ft. xwOBA {xwoba:.3f}. "
                f"Mechanics match production. Nothing to fix."
            )
    else:
        # Generic analysis
        tweets.append(
            f"Bat speed {bat_spd:.1f} mph ({bat_spd_pct}th pctl). "
            f"{brl:.1f}% barrels ({brl_pct}th pctl). "
            f"Swing length {sw_len:.1f} ft ({sw_len_pct}th pctl)."
        )
        sq_pct = sq_up * 100 if sq_up <= 1 else sq_up
        tweets.append(
            f"{sq_pct:.0f}% squared up ({sq_up_pct}th pctl). "
            f"{hard_sw * 100 if hard_sw <= 1 else hard_sw:.1f}% hard swing ({hard_sw_pct}th pctl). "
            f"{hip * 100 if hip <= 1 else hip:.0f}% hit into play ({hip_pct}th pctl)."
        )
        if not np.isnan(aa) and not np.isnan(tilt):
            tweets.append(
                f"{aa:.0f}° attack angle ({aa_pct}th pctl). "
                f"{tilt:.0f}° swing tilt ({tilt_pct}th pctl). "
                f"xwOBA {xwoba:.3f}."
            )
        else:
            tweets.append(
                f"xwOBA {xwoba:.3f}. {int(swords)} swords. "
                f"Swing+ says {sp:.1f} — "
                f"{'outperforming' if xwoba > 0.320 else 'underperforming'} the mechanics."
            )

    # Truncate any tweet over 150 chars
    for i, t in enumerate(tweets):
        if len(t) > 150:
            tweets[i] = t[:147] + "..."

    return tweets


def _make_tjstats_slug(player_name: str, player_id: int) -> str:
    """Convert player name + ID to TJStats URL slug (e.g. 'ke-bryan-hayes-663647')."""
    slug = player_name.lower().replace("'", "").replace(".", "").replace(" ", "-")
    # Collapse multiple hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{slug}-{int(player_id)}"


async def _screenshot_tjstats_card(player_name: str, player_id: int) -> Path | None:
    """Screenshot the TJStats player page as the header card."""
    from playwright.async_api import async_playwright

    slug = _make_tjstats_slug(player_name, player_id)
    url = f"https://tjstats.ca/player/{slug}/"
    safe = player_name.replace(" ", "_").lower()
    out = SCREENSHOTS_DIR / f"analysis_{safe}_card.png"

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 800, "height": 1200},
                device_scale_factor=2,
            )
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)

            # Wait for card content to render
            import asyncio as _aio
            await _aio.sleep(3)

            await page.screenshot(path=str(out), full_page=False)
            await browser.close()

        log.info("TJStats card screenshot saved: %s", out)
        return out
    except Exception:
        log.warning("TJStats screenshot failed for %s", player_name, exc_info=True)
        return None


class HitterAnalysisGenerator(ContentGenerator):
    name = "hitter_analysis"

    async def generate(self) -> PostContent:
        # Compute Swing+ for all qualified hitters
        result = _compute_swing_plus()
        if result is None:
            log.warning("Swing+ computation failed")
            return PostContent(text="")

        df = result

        # Pick a hitter
        hitter = _pick_hitter(df)
        if hitter is None:
            log.warning("No hitter picked")
            return PostContent(text="")

        name = hitter.get("name_fg", "?")
        sp = hitter["swing_plus"]
        pid = hitter.get("player_id")
        xwoba = hitter.get("xwOBA", 0)

        log.info("Picked hitter: %s (Swing+ %.1f, xwOBA %.3f)", name, sp, xwoba)

        # Get swing path data
        swing_path = _get_swing_path_data(MLB_SEASON)

        # Build league percentile reference (convert rates to percentages for comparison)
        league_df = df.copy()
        for col in ["squared_up_rate", "squared_up_speed_rate", "sweetspot_speed_high",
                     "hit_into_play_rate"]:
            if col in league_df.columns:
                mask = league_df[col] <= 1
                league_df.loc[mask, col] = league_df.loc[mask, col] * 100

        # Build analysis tweets
        tweets = _build_analysis_tweets(hitter, swing_path, league_df)

        # Get videos
        videos = {}
        if pid and not np.isnan(pid):
            videos = _get_hitter_videos(int(pid), name)

        # Screenshot TJStats card as header image
        header_image = None
        if pid and not np.isnan(pid):
            header_image = await _screenshot_tjstats_card(name, int(pid))

        # Determine profile label
        brl = hitter.get("brl_percent", 0)
        sq_up = hitter.get("squared_up_rate", 0)
        if sq_up > 0.5 and brl < 5:
            profile = "Elite contact, no damage"
        elif brl > 15 and sq_up < 0.25:
            profile = "Big power, low contact"
        elif brl > 12 and sq_up > 0.35:
            profile = "Elite all-around swing"
        else:
            profile = f"xwOBA {xwoba:.3f}"

        # Header tweet
        header_text = (
            f"{name} Hitter Analysis — Swing+ {sp:.1f}\n\n"
            f"{profile}.\n\n"
            f"@TJStats {DEFAULT_HASHTAGS}"
        )

        # Build reply chain
        replies = []
        video_keys = ["hard_out", "grounder", "hit"]
        for i, tweet_text in enumerate(tweets):
            video_path = None
            if i < len(video_keys) and video_keys[i] in videos:
                video_path = videos[video_keys[i]][0]  # (path, ev, la)

            replies.append(PostContent(
                text=tweet_text,
                video_path=video_path,
                tags=["hitter_analysis", name],
            ))

        tag = f"hitter_analysis_{name.replace(' ', '_').lower()}"
        return PostContent(
            text=header_text,
            image_path=header_image,
            alt_text=f"{name} {MLB_SEASON} batting percentiles",
            tags=["hitter_analysis", tag],
            replies=replies,
        )
