"""Fetch pitcher video clips from MLB Film Room via Statcast pitch data."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from pybaseball import statcast_pitcher

from .config import CLIPS_DIR, MLB_SEASON

log = logging.getLogger(__name__)

FILMROOM_GRAPHQL = "https://fastball-gateway.mlb.com/graphql"

# Statcast pitch_type codes → Film Room search keywords
_PITCH_TYPE_MAP = {
    "FF": "Four-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "SV": "Sweeper",
    "ST": "Sweeper",
    "KN": "Knuckleball",
}

MAX_CLIP_SIZE_MB = 50


def _fetch_statcast(pitcher_id: int, start: date, end: date):
    """Fetch Statcast data for a date range. Returns DataFrame or None."""
    log.info(
        "Fetching Statcast pitches for pitcher %s (%s to %s)",
        pitcher_id, start, end,
    )
    try:
        df = statcast_pitcher(
            start_dt=start.strftime("%Y-%m-%d"),
            end_dt=end.strftime("%Y-%m-%d"),
            player_id=pitcher_id,
        )
    except Exception:
        log.warning("Failed to fetch Statcast data for pitcher %s", pitcher_id, exc_info=True)
        return None
    if df is None or df.empty:
        return None
    return df


def find_best_pitch(pitcher_id: int, lookback_days: int = 14) -> dict | None:
    """Pull recent Statcast data and pick the nastiest pitch.

    Priority: strikeouts first (by release_speed desc), then hardest pitch.
    During offseason, falls back to last month of the previous completed season.
    Returns a dict of pitch attributes or None.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    df = _fetch_statcast(pitcher_id, start, end)

    # Offseason fallback: use the last month of MLB_SEASON (Sep 1 – Oct 31)
    if df is None:
        fallback_end = date(MLB_SEASON, 10, 31)
        fallback_start = date(MLB_SEASON, 9, 1)
        log.info("No recent data — falling back to %s season (%s to %s)", MLB_SEASON, fallback_start, fallback_end)
        df = _fetch_statcast(pitcher_id, fallback_start, fallback_end)

    if df is None:
        log.info("No Statcast data found for pitcher %s", pitcher_id)
        return None

    # Drop rows missing key fields
    required = ["release_speed", "batter", "game_date", "pitch_type", "inning", "balls", "strikes"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if df.empty:
        return None

    # Prefer strikeouts — events containing "strikeout" or "struck"
    best = None
    if "events" in df.columns:
        strikeouts = df[df["events"].str.contains("strikeout", case=False, na=False)]
        if not strikeouts.empty:
            best = strikeouts.loc[strikeouts["release_speed"].idxmax()]

    # Fallback: hardest pitch overall
    if best is None:
        best = df.loc[df["release_speed"].idxmax()]

    return {
        "pitcher_id": int(pitcher_id),
        "batter_id": int(best["batter"]),
        "game_date": str(best["game_date"])[:10],
        "pitch_type": str(best.get("pitch_type", "")),
        "pitch_name": str(best.get("pitch_name", "")),
        "inning": int(best["inning"]),
        "balls": int(best["balls"]),
        "strikes": int(best["strikes"]),
        "release_speed": float(best["release_speed"]),
    }


def _search_filmroom(pitcher_id: int, season: int) -> tuple[str | None, str | None]:
    """Search MLB Film Room for a strikeout clip of this pitcher.

    Returns (mp4_url, clip_title) or (None, None).
    """
    search_terms = f"PitcherId = {pitcher_id} AND Season = {season}"

    query = """
    {
      search(query: "%s", limit: 20) {
        total
        plays {
          gameDate
          mediaPlayback {
            slug
            title
            feeds {
              playbacks {
                name
                url
              }
            }
          }
        }
      }
    }
    """ % search_terms

    try:
        resp = requests.post(
            FILMROOM_GRAPHQL,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        log.warning("Film Room search request failed", exc_info=True)
        return None, None

    if data.get("errors"):
        log.warning("Film Room query error: %s", data["errors"][0].get("message", ""))
        return None, None

    plays = (data.get("data", {}).get("search") or {}).get("plays") or []
    if not plays:
        log.info("No Film Room clips found for pitcher %s season %s", pitcher_id, season)
        return None, None

    # Score and rank clips — prefer single-pitch strikeout clips over recaps
    scored = []
    for play in plays:
        mp_list = play.get("mediaPlayback") or []
        if not mp_list:
            continue
        mp = mp_list[0]
        title = (mp.get("title") or "").lower()
        slug = (mp.get("slug") or "").lower()
        score = 0
        # Strikeout content gets a boost
        if "strike" in title or "strike" in slug or "k's" in title:
            score += 10
        # Single-pitch clips: "K's [Name]", "strikes out [Name]", "looking", "swinging"
        if "k's " in title or "swinging" in title or "looking" in title:
            score += 5
        # Recaps are long and too large — penalize number words and recap indicators
        for word in ("six", "seven", "eight", "nine", "ten", "scoreless", "outing",
                      "innings", "earns", "career", "season", "complete"):
            if word in title:
                score -= 8
                break
        scored.append((score, mp))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None, None

    # Try candidates in order — check file size with HEAD request before downloading
    for _score, mp in scored:
        title = mp.get("title", "")
        mp4_url = _pick_best_mp4(mp)
        if not mp4_url:
            continue
        # Quick size check via HEAD to avoid downloading huge recap clips
        try:
            head = requests.head(mp4_url, timeout=10, allow_redirects=True)
            size = int(head.headers.get("Content-Length", 0))
            if size > MAX_CLIP_SIZE_MB * 1024 * 1024:
                log.info("Skipping clip (%.0f MB): %s", size / (1024 * 1024), title)
                continue
        except Exception:
            pass  # If HEAD fails, we'll catch it during download
        log.info("Found Film Room clip: %s", title)
        return mp4_url, title

    log.info("No suitably-sized clips found for pitcher %s", pitcher_id)
    return None, None


def _pick_best_mp4(media_playback: dict) -> str | None:
    """Extract the best quality MP4 URL from a mediaPlayback object."""
    feeds = media_playback.get("feeds") or []
    best_url = None
    best_quality = 0

    for feed in feeds:
        for pb in feed.get("playbacks", []):
            url = pb.get("url", "")
            name = pb.get("name", "")
            if not url.endswith(".mp4"):
                continue
            # Prefer higher bitrate; "highBit" > "mp4Avc" > other
            quality = 0
            if "highBit" in name:
                quality = 3
            elif "mp4Avc" in name:
                quality = 2
            elif "mp4" in name.lower():
                quality = 1
            if quality > best_quality or best_url is None:
                best_url = url
                best_quality = quality

    return best_url


def _download_mp4(url: str, output_path: Path) -> bool:
    """Stream-download an MP4 file. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > MAX_CLIP_SIZE_MB * 1024 * 1024:
                log.warning("Clip too large (%d MB), skipping", content_length // (1024 * 1024))
                return False

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_CLIP_SIZE_MB:
            log.warning("Downloaded clip too large (%.1f MB), removing", size_mb)
            output_path.unlink(missing_ok=True)
            return False

        log.info("Downloaded clip: %s (%.1f MB)", output_path.name, size_mb)
        return True
    except Exception:
        log.warning("Failed to download MP4 from %s", url[:80], exc_info=True)
        output_path.unlink(missing_ok=True)
        return False


def _cleanup_old_clips(max_age_hours: int = 24) -> None:
    """Delete clips older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    for f in CLIPS_DIR.glob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            log.debug("Cleaned up old clip: %s", f.name)


def get_game_strikeout_clip(game_pk: int, pitcher_id: int, pitcher_name: str) -> Path | None:
    """Fetch a strikeout video clip for a specific pitcher in a specific game.

    Uses the MLB Stats API play-by-play endpoint to find strikeouts,
    then fetches the sporty-clips mp4 from Baseball Savant.

    Returns the Path to the downloaded MP4, or None if unavailable.
    """
    _cleanup_old_clips()

    log.info("Fetching game %d play-by-play for pitcher %d (%s)",
             game_pk, pitcher_id, pitcher_name)

    # 1. Get play-by-play data from MLB Stats API
    pbp_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
    try:
        resp = requests.get(pbp_url, timeout=20)
        resp.raise_for_status()
        pbp_data = resp.json()
    except Exception:
        log.warning("Failed to fetch play-by-play for game %d", game_pk, exc_info=True)
        return None

    all_plays = pbp_data.get("allPlays", [])
    if not all_plays:
        log.info("No plays found in game %d", game_pk)
        return None

    # 2. Find this pitcher's best play (prefer K, fallback to swinging strike, then any play)
    strikeout_play_ids: list[str] = []
    swinging_strike_play_ids: list[str] = []
    any_play_ids: list[str] = []
    for play in all_plays:
        matchup = play.get("matchup", {})
        play_pitcher = matchup.get("pitcher", {})
        if play_pitcher.get("id") != pitcher_id:
            continue

        result = play.get("result", {})
        event = (result.get("event") or "").lower()
        play_events = play.get("playEvents", [])

        # Collect the last pitch's playId
        last_play_id = None
        for pe in reversed(play_events):
            pid = pe.get("playId")
            if pid:
                last_play_id = pid
                break

        if not last_play_id:
            continue

        if "strikeout" in event:
            strikeout_play_ids.append(last_play_id)
        else:
            any_play_ids.append(last_play_id)

        # Also check for swinging strikes within the at-bat
        for pe in play_events:
            desc = (pe.get("details", {}).get("description") or "").lower()
            pid = pe.get("playId")
            if pid and "swinging strike" in desc:
                swinging_strike_play_ids.append(pid)

    # Pick best available: strikeout > swinging strike > any play with a playId
    if strikeout_play_ids:
        target_play_id = strikeout_play_ids[-1]
        clip_type = "strikeout"
    elif swinging_strike_play_ids:
        target_play_id = swinging_strike_play_ids[-1]
        clip_type = "swinging strike"
    elif any_play_ids:
        target_play_id = any_play_ids[-1]
        clip_type = "pitch"
    else:
        log.info("No plays with video found for pitcher %d in game %d", pitcher_id, game_pk)
        return None

    log.info("Found %s clip for %s (K=%d, SwStr=%d); playId=%s",
             clip_type, pitcher_name, len(strikeout_play_ids),
             len(swinging_strike_play_ids), target_play_id)

    # 4. Fetch mp4 URL from Baseball Savant sporty-videos
    sporty_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={target_play_id}"
    mp4_url = None
    try:
        resp = requests.get(sporty_url, timeout=15)
        resp.raise_for_status()
        # Try JSON first (legacy format)
        try:
            video_data = resp.json()
            if isinstance(video_data, list) and video_data:
                for item in video_data:
                    if isinstance(item, dict):
                        mp4_url = item.get("video_url") or item.get("url")
                    elif isinstance(item, str) and item.endswith(".mp4"):
                        mp4_url = item
                    if mp4_url:
                        break
            elif isinstance(video_data, dict):
                mp4_url = video_data.get("video_url") or video_data.get("url")
        except (ValueError, TypeError):
            pass
        # Fallback: parse mp4 URL from HTML response
        if not mp4_url:
            import re
            mp4_matches = re.findall(r'(https?://[^\s"\']+\.mp4[^\s"\']*)', resp.text)
            if mp4_matches:
                mp4_url = mp4_matches[0]
                log.info("Extracted mp4 URL from HTML for playId=%s", target_play_id)
    except Exception:
        log.warning("Failed to fetch sporty-videos for playId=%s", target_play_id,
                    exc_info=True)
        return None

    if not mp4_url:
        log.info("No video URL returned for playId=%s", target_play_id)
        return None

    # 5. Download the clip
    safe_name = pitcher_name.replace(" ", "_").lower()
    output_path = CLIPS_DIR / f"game_{game_pk}_{safe_name}_k.mp4"

    if output_path.exists():
        log.info("Game K clip already downloaded: %s", output_path.name)
        return output_path

    if _download_mp4(mp4_url, output_path):
        return output_path

    return None


def get_pitcher_clip(pitcher_id: int, pitcher_name: str) -> Path | None:
    """Main entry point: find and download a video clip for a pitcher.

    Returns the Path to the downloaded MP4, or None if unavailable.
    """
    _cleanup_old_clips()

    # Try current season first, then previous season
    mp4_url, title = _search_filmroom(pitcher_id, MLB_SEASON)
    season_used = MLB_SEASON
    if not mp4_url:
        mp4_url, title = _search_filmroom(pitcher_id, MLB_SEASON - 1)
        season_used = MLB_SEASON - 1
    if not mp4_url:
        return None

    safe_name = pitcher_name.replace(" ", "_").lower()
    output_path = CLIPS_DIR / f"{safe_name}_{season_used}.mp4"

    if output_path.exists():
        log.info("Clip already downloaded: %s", output_path.name)
        return output_path

    if _download_mp4(mp4_url, output_path):
        return output_path

    return None
