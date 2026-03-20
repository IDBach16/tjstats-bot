"""Content generator: daily Reds game pitching summary thread.

Checks if the Reds played yesterday via MLB Stats API, then builds
a multi-tweet thread with pitcher cards and strikeout video clips.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..charts import plot_reds_game_summary
from ..config import MLB_SEASON, MLB_API_BASE
from ..video_clips import get_game_strikeout_clip

log = logging.getLogger(__name__)

REDS_TEAM_ID = 113
REDS_TEAM_ABBREV = "CIN"

# Noise pitch types to filter
_NOISE_PITCHES = {"PO", "IN", "EP", "AB", "AS", "UN", "XX", "NP", "SC"}

# Team abbreviation lookup from full MLB API team names
_TEAM_NAME_TO_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _summary_line(row: pd.Series, name: str) -> str:
    """Build a one-line text summary like 'Hunter Greene: 5.0 IP, 2 ER, 8 K'."""
    parts = [name + ":"]
    for col, label in [
        ("innings_pitched", "IP"),
        ("earned_runs", "ER"),
        ("strike_outs", "K"),
        ("walks", "BB"),
        ("pitches_thrown", "P"),
    ]:
        if col in row.index:
            try:
                val = float(row[col])
                if col == "innings_pitched":
                    parts.append(f"{val:.1f} {label}")
                else:
                    parts.append(f"{int(val)} {label}")
            except (TypeError, ValueError):
                pass
    return ", ".join(parts)


class RedsSummaryGenerator(ContentGenerator):
    name = "reds_summary"

    async def generate(self) -> PostContent:
        yesterday = date.today() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        display_date = yesterday.strftime("%m/%d/%Y")
        log.info("Checking for Reds game on %s", date_str)

        # ── 1. Check MLB schedule for yesterday's game ─────────────
        game_info = self._get_game_info(date_str)
        if game_info is None:
            log.info("No Reds game found for %s", date_str)
            return PostContent(
                text=f"No Reds game yesterday \u2014 {display_date}\n\n@TJStats #Reds #MLB",
                tags=["reds_summary", "no_game"],
            )

        game_pk = game_info["game_pk"]
        opponent = game_info["opponent"]
        score_line = game_info["score_line"]
        log.info("Found Reds game: pk=%d vs %s — %s", game_pk, opponent, score_line)

        # ── 2. Fetch pitcher data from Pitch Profiler ──────────────
        try:
            game_pitchers_df = pitch_profiler.get_game_pitchers(MLB_SEASON)
            game_pitches_df = pitch_profiler.get_game_pitches(MLB_SEASON)
            pbp_df = pitch_profiler.get_pbp_game(game_pk)
        except Exception:
            log.warning("Failed to fetch Pitch Profiler data", exc_info=True)
            return PostContent(
                text=f"Reds {score_line} vs {opponent} ({display_date})\n\n"
                     f"Pitch data unavailable today.\n\n@TJStats #Reds #MLB",
                tags=["reds_summary", "data_error"],
            )

        # ── 3. Filter to Reds pitchers for this game ───────────────
        gp_col = _find_col(game_pitchers_df, ["game_pk"])
        if gp_col is None:
            for c in game_pitchers_df.columns:
                if "game" in c.lower() and "pk" in c.lower():
                    gp_col = c
                    break
        if gp_col is None:
            log.error("Cannot find game_pk column in game_pitchers. Columns: %s",
                      list(game_pitchers_df.columns))
            return PostContent(text="", tags=["reds_summary", "error"])

        game_pitchers_df[gp_col] = pd.to_numeric(game_pitchers_df[gp_col], errors="coerce")
        reds_pitchers = game_pitchers_df[game_pitchers_df[gp_col] == game_pk].copy()

        # Filter by team
        team_col = _find_col(reds_pitchers, ["team", "team_abbreviation", "team_abbrev"])
        if team_col:
            reds_pitchers = reds_pitchers[
                reds_pitchers[team_col].str.upper() == REDS_TEAM_ABBREV
            ]

        if reds_pitchers.empty:
            log.warning("No Reds pitchers found for game_pk=%d", game_pk)
            return PostContent(
                text=f"Reds {score_line} vs {opponent} ({display_date})\n\n"
                     f"Pitcher data not yet available.\n\n@TJStats #Reds #MLB",
                tags=["reds_summary", "no_pitchers"],
            )

        name_col = _find_col(reds_pitchers, ["pitcher_name", "player_name", "name"])
        id_col = _find_col(reds_pitchers, ["pitcher_id", "player_id", "mlbam_id"])

        if name_col is None:
            log.error("No name column found. Columns: %s", list(reds_pitchers.columns))
            return PostContent(text="", tags=["reds_summary", "error"])

        # Get pitcher IDs
        pitcher_ids: set[int] = set()
        if id_col:
            reds_pitchers[id_col] = pd.to_numeric(reds_pitchers[id_col], errors="coerce")
            pitcher_ids = set(reds_pitchers[id_col].dropna().astype(int))

        # ── 4. Filter game pitches to Reds pitchers ────────────────
        reds_pitches = game_pitches_df.copy()
        gp_pitch_col = _find_col(reds_pitches, ["game_pk"])
        if gp_pitch_col is None:
            for c in reds_pitches.columns:
                if "game" in c.lower() and "pk" in c.lower():
                    gp_pitch_col = c
                    break
        if gp_pitch_col:
            reds_pitches[gp_pitch_col] = pd.to_numeric(reds_pitches[gp_pitch_col], errors="coerce")
            reds_pitches = reds_pitches[reds_pitches[gp_pitch_col] == game_pk]

        game_pitch_id_col = _find_col(reds_pitches, ["pitcher_id", "player_id", "mlbam_id"])
        if game_pitch_id_col and pitcher_ids:
            reds_pitches[game_pitch_id_col] = pd.to_numeric(
                reds_pitches[game_pitch_id_col], errors="coerce"
            )
            reds_pitches = reds_pitches[reds_pitches[game_pitch_id_col].isin(pitcher_ids)]

        log.info("Reds pitcher rows: %d, pitch rows: %d", len(reds_pitchers), len(reds_pitches))

        # ── 5. Generate cards for each pitcher ─────────────────────
        cards: list[dict] = []  # {name, pid, card_path, summary, video_path}

        for idx, (_, player_row) in enumerate(reds_pitchers.iterrows()):
            pname = str(player_row[name_col])
            pid = int(player_row[id_col]) if id_col and pd.notna(player_row[id_col]) else None

            log.info("Generating card for %s (pid=%s)", pname, pid)

            # Get this pitcher's pitch-type data
            if game_pitch_id_col and pid and not reds_pitches.empty:
                pitcher_pitches = reds_pitches[reds_pitches[game_pitch_id_col] == pid]
            else:
                pitch_name_col = _find_col(reds_pitches, ["pitcher_name", "player_name", "name"])
                if pitch_name_col:
                    pitcher_pitches = reds_pitches[reds_pitches[pitch_name_col] == pname]
                else:
                    pitcher_pitches = pd.DataFrame()

            # Filter PBP data for this pitcher
            pitcher_pbp = pd.DataFrame()
            if pbp_df is not None and not pbp_df.empty:
                pbp_pid_col = _find_col(pbp_df, ["pitcher_id", "pitcher", "player_id", "mlbam_id"])
                if pbp_pid_col and pid:
                    pbp_df[pbp_pid_col] = pd.to_numeric(pbp_df[pbp_pid_col], errors="coerce")
                    pitcher_pbp = pbp_df[pbp_df[pbp_pid_col] == pid]
                else:
                    pbp_name_col = _find_col(pbp_df, ["pitcher_name", "player_name", "name"])
                    if pbp_name_col:
                        pitcher_pbp = pbp_df[
                            pbp_df[pbp_name_col].str.contains(
                                pname.split()[-1], case=False, na=False
                            )
                        ]

            card_path = plot_reds_game_summary(
                name=pname,
                game_stats=player_row,
                game_pitches_df=pitcher_pitches,
                all_pitches_df=game_pitches_df,
                team=REDS_TEAM_ABBREV,
                player_id=pid,
                pbp_df=pitcher_pbp,
                season_df=game_pitchers_df,
                game_date=display_date,
                opponent=opponent,
                season=MLB_SEASON,
            )

            if not card_path:
                log.warning("Card generation failed for %s", pname)
                continue

            summary = _summary_line(player_row, pname)

            # Try to get a game-specific strikeout clip
            video_path = None
            if pid:
                try:
                    video_path = get_game_strikeout_clip(game_pk, pid, pname)
                    if video_path:
                        log.info("Got K clip for %s: %s", pname, video_path)
                except Exception:
                    log.warning("K clip failed for %s", pname, exc_info=True)

            cards.append({
                "name": pname,
                "pid": pid,
                "card_path": card_path,
                "summary": summary,
                "video_path": video_path,
            })

        if not cards:
            log.warning("No cards were generated for game %d", game_pk)
            return PostContent(
                text=f"Reds {score_line} vs {opponent} ({display_date})\n\n"
                     f"Card generation failed.\n\n@TJStats #Reds #MLB",
                tags=["reds_summary", "card_error"],
            )

        # ── 6. Build thread ────────────────────────────────────────
        # Main tweet: starter's card + recap text with all pitcher lines
        all_summaries = "\n".join(c["summary"] for c in cards)
        main_text = (
            f"Reds Pitching Recap \u2014 {display_date}\n"
            f"{score_line}\n\n"
            f"{all_summaries}\n\n"
            f"@TJStats #Reds #MLB"
        )

        # Truncate if over character limit (280 for tweet)
        if len(main_text) > 275:
            # Fall back to score + shorter pitcher lines
            short_summaries = []
            for c in cards:
                parts = c["summary"].split(", ")
                # Keep name + IP + K at minimum
                short = ", ".join(parts[:3]) if len(parts) >= 3 else c["summary"]
                short_summaries.append(short)
            main_text = (
                f"Reds Pitching Recap \u2014 {display_date}\n"
                f"{score_line}\n\n"
                + "\n".join(short_summaries)
                + "\n\n@TJStats #Reds #MLB"
            )

        main_card = cards[0]
        replies: list[PostContent] = []

        # Build replies for each subsequent pitcher
        for card in cards[1:]:
            # Card reply
            card_reply = PostContent(
                text=card["summary"],
                image_path=card["card_path"],
                alt_text=f"Game summary card for {card['name']}",
                tags=["reds_summary", card["name"]],
            )
            replies.append(card_reply)

            # Video reply (if available)
            if card["video_path"]:
                vid_reply = PostContent(
                    text=f"{card['name']} strikeout clip",
                    video_path=card["video_path"],
                    tags=["reds_summary", "video", card["name"]],
                )
                replies.append(vid_reply)

        # Video for the starter (first pitcher) — added after all card replies
        if main_card["video_path"]:
            starter_vid = PostContent(
                text=f"{main_card['name']} strikeout clip",
                video_path=main_card["video_path"],
                tags=["reds_summary", "video", main_card["name"]],
            )
            # Insert at position 0 so starter's video comes right after main tweet
            replies.insert(0, starter_vid)

        return PostContent(
            text=main_text,
            image_path=main_card["card_path"],
            alt_text=f"Game summary card for {main_card['name']}",
            tags=["reds_summary", date_str],
            replies=replies,
        )

    def _get_game_info(self, date_str: str) -> dict | None:
        """Check MLB Stats API schedule for a Reds game on the given date.

        Returns dict with game_pk, opponent, is_home, score_line or None.
        """
        url = (
            f"{MLB_API_BASE}/schedule"
            f"?sportId=1&date={date_str}&teamId={REDS_TEAM_ID}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("MLB schedule API request failed for %s", date_str,
                        exc_info=True)
            return None

        dates = data.get("dates", [])
        if not dates:
            return None

        games = dates[0].get("games", [])
        if not games:
            return None

        game = games[0]
        game_pk = game.get("gamePk")
        status = game.get("status", {}).get("detailedState", "")

        # Only process completed games
        if "Final" not in status and "Completed" not in status:
            log.info("Game %d status: %s (not final)", game_pk, status)
            return None

        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})

        away_team = away.get("team", {}).get("name", "Unknown")
        home_team = home.get("team", {}).get("name", "Unknown")
        away_score = away.get("score", 0)
        home_score = home.get("score", 0)

        is_home = home.get("team", {}).get("id") == REDS_TEAM_ID

        if is_home:
            opponent = away_team
            score_line = f"Reds {home_score}, {away_team} {away_score}"
        else:
            opponent = home_team
            score_line = f"{home_team} {home_score}, Reds {away_score}"

        # Simplify opponent name (e.g. "San Francisco Giants" -> "SF Giants")
        opp_abbrev = _TEAM_NAME_TO_ABBREV.get(opponent, "")
        if opp_abbrev:
            # Use abbreviation + last word of team name
            opp_parts = opponent.split()
            opponent_short = f"{opp_abbrev} {opp_parts[-1]}" if opp_parts else opponent
        else:
            opponent_short = opponent

        return {
            "game_pk": game_pk,
            "opponent": opponent_short,
            "is_home": is_home,
            "score_line": score_line,
            "away_score": away_score,
            "home_score": home_score,
        }
