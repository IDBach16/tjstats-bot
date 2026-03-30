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
from ..charts import plot_reds_game_summary, plot_reds_matchup_header
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
        import os
        override = os.environ.get("REDS_GAME_DATE")
        target = date.fromisoformat(override) if override else date.today() - timedelta(days=1)
        date_str = target.strftime("%Y-%m-%d")
        display_date = target.strftime("%m/%d/%Y")
        log.info("Checking for Reds game on %s", date_str)

        # ── 1. Check MLB schedule for yesterday's game ─────────────
        game_info = self._get_game_info(date_str)
        if game_info is None:
            log.info("No Reds game found for %s", date_str)
            return PostContent(
                text=f"No Reds game yesterday \u2014 {display_date}\n\n@TJStats @PitchProfiler #Reds #MLB",
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
            game_pitchers_df = pd.DataFrame()
            game_pitches_df = pd.DataFrame()
            pbp_df = pd.DataFrame()

        # ── 3. Filter to Reds pitchers for this game ───────────────
        reds_pitchers = pd.DataFrame()
        name_col = None
        id_col = None
        pitcher_ids: set[int] = set()

        gp_col = _find_col(game_pitchers_df, ["game_pk"])
        if gp_col is None:
            for c in game_pitchers_df.columns:
                if "game" in c.lower() and "pk" in c.lower():
                    gp_col = c
                    break

        if gp_col is not None and not game_pitchers_df.empty:
            game_pitchers_df[gp_col] = pd.to_numeric(game_pitchers_df[gp_col], errors="coerce")
            reds_pitchers = game_pitchers_df[game_pitchers_df[gp_col] == game_pk].copy()

            team_col = _find_col(reds_pitchers, ["team", "team_abbreviation", "team_abbrev"])
            if team_col:
                reds_pitchers = reds_pitchers[
                    reds_pitchers[team_col].str.upper() == REDS_TEAM_ABBREV
                ]

            name_col = _find_col(reds_pitchers, ["pitcher_name", "player_name", "name"])
            id_col = _find_col(reds_pitchers, ["pitcher_id", "player_id", "mlbam_id"])

            if id_col and not reds_pitchers.empty:
                reds_pitchers[id_col] = pd.to_numeric(reds_pitchers[id_col], errors="coerce")
                pitcher_ids = set(reds_pitchers[id_col].dropna().astype(int))

        # ── 3b. Fallback: use MLB Stats API boxscore if Pitch Profiler is empty
        if reds_pitchers.empty:
            log.info("Pitch Profiler game data empty, falling back to MLB boxscore")
            reds_pitchers, name_col, id_col, pitcher_ids = self._get_boxscore_pitchers(game_pk)

        if reds_pitchers.empty:
            log.warning("No Reds pitchers found for game_pk=%d", game_pk)
            return PostContent(
                text=f"Reds {score_line} vs {opponent} ({display_date})\n\n"
                     f"Pitcher data not yet available.\n\n@TJStats @PitchProfiler #Reds #MLB",
                tags=["reds_summary", "no_pitchers"],
            )

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

        # ── 4b. Fallback: build pitch data from MLB play-by-play if empty
        if reds_pitches.empty and pitcher_ids:
            log.info("No Pitch Profiler pitch data, falling back to MLB play-by-play")
            reds_pitches = self._get_pbp_pitches(game_pk, pitcher_ids)
            if not reds_pitches.empty:
                game_pitch_id_col = "pitcher_id"
                log.info("MLB PBP fallback: %d pitch rows", len(reds_pitches))

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
                     f"Card generation failed.\n\n@TJStats @PitchProfiler #Reds #MLB",
                tags=["reds_summary", "card_error"],
            )

        # ── 6. Generate matchup header image ────────────────────────
        opp_abbrev = _TEAM_NAME_TO_ABBREV.get(game_info.get("opponent_full", ""), "")
        if not opp_abbrev:
            # Try to extract from short opponent name
            for full_name, abbrev in _TEAM_NAME_TO_ABBREV.items():
                if abbrev in opponent or opponent in full_name:
                    opp_abbrev = abbrev
                    break
        if not opp_abbrev:
            opp_abbrev = opponent.split()[-1][:3].upper()

        starter_name = cards[0]["name"] if cards else "Unknown"
        header_image = plot_reds_matchup_header(
            opponent_abbrev=opp_abbrev,
            game_date=display_date,
            starter_name=starter_name,
            num_pitchers=len(cards),
            score_line=score_line,
            is_home=game_info.get("is_home", True),
        )

        # ── 7. Build thread ────────────────────────────────────────
        main_text = (
            f"Reds Pitching Summary Thread\n"
            f"CIN vs {opponent} \u2014 {display_date}\n"
            f"{score_line}\n\n"
            f"Starter: {starter_name} | {len(cards)} pitchers used\n\n"
            f"@TJStats @PitchProfiler #Reds #MLB"
        )

        # Truncate if over character limit
        if len(main_text) > 275:
            main_text = (
                f"Reds Pitching Summary\n"
                f"{score_line} \u2014 {display_date}\n\n"
                f"Starter: {starter_name}\n\n"
                f"@TJStats @PitchProfiler #Reds #MLB"
            )

        replies: list[PostContent] = []

        # Build replies for each pitcher (card + video)
        for card in cards:
            card_reply = PostContent(
                text=card["summary"],
                image_path=card["card_path"],
                alt_text=f"Game summary card for {card['name']}",
                tags=["reds_summary", card["name"]],
            )
            replies.append(card_reply)

            if card["video_path"]:
                vid_reply = PostContent(
                    text=f"{card['name']} strikeout clip",
                    video_path=card["video_path"],
                    tags=["reds_summary", "video", card["name"]],
                )
                replies.append(vid_reply)

        return PostContent(
            text=main_text,
            image_path=header_image or (cards[0]["card_path"] if cards else None),
            alt_text="Reds matchup header",
            tags=["reds_summary", date_str],
            replies=replies,
        )

    def _get_pbp_pitches(self, game_pk: int, pitcher_ids: set[int]) -> pd.DataFrame:
        """Build pitch-type summary from MLB Stats API play-by-play."""
        url = f"{MLB_API_BASE}/game/{game_pk}/playByPlay"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("MLB PBP request failed for %d", game_pk, exc_info=True)
            return pd.DataFrame()

        # Pitch type code -> name mapping
        _PT_NAMES = {
            "FF": "4-Seam Fastball", "SI": "Sinker", "FC": "Cutter",
            "SL": "Slider", "CU": "Curveball", "CH": "Changeup",
            "FS": "Splitter", "KC": "Knuckle Curve", "ST": "Sweeper",
            "SV": "Slurve", "KN": "Knuckleball", "EP": "Eephus",
            "SC": "Screwball",
        }

        rows = []
        for play in data.get("allPlays", []):
            pitcher_id = play.get("matchup", {}).get("pitcher", {}).get("id")
            if pitcher_id not in pitcher_ids:
                continue
            for ev in play.get("playEvents", []):
                if not ev.get("isPitch"):
                    continue
                pitch_data = ev.get("pitchData", {})
                details = ev.get("details", {})
                pt_code = details.get("type", {}).get("code", "")
                if not pt_code or pt_code in ("PO", "IN", "AB", "AS", "UN", "XX", "NP"):
                    continue
                coords = pitch_data.get("coordinates", {})
                breaks = pitch_data.get("breaks", {})
                rows.append({
                    "pitcher_id": pitcher_id,
                    "pitch_type": pt_code,
                    "pitch_type_name": _PT_NAMES.get(pt_code, pt_code),
                    "velocity": pitch_data.get("startSpeed"),
                    "ivb": breaks.get("breakVerticalInduced"),
                    "hb": breaks.get("breakHorizontal"),
                    "spin_rate": breaks.get("spinRate"),
                    "px": coords.get("pX"),
                    "pz": coords.get("pZ"),
                    "description": details.get("description", ""),
                    "is_strike": details.get("isStrike", False),
                    "is_ball": details.get("isBall", False),
                    "is_in_play": details.get("isInPlay", False),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Build aggregated pitch-type stats per pitcher
        # The card expects columns: pitch_type, velocity, ivb, hb, spin_rate,
        # percentage_thrown, whiff_rate, chase_percentage, stuff_plus, woba, run_value_per_100_pitches
        result_rows = []
        for pid in pitcher_ids:
            pdf = df[df["pitcher_id"] == pid]
            if pdf.empty:
                continue
            total = len(pdf)
            for pt, grp in pdf.groupby("pitch_type"):
                n = len(grp)
                swings = grp["description"].str.contains(
                    "Swinging|Foul|In play", case=False, na=False
                ).sum()
                whiffs = grp["description"].str.contains(
                    "Swinging Strike|Missed Bunt", case=False, na=False
                ).sum()
                whiff_rate = whiffs / swings if swings > 0 else 0
                result_rows.append({
                    "pitcher_id": pid,
                    "pitch_type": pt,
                    "velocity": grp["velocity"].dropna().mean(),
                    "ivb": grp["ivb"].dropna().mean(),
                    "hb": grp["hb"].dropna().mean(),
                    "spin_rate": grp["spin_rate"].dropna().mean(),
                    "percentage_thrown": n / total,
                    "whiff_rate": whiff_rate,
                    "chase_percentage": 0,
                    "stuff_plus": None,
                    "woba": None,
                    "run_value_per_100_pitches": None,
                })

        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame()

    def _get_boxscore_pitchers(self, game_pk: int):
        """Fetch Reds pitchers from MLB Stats API boxscore as a fallback."""
        url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            log.warning("MLB boxscore request failed for %d", game_pk, exc_info=True)
            return pd.DataFrame(), None, None, set()

        teams = data.get("teams", {})
        # Find which side is the Reds
        reds_side = None
        for side in ("home", "away"):
            team_data = teams.get(side, {})
            team_id = team_data.get("team", {}).get("id")
            if team_id == REDS_TEAM_ID:
                reds_side = side
                break

        if not reds_side:
            return pd.DataFrame(), None, None, set()

        pitchers_ids = teams[reds_side].get("pitchers", [])
        players = teams[reds_side].get("players", {})

        rows = []
        for pid in pitchers_ids:
            player_data = players.get(f"ID{pid}", {})
            if not player_data:
                continue
            person = player_data.get("person", {})
            stats = player_data.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            rows.append({
                "pitcher_name": person.get("fullName", "Unknown"),
                "pitcher_id": pid,
                "innings_pitched": stats.get("inningsPitched", "0"),
                "earned_runs": stats.get("earnedRuns", 0),
                "strike_outs": stats.get("strikeOuts", 0),
                "walks": stats.get("baseOnBalls", 0),
                "hits_allowed": stats.get("hits", 0),
                "pitches_thrown": stats.get("pitchesThrown", 0),
                "strikes": stats.get("strikes", 0),
                "home_runs": stats.get("homeRuns", 0),
            })

        if not rows:
            return pd.DataFrame(), None, None, set()

        df = pd.DataFrame(rows)
        pitcher_ids = set(df["pitcher_id"].astype(int))
        log.info("Boxscore fallback: found %d Reds pitchers", len(df))
        return df, "pitcher_name", "pitcher_id", pitcher_ids

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
            "opponent_full": opponent,
            "is_home": is_home,
            "score_line": score_line,
            "away_score": away_score,
            "home_score": home_score,
        }
