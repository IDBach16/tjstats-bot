"""Content generator: Best pitcher outing from last night.

Finds the top pitching performance from yesterday's games using
Pitch Profiler 2026 data, generates a pitching summary card with
movement plot + percentiles, and fetches Savant video.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import anthropic
import os
import pandas as pd
import requests

from .base import ContentGenerator, PostContent
from .. import pitch_profiler
from ..charts import plot_pitching_summary
from ..config import DEFAULT_HASHTAGS, MLB_SEASON, MLB_API_BASE
from ..video_clips import get_game_strikeout_clip

log = logging.getLogger(__name__)


def _get_yesterday_game_pks() -> list[int]:
    """Get all game PKs from yesterday via MLB Stats API."""
    yesterday = date.today() - timedelta(days=1)
    url = f"{MLB_API_BASE}/schedule?sportId=1&date={yesterday.strftime('%Y-%m-%d')}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pks = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("detailedState", "")
                if "Final" in status or "Completed" in status:
                    pks.append(g["gamePk"])
        return pks
    except Exception:
        log.warning("Failed to get yesterday's schedule", exc_info=True)
        return []


def _generate_analysis(name: str, stats_str: str) -> str | None:
    """Generate 2-sentence analysis of the outing."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content":
                f"Write exactly 2 sentences analyzing {name}'s pitching outing last night. "
                f"Stats: {stats_str}. "
                f"Be sharp and insightful. Reference specific stats. "
                f"No hashtags, emojis, @, dashes, hyphens, or character counts. "
                f"Do NOT start with the pitcher's name. "
                f"Output ONLY the 2 sentences."
            }],
        )
        text = message.content[0].text.strip()
        text = re.sub(r'\s*\(\d+ characters?\)\s*$', '', text)
        return text
    except Exception:
        log.warning("Analysis generation failed", exc_info=True)
        return None


class BestOutingGenerator(ContentGenerator):
    name = "best_outing"

    async def generate(self) -> PostContent:
        yesterday = date.today() - timedelta(days=1)
        display_date = yesterday.strftime("%m/%d/%Y")

        # Get yesterday's game PKs
        game_pks = _get_yesterday_game_pks()
        if not game_pks:
            log.info("No completed games found for yesterday")
            return PostContent(text="")

        # Get game-level pitch data from Pitch Profiler
        try:
            game_pitches_df = pitch_profiler.get_game_pitches(MLB_SEASON)
        except Exception:
            log.warning("Failed to fetch Pitch Profiler game pitches", exc_info=True)
            return PostContent(text="")

        # Filter to regular season + yesterday's games
        reg = game_pitches_df[game_pitches_df["game_type"] == "R"]
        yesterday_pitches = reg[reg["game_pk"].isin(game_pks)]

        if yesterday_pitches.empty:
            log.info("No pitch data for yesterday's games")
            return PostContent(text="")

        # Aggregate by pitcher per game — find best outing
        pitcher_games = yesterday_pitches.groupby(
            ["game_pk", "pitcher_name", "pitcher_id"]
        ).agg(
            total_pitches=("thrown", "sum"),
            total_whiffs=("whiff", "sum"),
            total_swings=("swings", "sum"),
            total_k_csw=("called_strikes_plus_whiffs", "sum"),
            avg_stuff_plus=("stuff_plus", "mean"),
            avg_pitching_plus=("pitching_plus", "mean"),
            n_pitch_types=("pitch_type", "nunique"),
        ).reset_index()

        # Filter: at least 40 pitches (roughly 3+ innings for starters)
        starters = pitcher_games[pitcher_games["total_pitches"] >= 40].copy()
        if starters.empty:
            # Fallback: at least 15 pitches
            starters = pitcher_games[pitcher_games["total_pitches"] >= 15].copy()

        if starters.empty:
            log.info("No qualifying pitcher outings found")
            return PostContent(text="")

        starters["whiff_rate"] = starters["total_whiffs"] / starters["total_swings"].replace(0, 1)

        # Score: weighted combo of stuff+, whiff rate, and pitches thrown
        starters["score"] = (
            starters["avg_stuff_plus"] * 0.4
            + starters["whiff_rate"] * 100 * 0.3
            + starters["avg_pitching_plus"] * 0.3
        )

        # Skip recently posted pitchers
        from ..scheduler import was_recently_posted
        starters = starters[
            ~starters["pitcher_name"].apply(lambda n: was_recently_posted(n, lookback=10))
        ]

        if starters.empty:
            log.info("All top outings were recently posted")
            return PostContent(text="")

        best = starters.sort_values("score", ascending=False).iloc[0]
        name = best["pitcher_name"]
        pid = int(best["pitcher_id"])
        game_pk = int(best["game_pk"])

        log.info("Best outing: %s (pk=%d, stuff+=%.0f, whiff=%.1f%%)",
                 name, game_pk, best["avg_stuff_plus"], best["whiff_rate"] * 100)

        # Get game-level pitch data for the card (not season)
        game_pitcher_pitches = yesterday_pitches[
            (yesterday_pitches["game_pk"] == game_pk) &
            (yesterday_pitches["pitcher_name"] == name)
        ].copy()

        # Build a game-level pitcher row for the card header
        game_row = pd.Series({
            "pitcher_name": name,
            "pitcher_id": pid,
            "total_pitches": int(best["total_pitches"]),
            "whiff_rate": best["whiff_rate"],
            "stuff_plus": best["avg_stuff_plus"],
            "pitching_plus": best["avg_pitching_plus"],
            "n_pitch_types": int(best["n_pitch_types"]),
        })

        # Get team + season data for context (subtitle only)
        team = None
        season_df = pd.DataFrame()
        try:
            season_df = pitch_profiler.get_season_pitchers(MLB_SEASON)
            if "season_teams" in season_df.columns:
                match = season_df[season_df["pitcher_name"] == name]
                if not match.empty:
                    team = str(match.iloc[0].get("season_teams", "")).split(",")[0].strip()
        except Exception:
            log.warning("Failed to fetch season data for team lookup", exc_info=True)

        # Fetch season-level pitch data for league comparison in the card
        try:
            all_pitches_df = pitch_profiler.get_season_pitches(MLB_SEASON)
        except Exception:
            all_pitches_df = None

        # Generate card using game-level data
        game_pitcher_df = pd.DataFrame([game_row])
        image_path = plot_pitching_summary(
            name, game_pitcher_df, game_pitcher_pitches,
            all_pitches_df=all_pitches_df,
            team=team, player_id=pid, level="MLB",
        )
        if not image_path:
            log.warning("Card generation failed for %s", name)
            return PostContent(text="")

        # Build game-specific stats string for the AI prompt
        stats_str = (
            f"{int(best['total_pitches'])} pitches, "
            f"Stuff+ {best['avg_stuff_plus']:.0f}, "
            f"Pitching+ {best['avg_pitching_plus']:.0f}, "
            f"{best['whiff_rate']*100:.1f}% whiff rate, "
            f"{int(best['n_pitch_types'])} pitch types"
        )

        # Generate analysis (game stats only, no season)
        analysis = _generate_analysis(name, stats_str)

        # Get strikeout video clip
        video_path = None
        try:
            video_path = get_game_strikeout_clip(game_pk, pid, name)
        except Exception:
            log.warning("Video clip failed for %s", name, exc_info=True)

        # Build tweet
        if analysis:
            text = (
                f"{analysis}\n\n"
                f"{name} — Best Outing {display_date}\n\n"
                f"@TJStats @PitchProfiler {DEFAULT_HASHTAGS}"
            )
        else:
            text = (
                f"{name} — Best Outing {display_date}\n"
                f"{stats_str}\n\n"
                f"@TJStats @PitchProfiler {DEFAULT_HASHTAGS}"
            )

        return PostContent(
            text=text,
            image_path=image_path,
            video_path=video_path,
            alt_text=f"Pitching summary for {name}",
            tags=["best_outing", name],
        )
