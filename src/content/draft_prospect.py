"""Content generator: MLB Draft prospect pitcher card.

Renders a Statcast prospect card for a standout amateur pitcher using
Hawk-Eye tracking from whichever draft showcase is currently being played
and tracked — NCAA D1 in spring, the Cape Cod / MLB Draft / Appalachian
leagues in summer — and threads a YouTube highlight clip as a reply
(amateur Savant has no video of its own).
"""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..college_statcast import (
    default_window,
    find_player_video,
    get_college_pitchers,
    get_college_pitches,
    pick_college_prospect,
    pick_target_league,
)
from ..charts import plot_milb_pitcher_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

# Full-name subtitle per league for the tweet copy.
_LEAGUE_NAMES = {
    "Cape Cod Baseball League": "Cape Cod League",
    "MLB Draft League": "MLB Draft League",
    "College Baseball": "NCAA D1",
    "Appalachian League": "Appalachian League",
    "Northwoods League": "Northwoods League",
    "Prospect League": "Prospect League",
}
_MIN_PITCHES = 40


class DraftProspectGenerator(ContentGenerator):
    name = "draft_prospect"

    async def generate(self) -> PostContent:
        start, end = default_window()

        # Feature the most-scouted showcase with a usable percentile pool.
        target = pick_target_league(start, end, min_pitches=_MIN_PITCHES)
        leagues = {target} if target else None

        prospect = pick_college_prospect(
            start, end, min_pitches=_MIN_PITCHES, leagues=leagues)
        if not prospect:
            log.warning("No qualified prospect found in %s..%s", start, end)
            return PostContent(text="")

        name = prospect["name"]
        # Savant hands out "Last, First" — normalise to "First Last".
        if "," in name:
            parts = name.split(", ")
            if len(parts) == 2:
                name = f"{parts[1]} {parts[0]}"

        team = prospect.get("team", "")
        team_name = prospect.get("team_name", "") or team
        league = prospect.get("league", "")
        league_label = prospect.get("league_label", "COLLEGE")
        league_full = _LEAGUE_NAMES.get(league, "college baseball")

        # Same-league pool → fair percentiles + correct arsenal.
        pool = {league} if league else None
        season_df = get_college_pitchers(
            start, end, min_pitches=_MIN_PITCHES, leagues=pool)
        pitches_df = get_college_pitches(start, end, leagues=pool)
        if season_df.empty or pitches_df.empty:
            log.warning("No data to render prospect card for %s", name)
            return PostContent(text="")

        # Fix the display name in both frames (match on player_id).
        pid = prospect.get("id")
        if pid is not None:
            season_df = season_df.copy()
            pitches_df = pitches_df.copy()
            season_df.loc[season_df["player_id"] == pid, "pitcher_name"] = name
            pitches_df.loc[pitches_df["player_id"] == pid, "pitcher_name"] = name

        # player_id passed as None: amateurs have no MLB headshot and their
        # id would make the card resolve a bogus MLB "current team".
        image_path = plot_milb_pitcher_card(
            name, season_df, pitches_df,
            team=team, player_id=None, level=league_label,
        )
        if not image_path:
            log.warning("Draft prospect card render failed for %s", name)
            return PostContent(text="")

        subtitle = f" ({team_name})" if team_name else ""
        text = (
            f"{name}{subtitle} — {MLB_SEASON} MLB Draft Prospect Card\n\n"
            f"Full arsenal + Hawk-Eye tracking from the {league_full}.\n\n"
            f"@TJStats {DEFAULT_HASHTAGS} #MLBDraft #CollegeBaseball"
        )

        # Thread a highlight clip if we can find one — worded honestly
        # depending on whether it's the player or just their team.
        reply = None
        found = find_player_video(name, team_name, extra="baseball highlights")
        if found:
            video_url, kind = found
            if kind == "player":
                rtext = f"🎥 Watch {name}: {video_url}"
            else:
                rtext = f"🎥 {team_name} highlights: {video_url}"
            reply = PostContent(text=rtext, tags=["draft_prospect_video"])
            log.info("Draft prospect video (%s) for %s: %s",
                     kind, name, video_url)

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"{MLB_SEASON} MLB Draft prospect pitcher card for {name}",
            tags=["draft_prospect", name, team_name, league_label],
            reply=reply,
        )
