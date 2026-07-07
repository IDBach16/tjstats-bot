"""Content generator: MLB Draft prospect pitcher card.

Renders a Statcast prospect card for a standout NCAA D1 pitcher using
Hawk-Eye tracking from the tracked college season, in the Reds
game-summary layout (+ release-point plot & vs-league percentile panel).
Prefers pitchers on Lance Brozdowski's 2026 Draft rankings when they have
tracked data (crediting @LanceBroz), and threads a YouTube highlight clip.
"""

from __future__ import annotations

import logging

from .base import ContentGenerator, PostContent
from ..college_statcast import (
    NCAA_LEAGUE,
    college_season_window,
    download_youtube_clip,
    fetch_college_window,
    find_player_video,
    get_college_pitchers,
    get_college_pitches,
    pick_college_prospect,
)
from ..charts import plot_draft_prospect_card
from ..config import DEFAULT_HASHTAGS, MLB_SEASON

log = logging.getLogger(__name__)

_MIN_PITCHES = 40
# Upload the highlight natively (download + attach) vs. just linking it.
# Native upload re-hosts third-party footage — flip to False to post the
# YouTube link instead if that's a concern.
_NATIVE_VIDEO = True


class DraftProspectGenerator(ContentGenerator):
    name = "draft_prospect"

    async def generate(self) -> PostContent:
        # NCAA D1 tracked games across the full college season — this is
        # the domain of the draft pitching rankings we lean on.
        start, end = college_season_window(MLB_SEASON)
        pool = {NCAA_LEAGUE}

        # Don't repeat a prospect we've already featured. Pull the names of
        # previously-posted draft prospects (tags[1]) and exclude them so the
        # deterministic "highest-ranked" pick advances to the next arm.
        # (local import: scheduler imports this generator, so avoid a cycle)
        from ..scheduler import recent_generator_tags
        already_posted = recent_generator_tags(self.name)
        if already_posted:
            log.info("Excluding %d already-posted prospect(s): %s",
                     len(already_posted), sorted(already_posted))

        prospect = pick_college_prospect(
            start, end, min_pitches=_MIN_PITCHES, leagues=pool,
            exclude_names=already_posted)
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
        rank = prospect.get("rank")
        fv = prospect.get("fv")
        ranked_source = prospect.get("ranked_source")

        season_df = get_college_pitchers(
            start, end, min_pitches=_MIN_PITCHES, leagues=pool)
        pitches_df = get_college_pitches(start, end, leagues=pool)
        if season_df.empty or pitches_df.empty:
            log.warning("No data to render prospect card for %s", name)
            return PostContent(text="")

        pid = prospect.get("id")
        if pid is None:
            log.warning("Prospect %s has no player_id", name)
            return PostContent(text="")

        # This player's slices: season aggregate row, per-pitch-type rows,
        # and raw pitches (for the location/result/release scatter plots).
        srow = season_df[season_df["player_id"] == pid]
        prows = pitches_df[pitches_df["player_id"] == pid]
        if srow.empty or prows.empty:
            log.warning("No aggregated data for prospect %s", name)
            return PostContent(text="")
        season_row = srow.iloc[0]
        p_throws = str(season_row.get("p_throws", "R") or "R")

        raw = fetch_college_window(start, end, leagues=pool)
        pbp_df = raw[raw["pitcher"] == pid] if not raw.empty else None

        # Reds game-summary layout + release plot + percentile panel; the
        # league-wide aggregates drive the vs-league colouring/percentiles.
        image_path = plot_draft_prospect_card(
            name, season_row, prows, pbp_df=pbp_df,
            league_pitches_df=pitches_df, league_season_df=season_df,
            p_throws=p_throws, team=team, team_name=team_name, league=league,
            league_label=league_label, league_full=league, season=MLB_SEASON,
            rank=rank, fv=fv, ranked_source=ranked_source,
        )
        if not image_path:
            log.warning("Draft prospect card render failed for %s", name)
            return PostContent(text="")

        # Tweet copy — headline the ranking when the arm is on the list.
        subtitle = f" ({team_name})" if team_name else ""
        if rank is not None:
            headline = (f"{name}{subtitle} — #{rank} on {ranked_source}'s "
                        f"{MLB_SEASON} MLB Draft Big Board")
            credit = f"Rankings via {ranked_source}.\n"
        else:
            headline = f"{name}{subtitle} — {MLB_SEASON} MLB Draft Prospect Card"
            credit = ""
        text = (
            f"{headline}\n\n"
            f"Full arsenal + Hawk-Eye tracking from the college season.\n"
            f"{credit}\n"
            f"@TJStats {DEFAULT_HASHTAGS} #MLBDraft #CollegeBaseball"
        )

        # Thread a highlight clip if we can find one. For a player-named
        # clip we try to upload it natively (download + transcode); if that
        # fails, or it's only a team highlight, we fall back to the link.
        replies = []
        found = find_player_video(name, team_name, extra="baseball highlights")
        if found:
            video_url, kind = found
            clip = None
            if kind == "player" and _NATIVE_VIDEO:
                clip = download_youtube_clip(video_url, name, max_seconds=30)
            if clip:
                replies.append(PostContent(
                    text=f"🎥 {name} — highlights\n(clip via YouTube: {video_url})",
                    video_path=clip, tags=["draft_prospect_video"]))
                log.info("Draft prospect NATIVE video for %s: %s", name, clip)
            else:
                label = (f"Watch {name}" if kind == "player"
                         else f"{team_name} highlights")
                replies.append(PostContent(
                    text=f"🎥 {label}: {video_url}",
                    tags=["draft_prospect_video"]))
                log.info("Draft prospect video LINK (%s) for %s: %s",
                         kind, name, video_url)

        return PostContent(
            text=text,
            image_path=image_path,
            alt_text=f"{MLB_SEASON} MLB Draft prospect pitcher card for {name}",
            tags=["draft_prospect", name, team_name, league_label],
            replies=replies,
        )
